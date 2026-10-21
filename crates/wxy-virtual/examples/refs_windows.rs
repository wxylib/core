//! Read windows of one array from a reference store and compare each with
//! the GDAL SHA-256 in a manifest: the generic form of `bran_refs`.
//!
//! The manifest is `{"array": NAME, "oracle": {WINDOW: {"start", "count",
//! "sha256"}}}`, as raad.virtualize `py/oracle_windows.py` writes it (GDAL
//! multidim on the store's plan VRT, the hash over raw little-endian
//! values). The store is any location `wxy_refs::open_refs_at` takes: a flat
//! mdim-refs export (`arrays.json` beside `refs_<array>.parquet`) or a
//! kerchunk Parquet directory, local, `s3://`, `https://` or `/vsis3/`.
//!
//! Read paths:
//! - default: the table written to a local Icechunk repository by wxylib,
//!   then read through it;
//! - `--direct`: the table served to zarrs by `wxy_refs::RefsStore`;
//! - `--icechunk REPO`: an existing local Icechunk repository (such as the
//!   one raad.virtualize `py/render_icechunk.py` writes), opened with the
//!   container prefixes of the table, which is still needed for those.
//!
//! `--target T` applies the table's remap target `T` to the source paths.
//!
//! GHRSST MUR 2024 (366 COGs, ZSTD with the TIFF horizontal predictor,
//! read through `HorizontalDeltaCodec` from wxy-predictor), 2026-10-21:
//!
//! ```sh
//! cargo run --release -p wxy-virtual --features refs --example refs_windows -- \
//!   work/out/flat --manifest work/manifest.json --target s3direct --direct
//! ```

use std::time::Instant;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wxy_source::{read_window, ChunkSource, Window};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let direct = args.iter().any(|a| a == "--direct");
    let target = opt("--target");
    let ice = opt("--icechunk");
    let manifest: Value = serde_json::from_reader(
        std::fs::File::open(opt("--manifest").expect("--manifest FILE")).unwrap(),
    )
    .unwrap();
    let valued = ["--manifest", "--target", "--icechunk"];
    let location = args
        .iter()
        .enumerate()
        .find(|(i, a)| !a.starts_with("--") && (*i == 0 || !valued.contains(&args[i - 1].as_str())))
        .map(|(_, a)| a.clone())
        .expect("a table location");
    let name = manifest["array"].as_str().unwrap();

    let t0 = Instant::now();
    let arrays = wxy_refs::open_refs_at(&location, target.as_deref())
        .await
        .unwrap();
    let prefixes = wxy_virtual::prefixes(&arrays);
    let refs: usize = arrays.iter().map(|a| a.refs.len()).sum();
    println!(
        "table read in {:.2} s: {} arrays, {refs} references, containers {prefixes:?}",
        t0.elapsed().as_secs_f64(),
        arrays.len()
    );

    let tmp = std::env::temp_dir().join(format!("wxy-refs-windows-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let t0 = Instant::now();
    let (storage, how): (
        std::sync::Arc<dyn zarrs::storage::AsyncReadableStorageTraits>,
        _,
    ) = if direct {
        let s = wxy_refs::RefsStore::new(&arrays, None).unwrap();
        (std::sync::Arc::new(s), "refs store".to_string())
    } else if let Some(repo) = &ice {
        let s = wxy_virtual::open_icechunk(repo, &prefixes).await.unwrap();
        (s, format!("Icechunk {repo}"))
    } else {
        wxy_virtual::write_icechunk(&tmp, &arrays).await.unwrap();
        let s = wxy_virtual::open_icechunk(&tmp, &prefixes).await.unwrap();
        (s, "Icechunk written by wxylib".to_string())
    };
    println!("{how} ready in {:.2} s", t0.elapsed().as_secs_f64());
    let ds = wxy_zarr::open(storage, name).await.unwrap();
    let grid = ds.source.grid();
    println!("{name}: shape {:?}, chunks {:?}", grid.shape, grid.chunk);

    let mut ok = true;
    for (w, o) in manifest["oracle"].as_object().unwrap() {
        let v = |k: &str| -> Vec<u64> {
            o[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let t0 = Instant::now();
        let buf = match read_window(&ds.source, &Window::new(v("start"), v("count"))).await {
            Ok(buf) => buf,
            Err(e) => {
                println!("{w}: read failed: {e}");
                ok = false;
                continue;
            }
        };
        let got: String = Sha256::digest(buf.values.to_le_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let want = o["sha256"].as_str().unwrap();
        ok &= got == want;
        println!(
            "{w}: {} GDAL ({:.2} s)",
            if got == want {
                "identical to"
            } else {
                "DIFFERS from"
            },
            t0.elapsed().as_secs_f64()
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
    if !ok {
        std::process::exit(1);
    }
}
