//! 32 days of BRAN2023 `temp` (CSIRO Bluelink) from an mdim-refs table:
//! 51 depths of 1500 x 3600 in 300 x 300 chunks, 97,920 references into
//! two netCDF files on NCI's THREDDS server. The many-chunks-per-file case
//! (OISST, in `oisst_refs`, has one chunk per file).
//!
//! The table is wxylib/fixtures `mdimrefs/bran2023-temp-201001`, converted
//! by `scripts/make_mdimrefs_bran2023_temp.py` from the kerchunk Parquet
//! index in mdsumner/virtualized. It is written to a local Icechunk
//! repository, opened through wxy-zarr, and a warp to EPSG:3031 is planned
//! over every day and depth (printed, not run). Then each oracle window in
//! the manifest is read: compared with GDAL when the manifest has its
//! SHA-256 (`--oracle`), otherwise summarised. Reads need network access to
//! thredds.nci.org.au; not a test.
//!
//! With `--direct` the table is read straight through `wxy_refs::RefsStore`
//! (doc 03, S3c) instead of an Icechunk repository, so the two read paths
//! can be checked against the same GDAL hashes.
//!
//! The table can also be named by URL, such as the `flat/` export of a
//! store on S3, or be a kerchunk Parquet store (`.zmetadata`, such as the
//! same store's `kerchunk/`) (credentials and endpoint from the `AWS_*` environment,
//! GDAL's names included; see `wxy_refs::location`). With no
//! `manifest.json` there (or `--manifest <file>`), two windows are read,
//! the first day's chunk at row 1200 and a row of the last day, and their
//! SHA-256 printed for comparison with GDAL.
//!
//! ```sh
//! cargo run --release -p wxy-virtual --example bran_refs -- ../fixtures/mdimrefs/bran2023-temp-201001
//! cargo run --release -p wxy-virtual --example bran_refs -- ../fixtures/mdimrefs/bran2023-temp-201001 --direct
//! AWS_S3_ENDPOINT=transfer.data.aad.gov.au AWS_VIRTUAL_HOSTING=NO \
//!   cargo run --release -p wxy-virtual --example bran_refs -- \
//!   s3://aadc-transfer/michae_sum/e2e/bran2023-daily-temp/flat --direct
//! cargo run --release -p wxy-virtual --example bran_refs -- \
//!   s3://aadc-transfer/michae_sum/e2e/bran2023-daily-temp/kerchunk --direct
//! ```

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wxy_source::{read_window, ChunkSource, Window};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let direct = args.iter().any(|a| a == "--direct");
    let manifest_arg = args
        .iter()
        .position(|a| a == "--manifest")
        .map(|i| PathBuf::from(&args[i + 1]));
    let dir = args
        .iter()
        .enumerate()
        .find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--manifest"))
        .map(|(_, a)| a.clone())
        .unwrap_or_else(|| "../fixtures/mdimrefs/bran2023-temp-201001".into());
    let manifest_file = manifest_arg.or_else(|| {
        Some(PathBuf::from(&dir).join("manifest.json"))
            .filter(|p| !dir.contains("://") && p.is_file())
    });
    let manifest: Option<Value> =
        manifest_file.map(|f| serde_json::from_reader(std::fs::File::open(f).unwrap()).unwrap());

    let t0 = Instant::now();
    let arrays = wxy_refs::open_refs_at(&dir, None).await.unwrap();
    println!("table read in {:.2} s", t0.elapsed().as_secs_f64());
    let prefixes = wxy_virtual::prefixes(&arrays);
    let refs: usize = arrays.iter().map(|a| a.refs.len()).sum();
    println!(
        "{} arrays, {refs} references, chunk containers {prefixes:?}",
        arrays.len()
    );

    let repo = std::env::temp_dir().join(format!("wxy-bran-refs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    let t0 = Instant::now();
    let storage: std::sync::Arc<dyn zarrs::storage::AsyncReadableStorageTraits> = if direct {
        let s = wxy_refs::RefsStore::new(&arrays, None).unwrap();
        println!("refs store built in {:.2} s", t0.elapsed().as_secs_f64());
        std::sync::Arc::new(s)
    } else {
        wxy_virtual::write_icechunk(&repo, &arrays).await.unwrap();
        println!("Icechunk written in {:.2} s", t0.elapsed().as_secs_f64());
        wxy_virtual::open_icechunk(&repo, &prefixes).await.unwrap()
    };
    let ds = wxy_zarr::open(storage, "temp").await.unwrap();
    let grid = ds.source.grid();
    println!("temp: shape {:?}, chunks {:?}", grid.shape, grid.chunk);

    // The Southern Ocean south of about 45S, in polar stereographic.
    let dst = wxy_warp::GridSpec::from_extent("EPSG:3031", [-4e6, 4e6, -4e6, 4e6], [400, 400]);
    let nodata = ds.collection.fields[0].values.nodata;
    let plan = wxy_warp::WarpPlan::new(
        &ds.collection.domains[0],
        ds.source.dtype(),
        &dst,
        &wxy_warp::WarpOptions::default(),
        nodata,
    )
    .unwrap();
    let e = plan.explain(&ds.source, 1 << 30);
    println!(
        "warp to EPSG:3031 planned over {} slices: {} chunks, {:.1} MB decoded",
        plan.leading().iter().product::<u64>(),
        e.fetch.chunks,
        e.fetch.decoded_bytes as f64 / 1e6
    );

    let windows = match &manifest {
        Some(m) => m["oracle"].clone(),
        None => serde_json::json!({
            "chunk-t0-z0-y4-x0": {"start": [0, 0, 1200, 0], "count": [1, 1, 300, 300]},
            "row-tlast-z10-y750": {"start": [grid.shape[0] - 1, 10, 750, 0], "count": [1, 1, 1, 3600]},
        }),
    };
    let mut ok = true;
    for (name, o) in windows.as_object().unwrap() {
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
                println!("{name}: read failed: {e}");
                ok = false;
                continue;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let got: String = Sha256::digest(buf.values.to_le_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if let Some(want) = o["sha256"].as_str() {
            ok &= got == want;
            let verdict = if got == want {
                "identical to"
            } else {
                "DIFFERS from"
            };
            println!("{name}: {verdict} GDAL ({secs:.2} s)");
        } else {
            let x = buf.values.to_f64();
            let valid: Vec<f64> = x.into_iter().filter(|x| Some(*x) != nodata).collect();
            let lo = valid.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{name}: {} valid, raw {lo} to {hi}, sha256 {got} ({secs:.2} s; no GDAL SHA-256 to compare)",
                valid.len()
            );
        }
    }
    let _ = std::fs::remove_dir_all(&repo);
    if !ok {
        std::process::exit(1);
    }
}
