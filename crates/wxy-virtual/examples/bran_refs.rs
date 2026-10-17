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
//! ```sh
//! cargo run --release -p wxy-virtual --example bran_refs -- ../fixtures/mdimrefs/bran2023-temp-201001
//! ```

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wxy_source::{read_window, ChunkSource, Window};

#[tokio::main]
async fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../fixtures/mdimrefs/bran2023-temp-201001"));
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(dir.join("manifest.json")).unwrap()).unwrap();

    let arrays = wxy_virtual::from_refs_table(&dir, None).unwrap();
    let prefixes = wxy_virtual::prefixes(&arrays);
    let refs: usize = arrays.iter().map(|a| a.refs.len()).sum();
    println!(
        "{} arrays, {refs} references, chunk containers {prefixes:?}",
        arrays.len()
    );

    let repo = std::env::temp_dir().join(format!("wxy-bran-refs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    let t0 = Instant::now();
    wxy_virtual::write_icechunk(&repo, &arrays).await.unwrap();
    println!("Icechunk written in {:.2} s", t0.elapsed().as_secs_f64());
    let storage = wxy_virtual::open_icechunk(&repo, &prefixes).await.unwrap();
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

    let mut ok = true;
    for (name, o) in manifest["oracle"].as_object().unwrap() {
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
        if let Some(want) = o["sha256"].as_str() {
            let got: String = Sha256::digest(buf.values.to_le_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
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
                "{name}: {} valid, raw {lo} to {hi} ({secs:.2} s; no GDAL SHA-256 in the manifest)",
                valid.len()
            );
        }
    }
    let _ = std::fs::remove_dir_all(&repo);
    if !ok {
        std::process::exit(1);
    }
}
