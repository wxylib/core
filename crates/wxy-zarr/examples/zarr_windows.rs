//! Read windows of one array from a Zarr hierarchy and compare each with
//! the GDAL SHA-256 in a manifest: the native-Zarr form of wxy-virtual's
//! `refs_windows`.
//!
//! The manifest is `{"array": NAME, "oracle": {WINDOW: {"start", "count",
//! "sha256"}}}`, as wxylib/fixtures `scripts/oracle_zarr_windows.py` writes
//! it (GDAL multidim on `ZARR:"/vsicurl/URL"`, the hash over raw
//! little-endian values). The store is an http(s) URL of the hierarchy's
//! root, read by GETs alone (`wxy_zarr::open_url`), or a local directory.
//!
//! Copernicus Marine ARCO, global physics reanalysis, daily:
//!
//! ```sh
//! cargo run --release -p wxy-zarr --example zarr_windows -- \
//!   https://s3.waw3-1.cloudferro.com/mdl-arco-time-025/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/timeChunked.zarr \
//!   --manifest glorys-thetao.json
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
    let manifest: Value = serde_json::from_reader(
        std::fs::File::open(opt("--manifest").expect("--manifest FILE")).unwrap(),
    )
    .unwrap();
    let location = args
        .iter()
        .enumerate()
        .find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--manifest"))
        .map(|(_, a)| a.clone())
        .expect("a Zarr location");
    let name = manifest["array"].as_str().unwrap();

    let t0 = Instant::now();
    let ds = if wxy_zarr::is_zarr_url(&location) {
        wxy_zarr::open_url(&location, name).await
    } else {
        wxy_zarr::open_path(&location, name).await
    }
    .unwrap();
    let grid = ds.source.grid();
    let nmap = ds.collection.domains[0].mappings.len();
    let dims: Vec<&str> = ds.collection.domains[0]
        .dims
        .iter()
        .map(|d| d.name.as_str())
        .collect();
    println!(
        "{name}: dims {dims:?}, {} mappings, shape {:?}, chunks {:?}, opened in {:.2} s",
        nmap,
        grid.shape,
        grid.chunk,
        t0.elapsed().as_secs_f64()
    );

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
            "{w}: {} GDAL ({:.2} s) sha256 {got}",
            if got == want {
                "identical to"
            } else {
                "DIFFERS from"
            },
            t0.elapsed().as_secs_f64()
        );
    }
    if !ok {
        std::process::exit(1);
    }
}
