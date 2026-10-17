//! P3 demo (design doc m02): 30 days of real NOAA OISST v2.1 (September
//! 1981) from an mdim-refs table, remapped to the public NOAA bucket over
//! HTTPS, written to a local Icechunk repository as virtual references,
//! and read back for day 30. Each array's bytes are compared with what
//! GDAL read from the source netCDF (SHA-256 in the table's manifest).
//!
//! The table is made by wxylib/fixtures
//! `scripts/make_mdimrefs_oisst_198109.py`. Needs network access; not a
//! test.
//!
//! ```sh
//! cargo run -p wxy-virtual --example oisst_refs -- ../fixtures/mdimrefs/oisst-198109
//! ```
//!
//! With a second argument, day 30 `sst` is also warped to EPSG:3031 (a
//! 400 x 400 grid of 20 km cells, nearest) and written there as raw
//! little-endian int16, to compare with `gdalwarp`: one slice of a warp
//! planned over the whole 4D array (stage S4b).

use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wxy_domain::Domain;
use wxy_source::{read_window, ChunkSource, Window};

/// Warp the (1, 1, y, x) day at `start` of the 4D array to EPSG:3031
/// and write the raw values: one slice of the plan, read from the
/// virtual store directly.
async fn warp_day<S: ChunkSource>(
    source: &S,
    domain: &Domain,
    start: &[u64],
    nodata: Option<f64>,
    out: &std::path::Path,
) {
    let dst = wxy_warp::GridSpec::from_extent("EPSG:3031", [-4e6, 4e6, -4e6, 4e6], [400, 400]);
    let opts = wxy_warp::WarpOptions::default();
    let plan = wxy_warp::WarpPlan::new(domain, source.dtype(), &dst, &opts, nodata).unwrap();
    let lead = plan.leading().len();
    let day = Window::new(start[..lead].to_vec(), vec![1; lead]);
    let w = plan.warp_slices(source, &day).await.unwrap();
    std::fs::write(out, w.values.to_le_bytes()).unwrap();
    println!(
        "sst: day 30 warped to EPSG:3031, 400 x 400, in {}",
        out.display()
    );
}

#[tokio::main]
async fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../fixtures/mdimrefs/oisst-198109"));
    let warp_out = std::env::args().nth(2).map(PathBuf::from);
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(dir.join("manifest.json")).unwrap()).unwrap();
    let target = manifest["remap_target"].as_str();

    let arrays = wxy_virtual::from_refs_table(&dir, target).unwrap();
    let prefixes = wxy_virtual::prefixes(&arrays);
    println!("{} arrays, chunk containers {prefixes:?}", arrays.len());

    let repo = std::env::temp_dir().join(format!("wxy-oisst-refs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    wxy_virtual::write_icechunk(&repo, &arrays).await.unwrap();
    let storage = wxy_virtual::open_icechunk(&repo, &prefixes).await.unwrap();

    let mut ok = true;
    for (name, o) in manifest["oracle"].as_object().unwrap() {
        let v = |k: &str| -> Vec<u64> {
            o[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let ds = wxy_zarr::open(storage.clone(), name).await.unwrap();
        let t0 = std::time::Instant::now();
        let buf = read_window(&ds.source, &Window::new(v("start"), v("count")))
            .await
            .unwrap();
        let got: String = Sha256::digest(buf.values.to_le_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let same = got == o["sha256"].as_str().unwrap();
        ok &= same;
        println!(
            "{name}: day 30 {} GDAL ({:.2} s)",
            if same { "identical to" } else { "DIFFERS from" },
            t0.elapsed().as_secs_f64()
        );
        if let (Some(out), "sst") = (&warp_out, name.as_str()) {
            let d = &ds.collection.domains[0];
            let nodata = ds.collection.fields[0].values.nodata;
            warp_day(&ds.source, d, &v("start"), nodata, out).await;
        }
    }
    let _ = std::fs::remove_dir_all(&repo);
    if !ok {
        std::process::exit(1);
    }
}
