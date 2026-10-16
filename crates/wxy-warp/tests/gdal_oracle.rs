//! Warps of the COG fixtures compared with GDAL's warper (wxylib/fixtures,
//! warp/, made by scripts/make_warp_fixtures.py).
//!
//! The gate (design doc 03, S4): nearest neighbour is identical to GDAL on
//! every case, and every method is identical where the target is near 1:1
//! with the source window (`utm-to-4326`). Interpolated warps whose target
//! is coarser than the source window differ by design until kernel scaling
//! lands (stage S5, rwarp's documented limitation); they are reported, not
//! failed.

use std::path::PathBuf;

use serde_json::Value;
use wxy_warp::{warp, GridSpec, ResampleAlg, WarpOptions};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

fn alg(name: &str) -> ResampleAlg {
    match name {
        "near" => ResampleAlg::NearestNeighbour,
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" => ResampleAlg::Cubic,
        o => panic!("unknown resampling {o}"),
    }
}

/// Element-wise comparison in f64, reporting how many cells differ and by
/// how much at most.
fn compare(got: &[f64], want: &[f64]) -> (usize, f64) {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .filter(|(a, b)| a != b && !(a.is_nan() && b.is_nan()))
        .fold((0, 0.0f64), |(n, m), (a, b)| (n + 1, m.max((a - b).abs())))
}

fn decode(bytes: &[u8], dtype: &str) -> Vec<f64> {
    macro_rules! le {
        ($t:ty) => {
            bytes
                .chunks_exact(std::mem::size_of::<$t>())
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect()
        };
    }
    match dtype {
        "uint8" => le!(u8),
        "int16" => le!(i16),
        "uint16" => le!(u16),
        "float32" => le!(f32),
        "float64" => le!(f64),
        o => panic!("dtype {o}"),
    }
}

#[tokio::test]
async fn warps_match_gdal() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("warp/manifest.json")) else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let mut report = Vec::new();

    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ds = wxy_tiff::open_path(root.join(case["source"].as_str().unwrap()))
            .await
            .unwrap();
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let dst = GridSpec {
            crs: case["crs"].as_str().unwrap().into(),
            geotransform: gt.try_into().unwrap(),
            size: [
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
            ],
        };
        let nodata = ds.collection.fields[0].values.nodata;
        for r in case["results"].as_array().unwrap() {
            let m = r["resampling"].as_str().unwrap();
            let opts = WarpOptions {
                resampling: alg(m),
                wrap_longitude: false,
                ..Default::default()
            };
            let out = warp(
                &ds.levels[0],
                &ds.collection.domains[0],
                &dst,
                &opts,
                nodata,
            )
            .await
            .unwrap();
            let want = std::fs::read(root.join("warp").join(r["file"].as_str().unwrap())).unwrap();
            let want = decode(&want, case["dtype"].as_str().unwrap());
            let got = out.values.to_f64();
            let (n, max) = compare(&got, &want);
            let gated = m == "near" || name == "utm-to-4326";
            report.push((
                gated,
                n,
                format!("{name} {m}: {n} of {} cells differ (max {max})", want.len()),
            ));
        }
    }
    for (gated, _, line) in &report {
        eprintln!("{} {line}", if *gated { "gate " } else { "known" });
    }
    let failed: Vec<&String> = report
        .iter()
        .filter(|(gated, n, _)| *gated && *n > 0)
        .map(|(_, _, l)| l)
        .collect();
    assert!(failed.is_empty(), "warps differ from GDAL: {failed:#?}");
}

/// With longitude wrapping (the default), a target column whose cell
/// centres sit exactly on the antimeridian reads from the source instead of
/// coming back empty as it does from GDAL; nothing else changes.
#[tokio::test]
async fn antimeridian_column_wraps() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("warp/manifest.json")) else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let case = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "world-to-3031")
        .unwrap();
    let ds = wxy_tiff::open_path(root.join(case["source"].as_str().unwrap()))
        .await
        .unwrap();
    let dst = GridSpec::from_extent("EPSG:3031", [-4.0e6, 4.0e6, -4.0e6, 4.0e6], [145, 145]);
    let out = warp(
        &ds.levels[0],
        &ds.collection.domains[0],
        &dst,
        &WarpOptions::default(),
        None,
    )
    .await
    .unwrap();
    let want = decode(
        &std::fs::read(root.join("warp/world-to-3031-near.bin")).unwrap(),
        "int16",
    );
    let got = out.values.to_f64();
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    // Column 72 is x = 0; its cells below the pole are at longitude 180.
    assert!(differ.len() > 60, "{} cells differ", differ.len());
    for i in differ {
        assert_eq!(i % 145, 72, "cell {i} differs");
        assert_eq!(want[i], 0.0, "GDAL filled cell {i}");
    }
}
