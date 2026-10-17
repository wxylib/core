//! `wxy warp` end to end: the GeoZarr it writes holds what GDAL's warper
//! writes for the same source and target grid (wxylib/fixtures, warp/),
//! from a GeoTIFF source and from the same data as a Zarr source. Cases
//! are those the wxy-warp oracle test gates on.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use wxy_source::{read_window, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

fn decode(bytes: &[u8], dtype: &str) -> Vec<f64> {
    match dtype {
        "int16" => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        "uint16" => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        o => panic!("dtype {o}"),
    }
}

#[tokio::test]
async fn warp_writes_what_gdal_writes() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("warp/manifest.json")) else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let runs = [
        ("utm-to-4326", "cubic", "cog/utm-u16-lzw-pred2-3band.tif"),
        (
            "utm-to-4326-ovr",
            "bilinear",
            "cog/utm-u16-lzw-pred2-3band.tif",
        ),
        ("world-to-3031", "near", "cog/world-int16-deflate.tif"),
        ("world-to-3031", "near", "zarr/world-int16-spatial.zarr"),
    ];
    for (k, (name, method, source)) in runs.iter().enumerate() {
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == *name)
            .unwrap();
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let (w, h) = (
            case["width"].as_u64().unwrap(),
            case["height"].as_u64().unwrap(),
        );
        let extent = [
            gt[0],
            gt[3] + gt[5] * h as f64,
            gt[0] + gt[1] * w as f64,
            gt[3],
        ];
        let out = tmp.join(format!("cli-warp-{k}.zarr"));
        let _ = std::fs::remove_dir_all(&out);
        let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("warp")
            .arg(root.join(source))
            .arg(&out)
            .args([
                "--crs",
                case["crs"].as_str().unwrap(),
                "-r",
                method,
                "--extent",
            ])
            .args(extent.map(|v| v.to_string()))
            .args(["--size", &w.to_string(), &h.to_string()])
            .arg("--no-wrap")
            .status()
            .unwrap();
        assert!(status.success(), "{name} {method} from {source}");

        let ds = wxy_zarr::open_path(&out, "/data").await.unwrap();
        let shape = ds.collection.domains[0].shape();
        let got = read_window(&ds.source, &Window::new(vec![0; shape.len()], shape))
            .await
            .unwrap();
        let file = case["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["resampling"] == *method)
            .unwrap()["file"]
            .as_str()
            .unwrap();
        let want = decode(
            &std::fs::read(root.join("warp").join(file)).unwrap(),
            case["dtype"].as_str().unwrap(),
        );
        assert_eq!(got.values.to_f64(), want, "{name} {method} from {source}");
    }
}

#[test]
fn crs_the_writer_cannot_record_is_refused_before_reading() {
    let run = |crs: &str| {
        Command::new(env!("CARGO_BIN_EXE_wxy"))
            .args(["warp", "no-such-source.tif", "unused.zarr", "--crs", crs])
            .args(["--extent", "0", "0", "1", "1", "--size", "1", "1"])
            .output()
            .unwrap()
    };
    for crs in [
        "+proj=stere +lat_0=-90 +lat_ts=-71",
        "OGC:CRS84",
        "GEOGCRS[\"x\"]",
    ] {
        let out = run(crs);
        assert!(!out.status.success());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("CRS") || err.contains("crs"), "{crs}: {err}");
        assert!(
            !err.contains("no-such-source"),
            "{crs} read the source: {err}"
        );
    }
}

/// Stage S4b end to end: a time x depth x lat x lon Zarr store, chunked
/// two time steps deep, streams through `wxy warp` batch by batch, and the
/// store it writes keeps the leading dims and holds the in-memory warp.
#[tokio::test]
async fn a_4d_store_streams_through() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let (src, out) = (tmp.join("cli-4d-src.zarr"), tmp.join("cli-4d-out.zarr"));
    for p in [&src, &out] {
        let _ = std::fs::remove_dir_all(p);
    }
    let (nt, nz, ny, nx) = (3u64, 2u64, 90u64, 180u64);
    let mut v = Vec::new();
    for t in 0..nt {
        for z in 0..nz {
            for r in 0..ny {
                for c in 0..nx {
                    let (lon, lat) = (-179.0 + 2.0 * c as f64, 89.0 - 2.0 * r as f64);
                    let x =
                        100.0 * (lon.to_radians() * (1 + t) as f64).sin() * lat.to_radians().cos();
                    v.push((x + 10.0 * z as f64) as f32);
                }
            }
        }
    }
    let values = wxy_source::Values::F32(v);
    let grid = wxy_zarr::GridArray {
        dims: ["time", "depth", "lat", "lon"].map(String::from).to_vec(),
        shape: vec![nt, nz, ny, nx],
        chunk: vec![2, 1, 45, 90],
        geotransform: [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0],
        crs: Some("EPSG:4326".into()),
        nodata: None,
    };
    wxy_zarr::write_path(&src, "/sst", &values, &grid)
        .await
        .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
        .arg("warp")
        .arg(&src)
        .arg(&out)
        .args([
            "--source-array",
            "/sst",
            "--crs",
            "EPSG:3031",
            "-r",
            "bilinear",
        ])
        .args([
            "--extent", "-5e6", "-5e6", "5e6", "5e6", "--size", "60", "60",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let got = wxy_zarr::open_path(&out, "/data").await.unwrap();
    let d = &got.collection.domains[0];
    let names: Vec<&str> = d.dims.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["time", "depth", "y", "x"]);
    let shape = d.shape();
    assert_eq!(shape, vec![nt, nz, 60, 60]);
    let got = read_window(&got.source, &Window::new(vec![0; 4], shape))
        .await
        .unwrap();

    let ds = wxy_zarr::open_path(&src, "/sst").await.unwrap();
    let dst = wxy_warp::GridSpec::from_extent("EPSG:3031", [-5e6, 5e6, -5e6, 5e6], [60, 60]);
    let opts = wxy_warp::WarpOptions {
        resampling: wxy_warp::ResampleAlg::Bilinear,
        ..Default::default()
    };
    let want = wxy_warp::warp(&ds.source, &ds.collection.domains[0], &dst, &opts, None)
        .await
        .unwrap();
    assert_eq!(got.values, want.values);
}
