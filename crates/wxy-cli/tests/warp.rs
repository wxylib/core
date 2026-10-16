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
