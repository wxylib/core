//! Zarr fixtures (wxylib/fixtures, zarr/) hold the same pixels as the COG
//! fixtures, so the windows GDAL read from the COGs are the oracle here
//! too. Georeferencing is checked against the COG's GDAL geotransform.

use std::path::PathBuf;

use serde_json::Value;
use wxy_domain::normalise::{affine_from_axes, normalise_domain};
use wxy_domain::{Affine, Frame, Positions, Rung};
use wxy_source::{read_window, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("zarr")
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[tokio::test]
async fn zarr_matches_gdal() {
    let root = fixtures();
    if !root.join("manifest.json").exists() {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    }
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json")).unwrap()).unwrap();

    for case in manifest["cases"].as_array().unwrap() {
        let store = case["store"].as_str().unwrap();
        let ds = wxy_zarr::open_path(root.join(store), case["array"].as_str().unwrap())
            .await
            .unwrap();
        let d = &ds.collection.domains[0];
        let shape: Vec<u64> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(d.shape(), shape, "{store}");

        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let epsg = Frame::crs(format!("EPSG:{}", case["epsg"].as_u64().unwrap()));
        let n = shape.len();
        let (y, x) = (n - 2, n - 1);

        let (affine, frame): (Affine, Frame) = match case["georef"].as_str().unwrap() {
            "spatial" => {
                let m = d.primary_for(x).unwrap();
                let Positions::Affine(a) = &m.positions else {
                    panic!("{store}: expected an affine")
                };
                assert_eq!(m.dims, vec![y, x]);
                (*a, m.frame.clone())
            }
            "cf" => {
                let mut d = d.clone();
                assert_eq!(d.rung_for(x), Rung::Rectilinear);
                let reports = normalise_domain(&mut d);
                assert!(reports.iter().all(|r| r.accepted), "{reports:?}");
                affine_from_axes(&d, y, x).unwrap()
            }
            g => panic!("unknown georef {g}"),
        };
        let tol = 1e-6 * gt[1].abs().max(1.0);
        assert!(
            close(&affine.0, &gt, tol),
            "{store}: {:?} vs {gt:?}",
            affine.0
        );
        assert_eq!(frame, epsg, "{store}");

        let nodata = case["nodata"].as_f64();
        assert_eq!(ds.collection.fields[0].values.nodata, nodata, "{store}");

        for w in case["windows"].as_array().unwrap() {
            let g = |k: &str| w[k].as_u64().unwrap();
            let (col, row, width, height) = (g("col"), g("row"), g("width"), g("height"));
            let (mut off, mut size) = (vec![row, col], vec![height, width]);
            if n == 3 {
                off.insert(0, 0);
                size.insert(0, shape[0]);
            }
            let buf = read_window(&ds.source, &Window::new(off, size))
                .await
                .unwrap();
            let want = std::fs::read(root.join(w["file"].as_str().unwrap())).unwrap();
            assert_eq!(buf.values.to_le_bytes(), want, "{store} window {w}");
        }
    }
}
