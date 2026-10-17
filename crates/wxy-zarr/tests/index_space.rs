//! Index space (fixtures index/, design m07, I3): a CF rotated-pole store
//! as CORDEX writes it, 1D `rlon`, `rlat` and float32 2D `lon`, `lat`. The
//! 2D arrays are the primary mapping (R3); through the rotated pole they
//! are regular, and `regular_grid_through` recovers the affine the fixture
//! was built from at float32 tolerance.

use std::path::PathBuf;

use serde_json::Value;
use wxy_domain::normalise::{regular_grid_through, Tolerance};
use wxy_domain::{Positions, Rung};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("index")
}

/// Longitude and latitude to rotated longitude and latitude, degrees, on
/// the sphere: the inverse of the rotation that built the fixture.
fn to_rotated(pole_lat: f64, pole_lon: f64) -> impl Fn(f64, f64) -> Option<(f64, f64)> {
    let (s, c) = pole_lat.to_radians().sin_cos();
    move |lon, lat| {
        let (l, p) = ((lon - pole_lon - 180.0).to_radians(), lat.to_radians());
        let (x, y, z) = (p.cos() * l.cos(), p.cos() * l.sin(), p.sin());
        let (rx, ry, rz) = (s * x + c * z, y, -c * x + s * z);
        Some((ry.atan2(rx).to_degrees(), rz.asin().to_degrees()))
    }
}

#[tokio::test]
async fn rotated_pole_is_regular_through_its_pole() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("manifest.json")) else {
        eprintln!("skipping: no index fixtures at {}", root.display());
        return;
    };
    let m: Value = serde_json::from_reader(file).unwrap();
    let case = m["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["case"] == "I3")
        .unwrap();
    let ds = wxy_zarr::open_path(root.join(case["store"].as_str().unwrap()), "/tas")
        .await
        .unwrap();
    let d = &ds.collection.domains[0];
    assert_eq!(d.rung_for(1), Rung::Curvilinear);
    let Positions::Curvilinear(xy) = &d.primary_for(1).unwrap().positions else {
        panic!("not curvilinear")
    };

    let gm = &case["grid_mapping"];
    let f = to_rotated(
        gm["grid_north_pole_latitude"].as_f64().unwrap(),
        gm["grid_north_pole_longitude"].as_f64().unwrap(),
    );
    let (affine, report) = regular_grid_through(
        &xy[0],
        &xy[1],
        f,
        "rotated_latitude_longitude",
        Tolerance::RelativeToStep(1e-4),
    );
    assert!(report.accepted, "{report}");
    let got = affine.unwrap().0;
    let want: Vec<f64> = case["affine"]["geotransform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for k in 0..6 {
        assert!((got[k] - want[k]).abs() < 1e-4, "{got:?} vs {want:?}");
    }
}
