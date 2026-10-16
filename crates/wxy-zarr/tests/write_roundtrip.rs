//! A written grid opens again with the same values, georeferencing and CRS.

use wxy_domain::{Frame, Positions};
use wxy_source::{read_window, Values, Window};
use wxy_zarr::GridArray;

#[tokio::test]
async fn write_then_open() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("write-roundtrip.zarr");
    let _ = std::fs::remove_dir_all(&root);
    let shape = vec![2, 3, 700];
    let values = Values::I16((0..2 * 3 * 700).map(|i| (i % 300) as i16 - 100).collect());
    let gt = [100.0, 10.0, 0.0, 500.0, 0.0, -10.0];
    let mut grid = GridArray::new(shape.clone(), gt);
    grid.crs = Some("EPSG:3577".into());
    grid.nodata = Some(-100.0);
    assert_eq!(grid.chunk, vec![1, 3, 512]);
    wxy_zarr::write_path(&root, "/out/data", &values, &grid)
        .await
        .unwrap();

    let ds = wxy_zarr::open_path(&root, "/out/data").await.unwrap();
    let domain = &ds.collection.domains[0];
    let m = domain.primary_for(2).unwrap();
    let Positions::Affine(a) = &m.positions else {
        panic!("not affine")
    };
    assert_eq!(a.0, gt);
    assert_eq!(m.frame, Frame::crs("EPSG:3577"));
    assert_eq!(ds.collection.fields[0].values.nodata, Some(-100.0));
    let got = read_window(&ds.source, &Window::new(vec![0, 0, 0], shape))
        .await
        .unwrap();
    assert_eq!(got.values, values);
}

#[tokio::test]
async fn nan_nodata_survives() {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("write-nan.zarr");
    let _ = std::fs::remove_dir_all(&root);
    let values = Values::F32(vec![f32::NAN, 1.0, 2.0, 3.0]);
    let mut grid = GridArray::new(vec![2, 2], [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]);
    grid.nodata = Some(f64::NAN);
    wxy_zarr::write_path(&root, "/data", &values, &grid)
        .await
        .unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("data/zarr.json")).unwrap()).unwrap();
    assert_eq!(meta["fill_value"], "NaN");
    let ds = wxy_zarr::open_path(&root, "/data").await.unwrap();
    assert!(ds.collection.fields[0].values.nodata.unwrap().is_nan());
}
