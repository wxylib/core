//! An mdim-refs byte-reference table (wxylib/fixtures, mdimrefs/: three
//! OISST-shaped netCDF4 files mosaicked by GDAL and scanned with its raw
//! block info) written to Icechunk and read back through zarrs and
//! wxy-zarr gives the same windows as GDAL reading the mosaic, and the
//! affine lat/lon coordinates come back as GeoZarr georeferencing.

use std::path::PathBuf;

use serde_json::Value;
use wxy_domain::{Frame, Positions};
use wxy_source::{read_window, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("mdimrefs")
}

#[tokio::test]
async fn virtual_mdimrefs_matches_gdal() {
    let root = fixtures();
    if !root.join("manifest.json").exists() {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    }
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json")).unwrap()).unwrap();

    let arrays = wxy_virtual::from_refs_table(&root, Some("fixture")).unwrap();
    let names: Vec<&str> = arrays.iter().map(|a| a.path.as_str()).collect();
    for want in ["/sst", "/ice", "/time", "/zlev", "/lat", "/lon"] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
    for (name, m) in manifest["arrays"].as_object().unwrap() {
        let a = arrays
            .iter()
            .find(|a| a.path == format!("/{name}"))
            .unwrap();
        assert_eq!(
            a.refs.len() as u64,
            m["present"].as_u64().unwrap(),
            "{name}"
        );
    }

    let repo = std::env::temp_dir().join(format!("wxy-virtual-mdimrefs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    wxy_virtual::write_icechunk(&repo, &arrays).await.unwrap();
    let storage = wxy_virtual::open_icechunk(&repo, &wxy_virtual::prefixes(&arrays))
        .await
        .unwrap();

    for name in ["sst", "ice"] {
        let ds = wxy_zarr::open(storage.clone(), name).await.unwrap();
        let d = &ds.collection.domains[0];
        assert_eq!(d.shape(), vec![3, 1, 36, 72], "{name}");
        let m = d.primary_for(2).unwrap();
        let Positions::Affine(a) = &m.positions else {
            panic!("{name}: expected an affine")
        };
        // Cell corners: lon 0 to 360 by 5, lat -90 to 90 by 5 (south up).
        assert_eq!(a.0.to_vec(), vec![0.0, 5.0, 0.0, -90.0, 0.0, 5.0], "{name}");
        assert_eq!(m.frame, Frame::crs("EPSG:4326"), "{name}");
        assert_eq!(
            ds.collection.fields[0].values.nodata,
            Some(-999.0),
            "{name}"
        );
        assert_eq!(
            ds.collection.fields[0].values.units.as_deref(),
            Some(if name == "sst" { "Celsius" } else { "%" })
        );
    }

    // time stays a CF coordinate, stored inline.
    let time = wxy_zarr::open(storage.clone(), "time").await.unwrap();
    let t = read_window(&time.source, &Window::new(vec![0], vec![3]))
        .await
        .unwrap();
    let want: Vec<f64> = manifest["coords"]["time"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(t.values.to_f64(), want);

    let mut checked = 0;
    for win in manifest["windows"].as_array().unwrap() {
        let name = win["array"].as_str().unwrap();
        let v = |k: &str| -> Vec<u64> {
            win[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let ds = wxy_zarr::open(storage.clone(), name).await.unwrap();
        let buf = read_window(&ds.source, &Window::new(v("start"), v("count")))
            .await
            .unwrap();
        let want = std::fs::read(root.join(win["file"].as_str().unwrap())).unwrap();
        assert_eq!(buf.values.to_le_bytes(), want, "window {win}");
        checked += 1;
    }
    let _ = std::fs::remove_dir_all(&repo);
    assert_eq!(checked, 5);
}
