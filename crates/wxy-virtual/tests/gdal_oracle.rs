//! A COG scanned into Icechunk virtual references and read back through
//! zarrs and wxy-zarr gives the same windows and georeferencing as GDAL
//! reading the COG (wxylib/fixtures, cog/).

use std::path::PathBuf;

use serde_json::Value;
use wxy_domain::{Frame, Positions};
use wxy_source::{read_window, Error, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("cog")
}

#[tokio::test]
async fn virtual_cog_matches_gdal() {
    let root = fixtures();
    if !root.join("manifest.json").exists() {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    }
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json")).unwrap()).unwrap();
    let scratch = std::env::temp_dir().join(format!("wxy-virtual-{}", std::process::id()));
    let mut checked = 0;

    for (i, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
        let file = case["file"].as_str().unwrap();
        let va = match wxy_virtual::scan_tiff(root.join(file)).await {
            Ok(v) => v,
            Err(Error::Unsupported(why)) => {
                eprintln!("{file}: not referenceable ({why})");
                continue;
            }
            Err(e) => panic!("{file}: {e}"),
        };
        let repo = scratch.join(format!("repo-{i}"));
        let _ = std::fs::remove_dir_all(&repo);
        wxy_virtual::write_icechunk(&repo, std::slice::from_ref(&va))
            .await
            .unwrap();

        let storage = wxy_virtual::open_icechunk(&repo, &wxy_virtual::prefixes(&[va]))
            .await
            .unwrap();
        let ds = wxy_zarr::open(storage, "/data").await.unwrap();

        let d = &ds.collection.domains[0];
        let (w, h, bands) = (
            case["width"].as_u64().unwrap(),
            case["height"].as_u64().unwrap(),
            case["count"].as_u64().unwrap(),
        );
        assert_eq!(d.shape(), vec![bands, h, w], "{file}");
        let m = d.primary_for(2).unwrap();
        let Positions::Affine(a) = &m.positions else {
            panic!("{file}: expected an affine")
        };
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(a.0.to_vec(), gt, "{file}");
        let epsg = format!("EPSG:{}", case["epsg"].as_u64().unwrap());
        assert_eq!(m.frame, Frame::crs(epsg), "{file}");
        assert_eq!(
            ds.collection.fields[0].values.nodata,
            case["nodata"].as_f64(),
            "{file}"
        );

        for win in case["windows"].as_array().unwrap() {
            let g = |k: &str| win[k].as_u64().unwrap();
            let window = Window::new(
                vec![0, g("row"), g("col")],
                vec![bands, g("height"), g("width")],
            );
            let buf = read_window(&ds.source, &window).await.unwrap();
            let want = std::fs::read(root.join(win["file"].as_str().unwrap())).unwrap();
            assert_eq!(buf.values.to_le_bytes(), want, "{file} window {win}");
        }
        checked += 1;
    }
    let _ = std::fs::remove_dir_all(&scratch);
    assert!(checked >= 4, "only {checked} fixtures were referenceable");
}
