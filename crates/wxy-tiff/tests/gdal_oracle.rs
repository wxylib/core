//! Windows read through wxy-tiff against what GDAL reads for the same
//! windows. Fixtures and oracle values come from wxylib/fixtures
//! (`cog/`, made by `scripts/make_cog_fixtures.py`).
//!
//! Set `WXY_FIXTURES` to a checkout of wxylib/fixtures. If it is unset, the
//! tests look for `../fixtures` beside this workspace, and skip (with a
//! message) when neither exists.

use std::path::PathBuf;

use wxy_domain::{Positions, RelationKind};
use wxy_source::{read_window, Window};

fn fixtures() -> Option<PathBuf> {
    let p = std::env::var_os("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"));
    let p = p.join("cog");
    if p.join("manifest.json").exists() {
        Some(p)
    } else {
        eprintln!(
            "skipping: no wxylib/fixtures checkout at {} (set WXY_FIXTURES)",
            p.display()
        );
        None
    }
}

struct Case {
    file: String,
    width: u64,
    height: u64,
    count: u64,
    geotransform: Vec<f64>,
    epsg: u64,
    nodata: Option<f64>,
    overviews: Vec<f64>,
    levels: Vec<Vec<f64>>,
    windows: Vec<(u64, u64, u64, u64, String)>,
}

fn cases(dir: &PathBuf) -> Vec<Case> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let u = |x: &serde_json::Value| x.as_u64().unwrap();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            file: c["file"].as_str().unwrap().to_string(),
            width: u(&c["width"]),
            height: u(&c["height"]),
            count: u(&c["count"]),
            geotransform: c["geotransform"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect(),
            epsg: u(&c["epsg"]),
            nodata: c["nodata"].as_f64(),
            overviews: c["overviews"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect(),
            levels: c["levels"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|l| {
                            l["geotransform"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_f64().unwrap())
                                .collect()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            windows: c["windows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    (
                        u(&w["col"]),
                        u(&w["row"]),
                        u(&w["width"]),
                        u(&w["height"]),
                        w["file"].as_str().unwrap().to_string(),
                    )
                })
                .collect(),
        })
        .collect()
}

#[tokio::test]
async fn windows_match_gdal() {
    let Some(dir) = fixtures() else { return };
    let cases = cases(&dir);
    assert_eq!(cases.len(), 4);
    for c in cases {
        let ds = wxy_tiff::open_path(dir.join(&c.file)).await.unwrap();
        let level0 = &ds.collection.domains[0];
        assert_eq!(
            level0.shape(),
            vec![c.count, c.height, c.width],
            "{}",
            c.file
        );

        // Georeferencing as GDAL reports it.
        let m = level0.primary_for(2).unwrap();
        let Positions::Affine(a) = &m.positions else {
            panic!("{}: not affine", c.file)
        };
        for k in 0..6 {
            assert!(
                (a.0[k] - c.geotransform[k]).abs() < 1e-9,
                "{}: gt {:?}",
                c.file,
                a.0
            );
        }
        assert_eq!(m.frame, wxy_domain::Frame::crs(format!("EPSG:{}", c.epsg)));
        assert_eq!(
            ds.collection.fields[0].values.nodata, c.nodata,
            "{}",
            c.file
        );

        // Overview levels and their factors.
        assert_eq!(ds.levels.len(), 1 + c.overviews.len(), "{}", c.file);
        for (r, want) in ds.collection.related.iter().zip(&c.overviews) {
            let RelationKind::Overview { factor } = &r.kind else {
                panic!()
            };
            assert!(
                (factor[0] - want).abs() < 0.02 * want,
                "{}: factor {factor:?}",
                c.file
            );
        }

        // Overview georeferencing: same extent, scaled pixel size.
        for (k, want) in c.levels.iter().enumerate() {
            let d = &ds.collection.domains[k + 1];
            let Positions::Affine(a) = &d.primary_for(2).unwrap().positions else {
                panic!()
            };
            for j in 0..6 {
                assert!(
                    (a.0[j] - want[j]).abs() < 1e-6 * want[j].abs().max(1.0),
                    "{} level {}: {:?}",
                    c.file,
                    k + 1,
                    a.0
                );
            }
        }

        // Pixel values, window by window.
        for (col, row, w, h, f) in &c.windows {
            let win = Window::new(vec![0, *row, *col], vec![c.count, *h, *w]);
            let buf = read_window(&ds.levels[0], &win).await.unwrap();
            let want = std::fs::read(dir.join(f)).unwrap();
            assert_eq!(buf.values.to_le_bytes(), want, "{} window {f}", c.file);
        }
    }
}
