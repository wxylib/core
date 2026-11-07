//! Footprints against the planner, over the VRT corpus (wxylib/fixtures,
//! vrt/): for whole-pixel windows across each classic case, the sources whose
//! footprint meets the window's box are the sources the planner reads for
//! that window. The planner follows the reference library's window
//! arithmetic, so this ties the catalogue view to what a read touches.

use std::collections::BTreeSet;
use std::path::PathBuf;

use vrt_model::footprint::rect_bounds;
use vrt_model::{plan_band, sources_in_bounds, Dataset, Dsn, Rect, Request, Resolver};

fn root() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("vrt")
}

struct NoIo;
impl Resolver for NoIo {
    fn size(&self, _: &Dsn) -> Option<(usize, usize)> {
        None
    }
}

#[test]
fn footprints_agree_with_the_planner() {
    let dir = root();
    let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        eprintln!("fixtures not found at {}; skipping", dir.display());
        return;
    };
    let m: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut checked = 0;
    for case in m["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !name.ends_with(".vrt") {
            continue;
        }
        let ds = Dataset::from_xml(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap();
        let gt = ds.geotransform.unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let (w, h) = (ds.x_size, ds.y_size);
        for band in 1..=ds.bands.len() {
            for &(x0, y0, xs, ys) in &[
                (0, 0, w, h),
                (0, 0, 1, h),
                (w / 2, 0, 1, h),
                (0, h / 2, w, 1),
                (w / 3, h / 3, w / 3, h / 3),
                (w - 1, h - 1, 1, 1),
            ] {
                let r = Request::window(x0, y0, xs, ys);
                let Ok(rows) = plan_band(&ds, band, &r, "", &NoIo) else {
                    continue;
                };
                let planned: BTreeSet<usize> = rows.iter().map(|r| r.source).collect();
                let q = rect_bounds(
                    &gt,
                    &Rect {
                        x_off: x0 as f64,
                        y_off: y0 as f64,
                        x_size: xs as f64,
                        y_size: ys as f64,
                    },
                );
                let found: BTreeSet<usize> = sources_in_bounds(&ds, band, &q, "", &NoIo)
                    .unwrap()
                    .iter()
                    .map(|f| f.source)
                    .collect();
                assert_eq!(
                    found, planned,
                    "{name} band {band} window {x0},{y0},{xs},{ys}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
    eprintln!("{checked} windows checked");
}
