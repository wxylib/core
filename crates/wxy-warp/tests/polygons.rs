//! Stage S4c: the cells each polygon covers match what GDAL's rasterizer
//! burns for it (wxylib/fixtures, polygons/), cell centres and
//! ALL_TOUCHED alike, from polygons in each file's CRS and in EPSG:4326.

use std::path::PathBuf;

use serde_json::Value;
use wxy_warp::polygon_cells;

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

#[tokio::test]
async fn polygon_cells_match_gdal() {
    let dir = fixtures().join("polygons");
    let Ok(file) = std::fs::File::open(dir.join("manifest.json")) else {
        eprintln!("skipping: no polygon fixtures at {}", dir.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let mut bad = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let read = |s: &str| std::fs::read(dir.join(format!("{name}-{s}"))).unwrap();
        let coords: Vec<f64> = read("coords.bin")
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let rings: Vec<Vec<usize>> = serde_json::from_slice(&read("rings.json")).unwrap();
        let cells: Value = serde_json::from_slice(&read("cells.json")).unwrap();
        let mut at = 0;
        let polygons: Vec<Vec<Vec<[f64; 2]>>> = rings
            .iter()
            .map(|p| {
                p.iter()
                    .map(|&n| {
                        let r = (0..n)
                            .map(|k| [coords[2 * (at + k)], coords[2 * (at + k) + 1]])
                            .collect();
                        at += n;
                        r
                    })
                    .collect()
            })
            .collect();

        let ds = wxy_tiff::open_path(fixtures().join(case["source"].as_str().unwrap()))
            .await
            .unwrap();
        let domain = &ds.collection.domains[0];
        let w = *domain.shape().last().unwrap();
        let crs = case["crs"].as_str().unwrap();
        for (mode, touched) in [("centre", false), ("all_touched", true)] {
            let got = polygon_cells(domain, crs, &polygons, touched).unwrap();
            for (k, runs) in cells[mode].as_array().unwrap().iter().enumerate() {
                let want: Vec<[u64; 2]> = runs
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|r| {
                        let (s, n) = (r[0].as_u64().unwrap(), r[1].as_u64().unwrap());
                        (s..s + n).map(move |i| [i / w, i % w])
                    })
                    .collect();
                if got[k] != want {
                    let extra = got[k].iter().filter(|c| !want.contains(c)).count();
                    let missing = want.iter().filter(|c| !got[k].contains(c)).count();
                    eprintln!(
                        "{name} {mode} polygon {k}: {extra} extra, {missing} missing of {}",
                        want.len()
                    );
                    bad += 1;
                }
            }
        }
        eprintln!("{name}: {} polygons", polygons.len());
    }
    assert_eq!(bad, 0, "polygons whose cells differ from GDAL's");
}
