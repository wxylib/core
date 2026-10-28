//! Stage S4c: the cells each polygon covers match what GDAL's rasterizer
//! burns for it (wxylib/fixtures, polygons/), cell centres and
//! ALL_TOUCHED alike, from polygons in each file's CRS and in EPSG:4326.

use std::path::PathBuf;

use serde_json::Value;
use wxy_warp::{polygon_cells, polygon_coverage};

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
        // exact coverage: every cell GDAL burns by centre is covered, and
        // a polygon inside the grid covers its own area, in cells
        let cover = polygon_coverage(domain, crs, &polygons).unwrap();
        let centre = polygon_cells(domain, crs, &polygons, false).unwrap();
        let gt = match &domain.primary_for(1).unwrap().positions {
            wxy_domain::Positions::Affine(a) => a.0,
            _ => unreachable!(),
        };
        let h = domain.shape()[domain.shape().len() - 2] as f64;
        for (k, p) in polygons.iter().enumerate() {
            let covered: std::collections::HashSet<[u64; 2]> =
                cover[k].iter().map(|c| c.0).collect();
            assert!(
                centre[k].iter().all(|c| covered.contains(c)),
                "{name} polygon {k}: a centre cell not covered ({} of {}, cover {})",
                centre[k].iter().filter(|c| !covered.contains(*c)).count(),
                centre[k].len(),
                cover[k].len()
            );
            assert!(cover[k].iter().all(|c| c.1 > 0.0 && c.1 <= 1.0));
            if crs == "EPSG:4326" && !name.starts_with("world") {
                continue;
            }
            let inside = p.iter().flatten().all(|v| {
                let col = (v[0] - gt[0]) / gt[1];
                let row = (v[1] - gt[3]) / gt[5];
                col >= 0.0 && col <= w as f64 && row >= 0.0 && row <= h
            });
            if !inside {
                continue;
            }
            let shoelace = |r: &Vec<[f64; 2]>| {
                r.windows(2)
                    .map(|s| s[0][0] * s[1][1] - s[1][0] * s[0][1])
                    .sum::<f64>()
                    .abs()
                    / 2.0
            };
            let area = shoelace(&p[0]) - p[1..].iter().map(shoelace).sum::<f64>();
            let cells = area / (gt[1] * gt[5]).abs();
            let got: f64 = cover[k].iter().map(|c| c.1).sum();
            assert!(
                (got - cells).abs() < 1e-4 * cells.max(1.0),
                "{name} polygon {k}: covers {got} cells, area is {cells}"
            );
        }
        eprintln!("{name}: {} polygons", polygons.len());
    }
    assert_eq!(bad, 0, "polygons whose cells differ from GDAL's");
}

/// A polygon with a hole, cut through cells, on the half degree world: the
/// covered area is the exterior's less the hole's, the hole's interior
/// cells are left out, and its boundary cells are half covered.
#[tokio::test]
async fn coverage_subtracts_holes() {
    let Ok(ds) = wxy_tiff::open_path(fixtures().join("cog/world-int16-deflate.tif")).await else {
        eprintln!("skipping: no fixtures at {}", fixtures().display());
        return;
    };
    let square = |a: f64, b: f64| vec![[a, a], [b, a], [b, b], [a, b], [a, a]];
    let polygon = vec![square(0.0, 10.0), square(2.25, 7.75)];
    let cover = polygon_coverage(&ds.collection.domains[0], "EPSG:4326", &[polygon]).unwrap();
    let total: f64 = cover[0].iter().map(|c| c.1).sum();
    // cells of 0.5 degrees: 4 to a square degree
    let want = (100.0 - 5.5 * 5.5) * 4.0;
    assert!((total - want).abs() < 1e-6, "covered {total}, want {want}");
    // lat 5, lon 5 is row 170, col 370, inside the hole
    assert!(
        cover[0].iter().all(|c| c.0 != [170, 370]),
        "a hole cell is covered"
    );
    // the hole's west edge (lon 2.25) halves col 364
    let edge = cover[0].iter().find(|c| c.0 == [170, 364]).unwrap();
    assert!((edge.1 - 0.5).abs() < 1e-6, "edge cell {edge:?}");
}
