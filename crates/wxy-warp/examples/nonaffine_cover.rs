//! Coverage on grids that are not a regular lattice, by both methods of
//! `cover`, written as JSON for figures and an independent oracle
//! (wxylib/docs examples/nonaffine/). Usage:
//!
//! ```text
//! cargo run --release -p wxy-warp --example nonaffine_cover -- out.json
//! ```

#[path = "../tests/support/grids.rs"]
mod grids;

use std::time::Instant;

use serde_json::{json, Value};
use wxy_domain::CellConvention;
use wxy_warp::{
    cell_outlines, polygon_cells, polygon_centres_world, polygon_coverage_by, polygon_to_index,
    CellCover, CoverMethod,
};

use grids::CRS;

fn covers(c: &[Vec<CellCover>]) -> Value {
    json!(c
        .iter()
        .map(|p| p
            .iter()
            .map(|(c, f)| json!([c[0], c[1], f]))
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "nonaffine.json".into());
    let mut cases = Vec::new();
    for case in grids::all() {
        let s = case.domain.shape();
        let (h, w) = (s[0], s[1]);
        let polys = grids::polygons(case.name);
        let outlines: Vec<Value> = cell_outlines(&case.domain, CRS, 0..h, 0..w)
            .unwrap()
            .into_iter()
            .map(|q| json!(q.map(|q| q.to_vec())))
            .collect();
        let index = |d: Option<f64>| -> Vec<Value> {
            polys
                .iter()
                .map(|p| json!(polygon_to_index(&case.domain, CRS, p, d).unwrap().0))
                .collect()
        };
        let by = |m| covers(&polygon_coverage_by(&case.domain, CRS, &polys, m).unwrap());
        // densify sweep: how far index-space fractions sit from world ones
        let world = polygon_coverage_by(&case.domain, CRS, &polys, CoverMethod::World).unwrap();
        let sweep: Vec<Value> = [
            None,
            Some(2.0),
            Some(1.0),
            Some(0.5),
            Some(0.25),
            Some(0.1),
            Some(0.05),
            Some(0.02),
        ]
        .into_iter()
        .map(|d| {
            let c =
                polygon_coverage_by(&case.domain, CRS, &polys, CoverMethod::Index { densify: d })
                    .unwrap();
            let worst = (0..polys.len())
                .map(|k| {
                    let mut m: std::collections::BTreeMap<[u64; 2], f64> =
                        world[k].iter().copied().collect();
                    for (cell, f) in &c[k] {
                        *m.entry(*cell).or_insert(0.0) -= f;
                    }
                    m.values().fold(0f64, |a, v| a.max(v.abs()))
                })
                .fold(0f64, f64::max);
            json!({"densify": d, "max_cell_diff": worst})
        })
        .collect();
        cases.push(json!({
            "name": case.name,
            "about": case.about,
            "shape": [h, w],
            "outlines": outlines,
            "polygons": polys,
            "index_vertices": index(None),
            "index_dense": index(Some(0.05)),
            "cover_index": by(CoverMethod::Index { densify: None }),
            "cover_dense": by(CoverMethod::Index { densify: Some(0.05) }),
            "cover_world": covers(&world),
            "centres_gdal": polygon_cells(&case.domain, CRS, &polys, false).unwrap(),
            "centres_world": polygon_centres_world(&case.domain, CRS, &polys).unwrap(),
            "sweep": sweep,
        }));
        eprintln!("{}: {} x {}", case.name, h, w);
    }

    // cost: the same polygons on finer fans
    let mut timing = Vec::new();
    for scale in [1usize, 2, 4, 8, 16] {
        let (nx, ny) = (24 * scale, 30 * scale);
        let domain = grids::fan_with(CellConvention::Centre, nx, ny);
        let polys = grids::polygons("fan");
        let mut row = json!({"cells": nx * ny, "nx": nx, "ny": ny});
        for (key, m) in [
            ("index", CoverMethod::Index { densify: None }),
            (
                "dense",
                CoverMethod::Index {
                    densify: Some(0.25),
                },
            ),
            ("world", CoverMethod::World),
        ] {
            let t = Instant::now();
            let c = polygon_coverage_by(&domain, CRS, &polys, m).unwrap();
            let secs = t.elapsed().as_secs_f64();
            row[key] = json!(secs);
            row[format!("{key}_cells")] = json!(c.iter().map(Vec::len).sum::<usize>());
        }
        eprintln!("{row}");
        timing.push(row);
    }
    let doc = json!({"crs": CRS, "cases": cases, "timing": timing});
    std::fs::write(&out, serde_json::to_string(&doc).unwrap()).unwrap();
    eprintln!("wrote {out}");
}
