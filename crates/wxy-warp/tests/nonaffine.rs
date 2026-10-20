//! Polygon coverage on grids that are not a regular lattice: rotated
//! affine, rectilinear (walls or centres given) and curvilinear (a polar
//! sector, centres or corners given), by the two methods of `cover`:
//! burning in index space, and clipping interpolated cells in the
//! polygon's CRS. The world-space clip is the reference: each polygon lies
//! inside its grid, so its cells' covered areas sum to its own area.

mod support;

use std::collections::BTreeMap;

use support::grids::{self, CRS};
use wxy_warp::{
    cell_outlines, polygon_cells, polygon_centres_world, polygon_coverage_by, CellCover,
    CoverMethod, Rings,
};

fn shoelace(r: &[[f64; 2]]) -> f64 {
    r.windows(2)
        .map(|s| s[0][0] * s[1][1] - s[1][0] * s[0][1])
        .sum::<f64>()
        .abs()
        / 2.0
}

fn polygon_area(p: &Rings) -> f64 {
    shoelace(&p[0]) - p[1..].iter().map(|r| shoelace(r)).sum::<f64>()
}

/// World area of every cell, by `(row, col)`.
fn cell_areas(domain: &wxy_domain::Domain) -> BTreeMap<[u64; 2], f64> {
    let s = domain.shape();
    let (h, w) = (s[0], s[1]);
    let out = cell_outlines(domain, CRS, 0..h, 0..w).unwrap();
    let mut m = BTreeMap::new();
    for r in 0..h {
        for c in 0..w {
            let q = out[(r * w + c) as usize].unwrap();
            let mut ring = q.to_vec();
            ring.push(q[0]);
            m.insert([r, c], shoelace(&ring));
        }
    }
    m
}

fn weighted(cover: &[CellCover], areas: &BTreeMap<[u64; 2], f64>) -> f64 {
    cover.iter().map(|(c, f)| f * areas[c]).sum()
}

/// Largest per-cell fraction difference between two covers.
fn max_diff(a: &[CellCover], b: &[CellCover]) -> f64 {
    let mut m: BTreeMap<[u64; 2], f64> = a.iter().copied().collect();
    for (c, f) in b {
        *m.entry(*c).or_insert(0.0) -= f;
    }
    m.values().fold(0.0, |acc, v| acc.max(v.abs()))
}

const DENSE: CoverMethod = CoverMethod::Index {
    densify: Some(0.05),
};
const VERTS: CoverMethod = CoverMethod::Index { densify: None };

struct Row {
    world: f64,
    verts: f64,
    dense: f64,
    d_verts: f64,
    d_dense: f64,
}

fn run(case: &grids::Case) -> Vec<Row> {
    let polys = grids::polygons(case.name);
    let areas = cell_areas(&case.domain);
    let world = polygon_coverage_by(&case.domain, CRS, &polys, CoverMethod::World).unwrap();
    let verts = polygon_coverage_by(&case.domain, CRS, &polys, VERTS).unwrap();
    let dense = polygon_coverage_by(&case.domain, CRS, &polys, DENSE).unwrap();
    polys
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let truth = polygon_area(p);
            let row = Row {
                world: weighted(&world[k], &areas) / truth - 1.0,
                verts: weighted(&verts[k], &areas) / truth - 1.0,
                dense: weighted(&dense[k], &areas) / truth - 1.0,
                d_verts: max_diff(&world[k], &verts[k]),
                d_dense: max_diff(&world[k], &dense[k]),
            };
            eprintln!(
                "{:13} polygon {k}: area error world {:+.2e} index {:+.2e} dense {:+.2e}; \
                 max cell diff from world: index {:.4} dense {:.4}",
                case.name, row.world, row.verts, row.dense, row.d_verts, row.d_dense
            );
            row
        })
        .collect()
}

#[test]
fn world_clip_covers_each_polygon_exactly() {
    for case in grids::all() {
        for row in run(&case) {
            assert!(row.world.abs() < 1e-9, "{}: {:e}", case.name, row.world);
        }
    }
}

#[test]
fn rotated_affine_is_exact_in_index_space() {
    // an affine map keeps straight edges straight and area ratios, so
    // moving vertices alone is already exact
    for row in run(&grids::rotated()) {
        assert!(row.verts.abs() < 1e-6 && row.d_verts < 1e-5);
        assert!(row.dense.abs() < 1e-6 && row.d_dense < 1e-5);
    }
}

#[test]
fn rectilinear_walls_converge_with_densify() {
    // piecewise affine cell by cell: densified edges converge on the
    // world answer; vertex-only edges cut corners across cell walls
    for row in run(&grids::rect_edges()) {
        assert!(row.dense.abs() < 1e-4, "{:e}", row.dense);
        assert!(row.d_dense < 0.02, "{}", row.d_dense);
        assert!(row.d_verts > 5.0 * row.d_dense);
    }
}

#[test]
fn centre_coordinates_bend_cells_in_index_space() {
    // given centres, a cell spans two interpolation patches in each
    // direction and its index-space area is not proportional to its world
    // area: densifying fixes the edges but not the measure
    for case in [
        grids::rect_centres(),
        grids::fan_centres(),
        grids::fan_edges(),
    ] {
        for row in run(&case) {
            assert!(row.d_dense < row.d_verts + 1e-9, "{}", case.name);
            assert!(row.d_dense > 1e-3, "{}: {}", case.name, row.d_dense);
        }
    }
}

#[test]
fn centre_rule_matches_coordinate_matching_where_affine() {
    // GDAL's rule (centre of the index-space cell inside the moved
    // polygon) against testing the grid's own centre coordinates
    for case in grids::all() {
        let polys = grids::polygons(case.name);
        let gdal = polygon_cells(&case.domain, CRS, &polys, false).unwrap();
        let world = polygon_centres_world(&case.domain, CRS, &polys).unwrap();
        for k in 0..polys.len() {
            let extra = gdal[k].iter().filter(|c| !world[k].contains(c)).count();
            let missing = world[k].iter().filter(|c| !gdal[k].contains(c)).count();
            eprintln!(
                "{:13} polygon {k}: centre cells {} by coordinates, {extra} extra and {missing} missing by moved vertices",
                case.name,
                world[k].len()
            );
            if case.name == "rotated" {
                assert_eq!(extra + missing, 0);
            }
        }
    }
}
