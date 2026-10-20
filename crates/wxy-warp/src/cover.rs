//! Polygon coverage on grids that are not a regular lattice in the
//! polygon's CRS: rotated or sheared affine grids, rectilinear grids
//! (irregular 1D axes) and curvilinear grids (2D coordinate arrays). Two
//! ways in, side by side so they can be compared:
//!
//! - [`CoverMethod::Index`], the vector moves to the grid: polygon
//!   vertices go to fractional `(pixel, line)` and are burned on a lattice
//!   of unit cells (controlledburn), as [`polygon_coverage`] does. With
//!   `densify`, edges are cut so they follow the curve a straight edge
//!   becomes in index space. Fractions are of index-space area: exact when
//!   the grid is affine within each cell, not otherwise.
//! - [`CoverMethod::World`], the grid moves to the vector (coordinate
//!   matching and interpolation): each cell's boundary is interpolated
//!   from the grid's coordinates into the polygon's CRS and the polygon is
//!   clipped by it there. Fractions are of world area, exact for the cell
//!   the coordinates define.
//!
//! A cell is the image of its unit square under the grid's own
//! interpolation (the geolocation transformer's bilinear patches), so both
//! methods partition the plane into the same cells and differ only in how
//! they measure them. A cell side is straight within each bilinear patch
//! and bends where it crosses into the next, at most once (at a centre
//! row or column when the coordinates are cell centres), so eight points,
//! the corners and side midpoints, trace each cell exactly.
//!
//! [`polygon_coverage`]: crate::polygon_coverage

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use wxy_domain::Domain;
use wxy_source::Result;

use crate::zonal::{cell_covers, Burner};
use crate::{CellCover, Rings};

/// How cell coverage is measured on a grid that may not be affine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverMethod {
    /// Polygon to index space, burned on unit cells. `densify` is the
    /// longest edge, in cells once moved, before an edge is cut; `None`
    /// moves vertices only (as GDAL's rasterizer does).
    Index { densify: Option<f64> },
    /// Cells to the polygon's CRS, each clipped by the polygon.
    World,
}

/// The cells of a source each polygon covers and the fraction of each it
/// covers (in `(0, 1]`), by `method`. Cells are `(row, col)`, sorted.
pub fn polygon_coverage_by(
    domain: &Domain,
    crs: &str,
    polygons: &[Rings],
    method: CoverMethod,
) -> Result<Vec<Vec<CellCover>>> {
    let b = Burner::new(domain, crs)?;
    let features: Vec<Vec<Rings>> = polygons.iter().map(|p| vec![p.clone()]).collect();
    match method {
        CoverMethod::Index { densify } => Ok(b
            .coverage(&features, densify)?
            .into_iter()
            .map(cell_covers)
            .collect()),
        CoverMethod::World => Ok(polygons.iter().map(|p| world_cover(&b, p)).collect()),
    }
}

/// The cells of a source whose centre, in the polygon's CRS, is inside
/// each polygon: the grid's own coordinates tested against the polygon,
/// with no move of the polygon at all. Cells are `(row, col)`, sorted.
pub fn polygon_centres_world(
    domain: &Domain,
    crs: &str,
    polygons: &[Rings],
) -> Result<Vec<Vec<[u64; 2]>>> {
    let b = Burner::new(domain, crs)?;
    Ok(polygons
        .iter()
        .map(|rings| {
            let Some(win) = window(&b, rings) else {
                return Vec::new();
            };
            let mut x = Vec::new();
            let mut y = Vec::new();
            let mut cells = Vec::new();
            for r in win.rows.clone() {
                for c in win.cols.clone() {
                    x.push(c as f64 + 0.5);
                    y.push(r as f64 + 0.5);
                    cells.push([r, c]);
                }
            }
            let ok = rwarp::Transformer::transform(&b.t, false, &mut x, &mut y);
            cells
                .into_iter()
                .enumerate()
                .filter(|&(i, _)| ok[i] && inside(rings, [x[i], y[i]]))
                .map(|(_, c)| c)
                .collect()
        })
        .collect())
}

/// A polygon's rings in the source's index space, `(pixel, line)`, as
/// [`CoverMethod::Index`] burns them, with the count of vertices that did
/// not transform.
pub fn polygon_to_index(
    domain: &Domain,
    crs: &str,
    rings: &Rings,
    densify: Option<f64>,
) -> Result<(Rings, usize)> {
    let b = Burner::new(domain, crs)?;
    let mut failed = 0;
    let out = rings
        .iter()
        .map(|r| {
            let (ring, f) = b.index_ring(r, densify);
            failed += f;
            ring
        })
        .collect();
    Ok((out, failed))
}

/// Each cell's boundary in the polygon's CRS, as eight points (corners
/// and side midpoints, anticlockwise in index space from the top left),
/// for the cells in `rows` by `cols`, row major. A cell that does not
/// transform gives `None`.
pub fn cell_outlines(
    domain: &Domain,
    crs: &str,
    rows: std::ops::Range<u64>,
    cols: std::ops::Range<u64>,
) -> Result<Vec<Option<[[f64; 2]; 8]>>> {
    let b = Burner::new(domain, crs)?;
    let mut out = Vec::new();
    for r in rows {
        for c in cols.clone() {
            out.push(outline(&b, r, c));
        }
    }
    Ok(out)
}

/// Rows and columns a polygon can touch.
struct Window {
    rows: std::ops::Range<u64>,
    cols: std::ops::Range<u64>,
}

/// The cells under a polygon's index-space envelope and one more all
/// round, or the whole grid when a vertex does not move (it is off a
/// curvilinear grid's arrays, which can still be under the polygon).
fn window(b: &Burner, rings: &Rings) -> Option<Window> {
    let ext = rings.first()?;
    let (ring, failed) = b.index_ring(ext, Some(0.5));
    if failed > 0 {
        return Some(Window {
            rows: 0..b.h,
            cols: 0..b.w,
        });
    }
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in &ring {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    let lo = |v: f64| (v.floor() - 1.0).max(0.0) as u64;
    let hi = |v: f64, n: u64| ((v.floor() + 2.0).max(0.0) as u64).min(n);
    let w = Window {
        rows: lo(y0)..hi(y1, b.h),
        cols: lo(x0)..hi(x1, b.w),
    };
    (!w.rows.is_empty() && !w.cols.is_empty()).then_some(w)
}

/// Cell `(r, c)`'s boundary in the polygon's CRS.
fn outline(b: &Burner, r: u64, c: u64) -> Option<[[f64; 2]; 8]> {
    let (r, c) = (r as f64, c as f64);
    // anticlockwise in index space with y down is clockwise on screen;
    // orientation does not matter, areas are taken signed then made positive
    let idx = [
        (c, r),
        (c, r + 0.5),
        (c, r + 1.0),
        (c + 0.5, r + 1.0),
        (c + 1.0, r + 1.0),
        (c + 1.0, r + 0.5),
        (c + 1.0, r),
        (c + 0.5, r),
    ];
    let mut x: Vec<f64> = idx.iter().map(|p| p.0).collect();
    let mut y: Vec<f64> = idx.iter().map(|p| p.1).collect();
    let ok = rwarp::Transformer::transform(&b.t, false, &mut x, &mut y);
    if ok.iter().any(|o| !o) {
        return None;
    }
    let mut q = [[0.0; 2]; 8];
    for i in 0..8 {
        q[i] = [x[i], y[i]];
    }
    Some(q)
}

/// World-space cover of one polygon: every cell near it clipped by it.
fn world_cover(b: &Burner, rings: &Rings) -> Vec<CellCover> {
    let Some(win) = window(b, rings) else {
        return Vec::new();
    };
    let env = envelope(rings);
    let mut out = BTreeMap::new();
    for r in win.rows.clone() {
        for c in win.cols.clone() {
            let Some(q) = outline(b, r, c) else { continue };
            let qe = envelope_of(&q);
            if qe[0] > env[2] || qe[2] < env[0] || qe[1] > env[3] || qe[3] < env[1] {
                continue;
            }
            let cell = area(&q).abs();
            if cell <= 0.0 {
                continue;
            }
            let f = overlap(rings, &q) / cell;
            if f > 1e-6 {
                out.insert([r, c], f.min(1.0));
            }
        }
    }
    out.into_iter().collect()
}

/// Area of a polygon (rings, exterior first) inside a simple cell
/// outline: the outline is fanned from its first point into triangles,
/// each signed by its winding, and the polygon is clipped by each
/// (Sutherland-Hodgman, exact for a convex clip and any simple subject).
/// The signed sum is the overlap for any simple outline, convex or not.
pub(crate) fn overlap(rings: &Rings, q: &[[f64; 2]]) -> f64 {
    let mut total = 0.0;
    let o = q[0];
    for i in 1..q.len() - 1 {
        let tri = [o, q[i], q[i + 1]];
        let s = tri_area(&tri);
        if s == 0.0 {
            continue;
        }
        let tri = if s > 0.0 { tri } else { [o, q[i + 1], q[i]] };
        let mut a = 0.0;
        for (k, ring) in rings.iter().enumerate() {
            let clipped = area(&clip(ring, &tri)).abs();
            a += if k == 0 { clipped } else { -clipped };
        }
        total += s.signum() * a;
    }
    total.abs()
}

fn tri_area(t: &[[f64; 2]; 3]) -> f64 {
    0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]))
}

/// Shoelace area, signed (anticlockwise positive); closing point optional.
pub(crate) fn area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    s / 2.0
}

/// `subject` clipped by the anticlockwise triangle `tri`.
fn clip(subject: &[[f64; 2]], tri: &[[f64; 2]; 3]) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = subject.to_vec();
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    for i in 0..3 {
        let (a, b) = (tri[i], tri[(i + 1) % 3]);
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let input = std::mem::take(&mut out);
        let n = input.len();
        for j in 0..n {
            let (p, q) = (input[j], input[(j + 1) % n]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

fn envelope(rings: &Rings) -> [f64; 4] {
    rings
        .first()
        .map(|r| envelope_of(r))
        .unwrap_or([f64::MAX, f64::MAX, f64::MIN, f64::MIN])
}

fn envelope_of(p: &[[f64; 2]]) -> [f64; 4] {
    p.iter()
        .fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |e, v| {
            [
                e[0].min(v[0]),
                e[1].min(v[1]),
                e[2].max(v[0]),
                e[3].max(v[1]),
            ]
        })
}

/// Even-odd point in polygon over all rings.
fn inside(rings: &Rings, p: [f64; 2]) -> bool {
    let mut c = false;
    for r in rings {
        let n = r.len();
        if n < 3 {
            continue;
        }
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (r[i], r[j]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                c = !c;
            }
            j = i;
        }
    }
    c
}

/// Cells (as a set) in a cover, for comparisons.
pub fn cover_cells(cover: &[CellCover]) -> BTreeSet<[u64; 2]> {
    cover.iter().map(|c| c.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_of_unit_square_and_shifted_square() {
        let sq: Rings = vec![vec![
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
            [0.5, 1.5],
            [0.5, 0.5],
        ]];
        let cell = [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 0.0],
            [0.5, 0.0],
        ];
        assert!((overlap(&sq, &cell) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn overlap_with_a_non_convex_cell_and_a_hole() {
        // an L-shaped outline (a chevron of 8 points) and a polygon with a hole
        let cell = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [0.0, 1.5],
            [0.0, 1.0],
        ];
        let poly: Rings = vec![
            vec![
                [-1.0, -1.0],
                [3.0, -1.0],
                [3.0, 3.0],
                [-1.0, 3.0],
                [-1.0, -1.0],
            ],
            vec![
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.75, 0.25],
                [0.25, 0.25],
            ],
        ];
        assert!((overlap(&poly, &cell) - (3.0 - 0.25)).abs() < 1e-12);
    }
}
