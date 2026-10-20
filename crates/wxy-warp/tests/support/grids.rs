//! Grids that are not a regular lattice, and polygons to burn on them,
//! shared by tests/nonaffine.rs and examples/nonaffine_cover.rs. All are
//! in EPSG:3031 metres near the origin (any projected CRS would do; the
//! polygons are in the same CRS, so only the grid's geometry matters).

#![allow(dead_code)]

use wxy_domain::{Affine, CellConvention, CoordArray, Dim, Domain, Frame, Mapping, Positions};
use wxy_warp::Rings;

pub const CRS: &str = "EPSG:3031";

pub struct Case {
    pub name: &'static str,
    pub about: &'static str,
    pub domain: Domain,
}

fn dims(ny: usize, nx: usize) -> Domain {
    Domain::new(vec![Dim::new("y", ny as u64), Dim::new("x", nx as u64)])
}

/// 40 by 30 cells of 1 km, turned 30 degrees anticlockwise about the top
/// left corner.
pub fn rotated() -> Case {
    let (t, d) = (30f64.to_radians(), 1000.0);
    let mut domain = dims(30, 40);
    let gt = [
        0.0,
        d * t.cos(),
        d * t.sin(),
        30_000.0,
        d * t.sin(),
        -d * t.cos(),
    ];
    let k = domain.add_mapping(Mapping::new(
        vec![0, 1],
        Positions::Affine(Affine(gt)),
        Frame::crs(CRS),
    ));
    domain.set_primary(k);
    Case {
        name: "rotated",
        about: "affine, rotated 30 degrees",
        domain,
    }
}

/// Cell walls of a stretched axis: `n` cells from `w0`, each `r` times
/// the one before, starting at width `d0`.
fn stretched(n: usize, w0: f64, d0: f64, r: f64) -> Vec<f64> {
    let mut w = vec![w0];
    let mut d = d0;
    for _ in 0..n {
        let last = *w.last().unwrap();
        w.push(last + d);
        d *= r;
    }
    w
}

fn rectilinear_with(walls_x: &[f64], walls_y: &[f64], cell: CellConvention) -> Domain {
    let (nx, ny) = (walls_x.len() - 1, walls_y.len() - 1);
    let mid = |w: &[f64]| {
        w.windows(2)
            .map(|p| (p[0] + p[1]) / 2.0)
            .collect::<Vec<_>>()
    };
    let (xs, ys) = match cell {
        CellConvention::Edge => (walls_x.to_vec(), walls_y.to_vec()),
        _ => (mid(walls_x), mid(walls_y)),
    };
    let mut domain = dims(ny, nx);
    let ky = domain.add_mapping(
        Mapping::new(vec![0], Positions::Rectilinear(vec![ys]), Frame::crs(CRS)).with_cell(cell),
    );
    let kx = domain.add_mapping(
        Mapping::new(vec![1], Positions::Rectilinear(vec![xs]), Frame::crs(CRS)).with_cell(cell),
    );
    domain.set_primary(ky);
    domain.set_primary(kx);
    domain
}

/// x walls growing 8 percent a cell (a stretched model grid), y walls
/// shrinking then growing; rows run north to south.
pub fn rect_walls() -> (Vec<f64>, Vec<f64>) {
    let x = stretched(36, 0.0, 300.0, 1.08);
    let mut y: Vec<f64> = stretched(14, 0.0, 2000.0, 0.85);
    let top = *y.last().unwrap();
    for w in stretched(14, top, 2000.0 * 0.85f64.powi(14), 1.0 / 0.85)
        .into_iter()
        .skip(1)
    {
        y.push(w);
    }
    y.reverse();
    (x, y)
}

/// Rectilinear, coordinates at the cell walls (one more than cells).
pub fn rect_edges() -> Case {
    let (x, y) = rect_walls();
    Case {
        name: "rect-edges",
        about: "rectilinear, walls given",
        domain: rectilinear_with(&x, &y, CellConvention::Edge),
    }
}

/// The same cells, coordinates at their centres (the walls' midpoints).
pub fn rect_centres() -> Case {
    let (x, y) = rect_walls();
    Case {
        name: "rect-centres",
        about: "rectilinear, centres given",
        domain: rectilinear_with(&x, &y, CellConvention::Centre),
    }
}

/// An annular sector (a polar grid): radius 8 to 40 km across columns,
/// angle 100 to 10 degrees down rows, coordinates at `cell` positions.
pub fn fan_with(cell: CellConvention, nx: usize, ny: usize) -> Domain {
    let e = usize::from(cell == CellConvention::Edge);
    let (cx, cy) = (nx + e, ny + e);
    let off = if e == 1 { 0.0 } else { 0.5 };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for j in 0..cy {
        for i in 0..cx {
            let r = 8_000.0 + 32_000.0 * (i as f64 + off) / nx as f64;
            let a = (100.0 - 90.0 * (j as f64 + off) / ny as f64).to_radians();
            xs.push(r * a.cos());
            ys.push(r * a.sin());
        }
    }
    let mut domain = dims(ny, nx);
    let k = domain.add_mapping(
        Mapping::new(
            vec![0, 1],
            Positions::Curvilinear(vec![
                CoordArray::new(vec![cy, cx], xs),
                CoordArray::new(vec![cy, cx], ys),
            ]),
            Frame::crs(CRS),
        )
        .with_cell(cell),
    );
    domain.set_primary(k);
    domain
}

pub fn fan_centres() -> Case {
    Case {
        name: "fan-centres",
        about: "curvilinear polar sector, centres given",
        domain: fan_with(CellConvention::Centre, 24, 30),
    }
}

pub fn fan_edges() -> Case {
    Case {
        name: "fan-edges",
        about: "curvilinear polar sector, corners given",
        domain: fan_with(CellConvention::Edge, 24, 30),
    }
}

pub fn all() -> Vec<Case> {
    vec![
        rotated(),
        rect_edges(),
        rect_centres(),
        fan_centres(),
        fan_edges(),
    ]
}

fn closed(mut r: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    r.push(r[0]);
    r
}

/// A five-pointed star with a square hole, centred at `c`, outer radius
/// `r`, anticlockwise exterior.
pub fn star(c: [f64; 2], r: f64) -> Rings {
    let ext = (0..10)
        .map(|k| {
            let a = (90.0 + 36.0 * k as f64).to_radians();
            let rr = if k % 2 == 0 { r } else { 0.45 * r };
            [c[0] + rr * a.cos(), c[1] + rr * a.sin()]
        })
        .collect();
    let h = 0.15 * r;
    let hole = vec![
        [c[0] - h, c[1] - h],
        [c[0] - h, c[1] + h],
        [c[0] + h, c[1] + h],
        [c[0] + h, c[1] - h],
    ];
    vec![closed(ext), closed(hole)]
}

/// A long thin quadrilateral, the kind of edge that bends most on a
/// curvilinear grid.
pub fn sliver(a: [f64; 2], b: [f64; 2], w: f64) -> Rings {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = dx.hypot(dy);
    let (nx, ny) = (-dy / l * w / 2.0, dx / l * w / 2.0);
    vec![closed(vec![
        [a[0] + nx, a[1] + ny],
        [a[0] - nx, a[1] - ny],
        [b[0] - nx, b[1] - ny],
        [b[0] + nx, b[1] + ny],
    ])]
}

/// Polygons for a case, all inside its grid (so each covers its own area).
pub fn polygons(name: &str) -> Vec<Rings> {
    match name {
        "rotated" => vec![
            star([24_820.0, 27_010.0], 9_000.0),
            sliver([12_000.0, 18_000.0], [33_000.0, 34_000.0], 1_500.0),
        ],
        "rect-edges" | "rect-centres" => vec![
            star([12_000.0, 12_500.0], 7_500.0),
            sliver([3_000.0, 3_000.0], [36_000.0, 20_000.0], 1_200.0),
        ],
        _ => vec![
            star([8_200.0, 22_550.0], 9_000.0),
            sliver([-1_000.0, 12_000.0], [36_000.0, 10_000.0], 1_500.0),
        ],
    }
}
