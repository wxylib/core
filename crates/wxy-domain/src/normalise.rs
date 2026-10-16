//! Normalisation: find a cheaper, equivalent mapping for what was stored.
//!
//! Readers record the stored form (the "costume" in regular-in-what).
//! Normalisation tests whether it is really a lower rung, within a stated
//! tolerance, and if so adds the cheaper mapping *alongside* the stored one
//! and marks it primary. Every test returns a [`Report`], accepted or not,
//! so it is always visible why a grid was labelled what it was.
//!
//! Three tests are provided:
//!
//! - [`regular_axis`]: a 1D coordinate vector is evenly spaced (R2 to R1;
//!   GDAL's `IsRegularlySpaced` and `GuessGeoTransform`);
//! - [`regular_axis_through`]: evenly spaced after a 1D change of frame
//!   (AVISO's latitude vector is regular in Mercator y);
//! - [`regular_grid_through`]: 2D coordinate arrays are an affine grid
//!   after a change of frame (cryptic curvilinear to R1, the NSIDC case).
//!
//! The frame change is a plain function supplied by the caller, so this
//! crate needs no CRS backend.

use std::fmt;

use crate::affine::Affine;
use crate::domain::Domain;
use crate::frame::Frame;
use crate::mapping::{CellConvention, Mapping};
use crate::positions::{CoordArray, Positions, Rung};

/// How far stored coordinates may sit from the fitted mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// One unit in the last place of `f32` at the largest magnitude in the
    /// coordinates: what rounding to 32-bit storage can produce.
    Float32,
    /// As `Float32`, for `f64`, scaled by 16 for accumulated arithmetic.
    Float64,
    /// An absolute distance in the frame's units.
    Absolute(f64),
    /// A fraction of one step (cell size).
    RelativeToStep(f64),
}

impl Tolerance {
    fn value(&self, max_abs: f64, step: f64) -> f64 {
        match *self {
            Tolerance::Float32 => ulp_f32(max_abs),
            Tolerance::Float64 => 16.0 * max_abs.max(f64::MIN_POSITIVE) * f64::EPSILON,
            Tolerance::Absolute(t) => t,
            Tolerance::RelativeToStep(f) => f * step.abs(),
        }
    }
}

/// Spacing of `f32` values at magnitude `m`.
fn ulp_f32(m: f64) -> f64 {
    let m = m.abs() as f32;
    if m == 0.0 || !m.is_finite() {
        return f32::MIN_POSITIVE as f64;
    }
    let next = f32::from_bits(m.to_bits() + 1);
    (next - m) as f64
}

/// The outcome of one normalisation test.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Domain dims the test covered.
    pub dims: Vec<usize>,
    pub from: Rung,
    /// The rung found, when accepted.
    pub to: Option<Rung>,
    /// The frame the stored coordinates were in.
    pub frame: Frame,
    /// The frame the test found regularity in, when it changed frame.
    pub through: Option<String>,
    /// Largest distance between a stored coordinate and the fitted mapping,
    /// in the tested frame's units.
    pub max_deviation: f64,
    /// The same, as a fraction of one step.
    pub max_deviation_steps: f64,
    /// The tolerance applied, in the tested frame's units.
    pub tolerance: f64,
    pub accepted: bool,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = if self.accepted {
            "regular"
        } else {
            "not regular"
        };
        write!(
            f,
            "dims {:?}: {:?} -> {}{}; max deviation {:.3e} ({:.2e} steps), tolerance {:.3e}",
            self.dims,
            self.from,
            verdict,
            self.through
                .as_ref()
                .map(|t| format!(" in {t}"))
                .unwrap_or_default(),
            self.max_deviation,
            self.max_deviation_steps,
            self.tolerance
        )
    }
}

/// Fit `origin + i * step` to a coordinate vector through its end values,
/// returning `(origin_at_index_0, step, max_deviation)`.
fn fit_line(c: &[f64], cell: CellConvention) -> Option<(f64, f64, f64)> {
    let n = c.len();
    if n < 2 || c.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let step = (c[n - 1] - c[0]) / (n - 1) as f64;
    if step == 0.0 {
        return None;
    }
    let max_dev = c
        .iter()
        .enumerate()
        .map(|(k, v)| (v - (c[0] + k as f64 * step)).abs())
        .fold(0.0, f64::max);
    let origin = match cell {
        CellConvention::Centre => c[0] - 0.5 * step,
        CellConvention::Edge => c[0],
    };
    Some((origin, step, max_dev))
}

/// Test whether a 1D coordinate vector is evenly spaced.
pub fn regular_axis(
    coords: &[f64],
    cell: CellConvention,
    tolerance: Tolerance,
) -> (Option<Positions>, Report) {
    regular_axis_inner(coords, cell, tolerance, None)
}

/// Test whether a 1D coordinate vector is evenly spaced after applying `f`
/// to every value (for example latitude to Mercator y). `through` names the
/// frame `f` maps into. On success the returned positions are in that frame.
pub fn regular_axis_through(
    coords: &[f64],
    f: impl Fn(f64) -> f64,
    through: &str,
    cell: CellConvention,
    tolerance: Tolerance,
) -> (Option<Positions>, Report) {
    let mapped: Vec<f64> = coords.iter().map(|&v| f(v)).collect();
    regular_axis_inner(&mapped, cell, tolerance, Some(through.to_string()))
}

fn regular_axis_inner(
    coords: &[f64],
    cell: CellConvention,
    tolerance: Tolerance,
    through: Option<String>,
) -> (Option<Positions>, Report) {
    let max_abs = coords.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut report = Report {
        dims: vec![],
        from: Rung::Rectilinear,
        to: None,
        frame: Frame::Index,
        through,
        max_deviation: f64::NAN,
        max_deviation_steps: f64::NAN,
        tolerance: f64::NAN,
        accepted: false,
    };
    let Some((origin, step, max_dev)) = fit_line(coords, cell) else {
        return (None, report);
    };
    let tol = tolerance.value(max_abs, step);
    report.max_deviation = max_dev;
    report.max_deviation_steps = max_dev / step.abs();
    report.tolerance = tol;
    if max_dev <= tol {
        report.accepted = true;
        report.to = Some(Rung::Regular);
        (Some(Positions::Regular1 { origin, step }), report)
    } else {
        (None, report)
    }
}

/// Test whether 2D coordinate arrays (centres, shaped `[nrow, ncol]`) form
/// an affine grid after mapping each `(x, y)` pair through `f`. `through`
/// names the frame `f` maps into. `None` from `f` fails the test.
///
/// On success returns the affine transform in the `through` frame, for
/// index order `[row, col]` as [`Positions::Affine`] expects.
pub fn regular_grid_through(
    x: &CoordArray,
    y: &CoordArray,
    f: impl Fn(f64, f64) -> Option<(f64, f64)>,
    through: &str,
    tolerance: Tolerance,
) -> (Option<Affine>, Report) {
    let mut report = Report {
        dims: vec![],
        from: Rung::Curvilinear,
        to: None,
        frame: Frame::Index,
        through: Some(through.to_string()),
        max_deviation: f64::NAN,
        max_deviation_steps: f64::NAN,
        tolerance: f64::NAN,
        accepted: false,
    };
    if x.shape.len() != 2 || x.shape != y.shape || x.shape[0] < 2 || x.shape[1] < 2 {
        return (None, report);
    }
    let (nr, nc) = (x.shape[0], x.shape[1]);
    let mut px = Vec::with_capacity(nr * nc);
    let mut py = Vec::with_capacity(nr * nc);
    for k in 0..nr * nc {
        match f(x.values[k], y.values[k]) {
            Some((a, b)) if a.is_finite() && b.is_finite() => {
                px.push(a);
                py.push(b);
            }
            _ => return (None, report),
        }
    }
    let at = |r: usize, c: usize| (px[r * nc + c], py[r * nc + c]);
    let (x00, y00) = at(0, 0);
    let (x01, y01) = at(0, nc - 1);
    let (x10, y10) = at(nr - 1, 0);
    // Per-index steps along col and row, from the three corner centres.
    let dxc = (x01 - x00) / (nc - 1) as f64;
    let dyc = (y01 - y00) / (nc - 1) as f64;
    let dxr = (x10 - x00) / (nr - 1) as f64;
    let dyr = (y10 - y00) / (nr - 1) as f64;
    let mut max_dev = 0.0f64;
    for r in 0..nr {
        for c in 0..nc {
            let (a, b) = at(r, c);
            let ex = x00 + c as f64 * dxc + r as f64 * dxr;
            let ey = y00 + c as f64 * dyc + r as f64 * dyr;
            max_dev = max_dev.max((a - ex).hypot(b - ey));
        }
    }
    let step = dxc.hypot(dyc).min(dxr.hypot(dyr));
    let max_abs = px.iter().chain(&py).fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = tolerance.value(max_abs, step);
    report.max_deviation = max_dev;
    report.max_deviation_steps = max_dev / step;
    report.tolerance = tol;
    if max_dev > tol {
        return (None, report);
    }
    report.accepted = true;
    report.to = Some(Rung::Regular);
    // Centre of cell (0, 0) is at index (0.5, 0.5); move to the outer edge.
    let x0 = x00 - 0.5 * dxc - 0.5 * dxr;
    let y0 = y00 - 0.5 * dyc - 0.5 * dyr;
    (Some(Affine([x0, dxc, dxr, y0, dyc, dyr])), report)
}

/// The conservative default (design doc 02, gate 2 default 3): every 1D
/// rectilinear mapping that is evenly spaced within `Float32` tolerance
/// gains a regular mapping, marked primary. Stored mappings are kept.
pub fn normalise_domain(domain: &mut Domain) -> Vec<Report> {
    normalise_domain_with(domain, Tolerance::Float32)
}

/// As [`normalise_domain`], with a chosen tolerance.
pub fn normalise_domain_with(domain: &mut Domain, tolerance: Tolerance) -> Vec<Report> {
    let mut reports = Vec::new();
    let candidates: Vec<usize> = domain
        .mappings
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(&m.positions, Positions::Rectilinear(v) if v.len() == 1))
        .map(|(i, _)| i)
        .collect();
    for i in candidates {
        let m = &domain.mappings[i];
        let Positions::Rectilinear(axes) = &m.positions else {
            unreachable!()
        };
        let (found, mut report) = regular_axis(&axes[0], m.cell, tolerance);
        report.dims = m.dims.clone();
        report.frame = m.frame.clone();
        if let Some(p) = found {
            let derived = Mapping::new(m.dims.clone(), p, m.frame.clone())
                .with_cell(m.cell)
                .with_provenance("normalise");
            let j = domain.add_mapping(derived);
            domain.set_primary(j);
        }
        reports.push(report);
    }
    reports
}

/// An affine transform from the primary mappings of a row dim and a col
/// dim, when both are R1 (`Regular1`) in the same frame.
pub fn affine_from_axes(domain: &Domain, row: usize, col: usize) -> Option<(Affine, Frame)> {
    let r = domain.primary_for(row)?;
    let c = domain.primary_for(col)?;
    if r.frame != c.frame {
        return None;
    }
    match (&r.positions, &c.positions) {
        (
            Positions::Regular1 {
                origin: y0,
                step: dy,
            },
            Positions::Regular1 {
                origin: x0,
                step: dx,
            },
        ) => Some((Affine([*x0, *dx, 0.0, *y0, 0.0, *dy]), r.frame.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_f32_at_180() {
        // f32 values between 128 and 256 are spaced 2^-16 apart.
        assert_eq!(ulp_f32(180.0), 2f64.powi(-16));
    }

    #[test]
    fn exact_axis_is_regular() {
        let c: Vec<f64> = (0..10).map(|i| 0.5 + i as f64).collect();
        let (p, r) = regular_axis(&c, CellConvention::Centre, Tolerance::Float64);
        assert!(r.accepted);
        match p.unwrap() {
            Positions::Regular1 { origin, step } => {
                assert_eq!(origin, 0.0);
                assert_eq!(step, 1.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn stretched_axis_is_not_regular() {
        let c: Vec<f64> = (0..10).map(|i| (i as f64).powf(1.2)).collect();
        let (p, r) = regular_axis(&c, CellConvention::Centre, Tolerance::Float32);
        assert!(p.is_none());
        assert!(!r.accepted);
        assert!(r.max_deviation > r.tolerance);
    }
}
