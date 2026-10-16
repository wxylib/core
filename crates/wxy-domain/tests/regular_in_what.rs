//! Cases from "Regular in what?" (hypertidy/regular-in-what), as tests of
//! the stage S1 domain model. Each builds the stored form a real product
//! uses and checks that normalisation recovers the regular grid under it.

use wxy_domain::normalise::{
    affine_from_axes, normalise_domain, regular_axis, regular_axis_through, regular_grid_through,
    Tolerance,
};
use wxy_domain::{
    Affine, CellConvention, CoordArray, Dim, Domain, Frame, Mapping, Positions, Rung,
};

fn to_f32(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x as f32 as f64).collect()
}

/// Chapter 3, degenerate rectilinear: a regular 0.01 degree longitude axis
/// with 36000 centres (as MUR GHRSST), stored as float32. Steps vary by
/// about 0.15% of a cell, so an exact-equality test calls it irregular;
/// a float32 tolerance does not.
#[test]
fn ghrsst_float32_axis_is_regular() {
    let lon: Vec<f64> = (0..36000).map(|i| -179.995 + 0.01 * i as f64).collect();
    let stored = to_f32(&lon);

    let steps: Vec<f64> = stored.windows(2).map(|w| w[1] - w[0]).collect();
    let (lo, hi) = steps
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &s| (a.min(s), b.max(s)));
    assert!(lo < 0.009995 && hi > 0.010009, "steps {lo} to {hi}");

    let (_, exact) = regular_axis(&stored, CellConvention::Centre, Tolerance::Absolute(0.0));
    assert!(!exact.accepted);

    let (p, report) = regular_axis(&stored, CellConvention::Centre, Tolerance::Float32);
    assert!(report.accepted, "{report}");
    let Some(Positions::Regular1 { origin, step }) = p else {
        panic!()
    };
    assert!((origin + 180.0).abs() < 1e-4, "origin {origin}");
    assert!((step - 0.01).abs() < 1e-9, "step {step}");
}

/// Chapter 3, hidden projections: AVISO's latitude vector is not evenly
/// spaced, but it is regular in Mercator y. 915 rows of 1/3 degree.
#[test]
fn aviso_latitude_is_regular_in_mercator() {
    let n = 915;
    let step = (1.0f64 / 3.0).to_radians();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 - (n as f64 - 1.0) / 2.0) * step)
        .collect();
    let lat: Vec<f64> = y.iter().map(|&v| v.sinh().atan().to_degrees()).collect();

    let (p, plain) = regular_axis(&lat, CellConvention::Centre, Tolerance::Float64);
    assert!(
        p.is_none(),
        "latitude itself should not be regular: {plain}"
    );

    let merc_y = |deg: f64| {
        (std::f64::consts::FRAC_PI_4 + deg.to_radians() / 2.0)
            .tan()
            .ln()
    };
    let (p, report) = regular_axis_through(
        &lat,
        merc_y,
        "mercator_y_radians",
        CellConvention::Centre,
        Tolerance::Float64,
    );
    assert!(report.accepted, "{report}");
    assert!(report.max_deviation_steps < 1e-10, "{report}");
    let Some(Positions::Regular1 { step: s, .. }) = p else {
        panic!()
    };
    assert!((s - step).abs() < 1e-12);
}

/// Spherical polar stereographic, south pole, true scale at 70 S. Enough
/// to build and test the NSIDC case; the test does not depend on it being
/// EPSG:3412's exact ellipsoidal form.
const R: f64 = 6_378_273.0;
fn k0() -> f64 {
    (1.0 + 70f64.to_radians().sin()) / 2.0
}
fn stere_s_inverse(x: f64, y: f64) -> (f64, f64) {
    let rho = x.hypot(y);
    let lat = 2.0 * (rho / (2.0 * R * k0())).atan() - std::f64::consts::FRAC_PI_2;
    (x.atan2(y).to_degrees(), lat.to_degrees())
}
fn stere_s_forward(lon: f64, lat: f64) -> Option<(f64, f64)> {
    let rho = 2.0 * R * k0() * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan();
    let l = lon.to_radians();
    Some((rho * l.sin(), rho * l.cos()))
}

/// Chapter 1: a 316 by 332 sea ice grid stored as 2D float32 longitude and
/// latitude (209,824 values) is the NSIDC 25 km south grid: six numbers.
#[test]
fn nsidc_cryptic_curvilinear_is_regular_in_its_projection() {
    let extent = [-3_950_000.0, 3_950_000.0, -3_950_000.0, 4_350_000.0];
    let (nc, nr) = (316usize, 332usize);
    let truth = Affine::from_extent(extent, [nc, nr]);

    let mut lon = Vec::with_capacity(nr * nc);
    let mut lat = Vec::with_capacity(nr * nc);
    for r in 0..nr {
        for c in 0..nc {
            let (x, y) = truth.forward(c as f64 + 0.5, r as f64 + 0.5);
            let (a, b) = stere_s_inverse(x, y);
            lon.push(a as f32 as f64);
            lat.push(b as f32 as f64);
        }
    }
    let lon = CoordArray::new(vec![nr, nc], lon);
    let lat = CoordArray::new(vec![nr, nc], lat);

    let stored = Positions::Curvilinear(vec![lon.clone(), lat.clone()]);
    assert_eq!(stored.stored_values(), 2 * 316 * 332);
    assert_eq!(stored.rung(), Rung::Curvilinear);

    let (affine, report) = regular_grid_through(
        &lon,
        &lat,
        stere_s_forward,
        "polar_stereographic_south",
        Tolerance::RelativeToStep(1e-3),
    );
    assert!(report.accepted, "{report}");
    let affine = affine.unwrap();
    assert_eq!(Positions::Affine(affine).stored_values(), 6);
    let got = affine.extent([nc, nr]);
    for k in 0..4 {
        assert!((got[k] - extent[k]).abs() < 50.0, "extent {got:?}");
    }
    // A few tens of metres on a 25 km grid is float32 noise.
    assert!(report.max_deviation < 25.0, "{report}");
}

/// The common case: OISST-like 1D centre coordinates on a quarter degree
/// grid normalise to two R1 axes and one affine, and the stored mappings
/// are kept.
#[test]
fn oisst_axes_normalise_to_an_affine() {
    let lon = to_f32(
        &(0..1440)
            .map(|i| 0.125 + 0.25 * i as f64)
            .collect::<Vec<_>>(),
    );
    let lat = to_f32(
        &(0..720)
            .map(|i| 89.875 - 0.25 * i as f64)
            .collect::<Vec<_>>(),
    );
    let frame = Frame::crs("OGC:CRS84");

    let mut d = Domain::new(vec![Dim::new("lat", 720), Dim::new("lon", 1440)]);
    d.add_mapping(Mapping::new(
        vec![0],
        Positions::Rectilinear(vec![lat]),
        frame.clone(),
    ));
    d.add_mapping(Mapping::new(
        vec![1],
        Positions::Rectilinear(vec![lon]),
        frame.clone(),
    ));
    assert_eq!(d.rung_for(1), Rung::Rectilinear);

    let reports = normalise_domain(&mut d);
    assert_eq!(reports.len(), 2);
    assert!(reports.iter().all(|r| r.accepted), "{reports:?}");
    assert_eq!(d.mappings.len(), 4, "stored mappings are kept");
    assert_eq!(d.rung_for(0), Rung::Regular);
    assert_eq!(d.rung_for(1), Rung::Regular);

    let (a, f) = affine_from_axes(&d, 0, 1).unwrap();
    assert_eq!(f, frame);
    let e = a.extent([1440, 720]);
    let want = [0.0, 360.0, -90.0, 90.0];
    for k in 0..4 {
        assert!((e[k] - want[k]).abs() < 1e-3, "extent {e:?}");
    }
}

/// Chapter 2: a rotated-pole grid keeps both mappings. The regular one in
/// the rotated frame is primary; the 2D geographic arrays ride along.
#[test]
fn rotated_pole_keeps_both_frames() {
    let mut d = Domain::new(vec![Dim::new("rlat", 2), Dim::new("rlon", 2)]);
    let rotated = Frame::Other("rotated_latitude_longitude grid_north_pole=(-170, 6.55)".into());
    let geo = Frame::crs("OGC:CRS84");
    let a = d.add_mapping(Mapping::new(
        vec![0, 1],
        Positions::Affine(Affine::from_extent([-1.0, 1.0, -1.0, 1.0], [2, 2])),
        rotated.clone(),
    ));
    d.add_mapping(Mapping::new(
        vec![0, 1],
        Positions::Curvilinear(vec![
            CoordArray::new(vec![2, 2], vec![10.0, 11.0, 10.2, 11.2]),
            CoordArray::new(vec![2, 2], vec![50.0, 50.1, 49.0, 49.1]),
        ]),
        geo,
    ));
    d.set_primary(a);
    assert_eq!(d.primary_for(1).unwrap().frame, rotated);
    assert_eq!(d.mappings_for(1).count(), 2);
}
