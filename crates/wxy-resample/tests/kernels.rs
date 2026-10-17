//! Kernel properties that hold whatever GDAL's arithmetic order: exact at
//! cell centres, constant on constant fields, invalid cells skipped. The
//! cell-for-cell comparison with GDAL is in wxy-warp's oracle tests.

use wxy_resample::{write_density, Kernel, Method, Sample, Scale, Window};

fn ramp(w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| ((i % w) * 10 + (i / w)) as f32)
        .collect()
}

#[test]
fn rounding_and_clamping_follow_gdal() {
    assert_eq!(i16::round_from(2.5), 3);
    assert_eq!(i16::round_from(-2.5), -2);
    assert_eq!(u8::round_from(2.5), 3);
    assert_eq!(u8::clamp_from(300.0), 255);
    assert_eq!(u8::clamp_from(-5.0), 0);
    assert_eq!(i16::clamp_from(-40000.0), i16::MIN);
    assert_eq!(f32::clamp_from(1.25), 1.25);
}

#[test]
fn nudge_steps_off_a_value() {
    assert_eq!(5u8.nudge(), 4);
    assert_eq!(0u8.nudge(), 1);
    assert_eq!(i16::MIN.nudge(), i16::MIN + 1);
    assert!((-9999.0f32).nudge() > -9999.0);
    assert!(f32::MAX.nudge() < f32::MAX);
}

#[test]
fn ratios_snap_as_gdal_does() {
    let [x, y] = Scale::ratios([145, 145], [720, 360], [0.0, 0.0]);
    assert_eq!(x, 0.2);
    assert!((y - 145.0 / 360.0).abs() < 1e-15);
    // within the kernel margin of the target size: 1
    let [x, _] = Scale::ratios([100, 100], [103, 100], [4.0, 0.0]);
    assert_eq!(x, 1.0);
    // a reciprocal within 0.05 of a whole number snaps to it
    let [x, _] = Scale::ratios([100, 100], [302, 100], [0.0, 0.0]);
    assert_eq!(x, 1.0 / 3.0);
}

#[test]
fn radius_widens_for_coarse_targets() {
    let s = Scale::new(Method::Cubic, 0.2, 1.0);
    assert_eq!(
        (s.x_radius, s.y_radius, s.x_first, s.y_first),
        (10, 2, -9, -1)
    );
    assert!(!s.four_samples());
    let s = Scale::new(Method::Bilinear, 1.0, 0.96);
    assert_eq!(
        (s.x_radius, s.y_radius, s.x_first, s.y_first),
        (1, 2, -1, -2)
    );
    assert!(s.four_samples());
}

#[test]
fn cell_centres_are_exact() {
    let (w, h) = (12, 9);
    let data = ramp(w, h);
    let win = Window {
        data: &data,
        xsize: w,
        ysize: h,
        valid: None,
    };
    for method in [Method::Bilinear, Method::Cubic] {
        let mut k = Kernel::new(method, Scale::new(method, 1.0, 1.0));
        for (cx, cy) in [(3usize, 4usize), (6, 2), (8, 6)] {
            let off = cx + cy * w;
            let v = k.value(&win, off, cx as f64 + 0.5, cy as f64 + 0.5);
            assert_eq!(v, data[off], "{method:?} at {cx},{cy}");
            let (d, r) = k.density(&win, off, cx as f64 + 0.5, cy as f64 + 0.5);
            assert_eq!((d, r as f32), (1.0, data[off]));
        }
    }
}

#[test]
fn scaled_kernels_keep_a_constant_field() {
    let (w, h) = (40, 30);
    let data = vec![7.0f32; w * h];
    let win = Window {
        data: &data,
        xsize: w,
        ysize: h,
        valid: None,
    };
    for method in [Method::Bilinear, Method::Cubic] {
        let mut k = Kernel::new(method, Scale::new(method, 0.2, 0.3));
        for (x, y) in [(20.3, 15.7), (2.1, 1.2), (38.9, 29.4)] {
            let off = x as usize + y as usize * w;
            assert!((k.value(&win, off, x, y) - 7.0).abs() < 1e-5);
            let (d, r) = k.density(&win, off, x, y);
            assert!((d - 1.0).abs() < 1e-9 && (r - 7.0).abs() < 1e-5);
        }
    }
}

#[test]
fn invalid_cells_are_skipped() {
    let (w, h) = (20, 20);
    let mut data = vec![5.0f32; w * h];
    let mut valid = vec![true; w * h];
    for i in 0..w * h {
        if i % w < 10 {
            data[i] = -9999.0;
            valid[i] = false;
        }
    }
    let win = Window {
        data: &data,
        xsize: w,
        ysize: h,
        valid: Some(&valid),
    };
    for method in [Method::Bilinear, Method::Cubic] {
        for scale in [1.0, 0.25] {
            let mut k = Kernel::new(method, Scale::new(method, scale, scale));
            let (x, y) = (10.2, 10.5);
            let (d, r) = k.density(&win, 10 + 10 * w, x, y);
            assert!(d > 0.0, "{method:?} {scale}: density {d}");
            assert!((r - 5.0).abs() < 1e-9, "{method:?} {scale}: value {r}");
        }
    }
}

#[test]
fn partial_density_mixes_into_valid_targets_only() {
    let mut out = vec![-9999.0f32, 10.0];
    // an invalid target cell takes the value whole
    write_density(&mut out, Some(&[false, true]), 0, 0.5, 4.0, Some(-9999.0));
    assert_eq!(out[0], 4.0);
    // a valid one is mixed with by density
    write_density(&mut out, Some(&[false, true]), 1, 0.5, 4.0, Some(-9999.0));
    assert_eq!(out[1], 7.0);
    // a result equal to nodata is stepped off it
    let mut out = vec![0i16];
    write_density(&mut out, None, 0, 1.0, -32768.0, Some(-32768.0));
    assert_eq!(out[0], -32767);
}
