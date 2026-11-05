//! The crate against its own small set (tests/data/mini), answered by
//! GDAL 3.8.4 through scripts/oraclecase.py in wxylib/fixtures.

use oracle_case::{compare, Error, Set, Status, Tolerance};

fn set() -> Set {
    Set::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/mini")).unwrap()
}

#[test]
fn reads_questions() {
    let s = set();
    assert_eq!(s.name.as_deref(), Some("mini"));
    assert_eq!(s.cases.len(), 5);
    let up = s.case("f32/up").unwrap();
    assert_eq!(up.window, Some([0, 0, 2, 2]));
    assert_eq!(up.buf, Some([4, 4]));
    assert_eq!(up.compare.tolerance, Tolerance::Abs { tol: 1e-6 });
    assert_eq!(s.cases_with_op("raster.read").count(), 4);
    assert!(s.resolve("f32.tif").ends_with("/tests/data/mini/f32.tif"));
    assert_eq!(s.resolve("vrt://i16.tif?bands=2"), "vrt://i16.tif?bands=2");
}

#[test]
fn reads_answers() {
    let s = set();
    assert_eq!(s.oracle_ids(), ["gdal-3.8.4"]);
    let a = s.answers_default().unwrap();
    assert_eq!(a.oracle["version"], "3.8.4");

    // f32.tif is 0..12 in a 4 x 3 grid with NaN at row 1, col 2
    let mut want: Vec<f64> = (0..12).map(f64::from).collect();
    want[6] = f64::NAN;
    let got = a.values_f64("f32/all").unwrap();
    assert!(compare(&got, &want, &s.case("f32/all").unwrap().compare).passed());

    // nearest upsampling of the top-left 2 x 2 to 4 x 4
    let up = a.values_f64("f32/up").unwrap();
    assert_eq!(
        up,
        [0., 0., 1., 1., 0., 0., 1., 1., 4., 4., 5., 5., 4., 4., 5., 5.]
    );

    // band 2 of the int16 file through vrt://, window from column 1
    let (v, _) = a.values_bytes("i16/band2").unwrap();
    assert_eq!(
        (v.dtype.as_str(), v.shape.as_slice()),
        ("int16", &[1usize, 2, 2][..])
    );
    assert_eq!(a.values_f64("i16/band2").unwrap(), [-1., 1., 5., 7.]);

    let info = a.get("f32/info").unwrap().info.as_ref().unwrap();
    assert_eq!(info["size"], serde_json::json!([4, 3]));
    assert_eq!(info["crs"]["code"], "EPSG:4326");
}

#[test]
fn refusal_is_an_answer() {
    let a = set().answers("gdal-3.8.4").unwrap();
    assert_eq!(a.get("missing").unwrap().status, Status::Error);
    assert!(matches!(a.values_f64("missing"), Err(Error::Refused(..))));
    assert!(matches!(
        set().answers("gdal-0.0"),
        Err(Error::NoAnswers(_))
    ));
}
