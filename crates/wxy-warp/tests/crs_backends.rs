//! Each CRS engine compiled into this build against PROJ 9.8.1, on the 35
//! cases of wxylib/design `docs/watch/crs-coverage.md` (cases.json from
//! `scripts/crs_coverage/make_cases.py`, copied to tests/data).
//!
//! A case is exact when one point goes forward and PROJ's answer comes
//! back within 1 mm. Each engine must stay exact on the cases it is known
//! to pass, so a regression fails here and a newly passing case is
//! printed for the list to be widened.
//!
//! `cargo test -p wxy-warp --test crs_backends -- --nocapture` prints the
//! grid; add `--features terrio-crs` for both engines side by side.

use serde_json::Value;
use wxy_warp::crs::{crs_pair_with, CrsBackend};

/// Cases proj4rs 0.3.0 passes (design PR #39 table).
const PROJ4RS_EXACT: &[&str] = &[
    "3031",
    "3412",
    "3413",
    "3976",
    "3577",
    "32755",
    "3857",
    "6933",
    "3112",
    "laea-south",
    "laea-north",
    "aeqd-south",
    "aeqd-oblique",
    "stere-south",
    "ortho",
    "geos-himawari",
    "geos-goes",
    "sinu-modis",
    "moll",
    "eqearth",
    "robin",
    "gnom-south",
    "tpers",
    "nsper",
];

/// Cases terrio-crs passes: every one but PROJJSON with PROJ's axis
/// `meridian` member.
const TERRIO_MISSES: &[&str] = &["3031-projjson"];

fn cases() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/crs_cases.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    v["cases"].as_array().unwrap().clone()
}

/// Distance in metres from `got` to `want`: degrees scaled at `lat` for a
/// geographic (or rotated) side, else plain.
fn metres(got: (f64, f64), want: &Value, angular: bool, lat: f64) -> f64 {
    let (wx, wy) = (want[0].as_f64().unwrap(), want[1].as_f64().unwrap());
    if angular {
        let dx = (got.0 - wx) * lat.to_radians().cos();
        (dx.hypot(got.1 - wy)) * 111_320.0
    } else {
        (got.0 - wx).hypot(got.1 - wy)
    }
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Exact,
    Off(f64),
    Refused(String),
}

fn judge(backend: CrsBackend, c: &Value) -> Verdict {
    let (src, dst) = (c["src"].as_str().unwrap(), c["dst"].as_str().unwrap());
    let (lon, lat) = (c["lon"].as_f64().unwrap(), c["lat"].as_f64().unwrap());
    let (to_src, to_dst) = match crs_pair_with(backend, src, dst) {
        Ok(p) => p,
        Err(e) => return Verdict::Refused(e.to_string()),
    };
    let Some(f) = to_dst.convert(lon, lat) else {
        return Verdict::Refused("forward failed".into());
    };
    let id = c["id"].as_str().unwrap();
    let out = &c["oracle"];
    let geo_dst = dst.starts_with("EPSG:4") || dst.contains("ob_tran") || id == "4283-7844";
    let d = metres(f, out, geo_dst, out[1].as_f64().unwrap());
    // back from PROJ's answer, compared with PROJ's own way back
    let o = (out[0].as_f64().unwrap(), out[1].as_f64().unwrap());
    let di = match to_src.convert(o.0, o.1) {
        Some(b) => metres(b, &c["oracle_inv"], true, lat),
        None => return Verdict::Refused("inverse failed".into()),
    };
    if d < 1e-3 && di < 1e-3 {
        Verdict::Exact
    } else {
        Verdict::Off(d.max(di))
    }
}

#[test]
fn engines_against_proj() {
    let cases = cases();
    assert_eq!(cases.len(), 35);
    let backends = CrsBackend::available();
    println!(
        "| case | {} |",
        backends
            .iter()
            .map(|b| b.name())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for c in &cases {
        let id = c["id"].as_str().unwrap();
        let mut row = Vec::new();
        for &b in backends {
            let v = judge(b, c);
            row.push(match &v {
                Verdict::Exact => "exact".to_string(),
                Verdict::Off(m) => format!("OFF {m:.3} m"),
                Verdict::Refused(_) => "no".to_string(),
            });
            let expected = match b {
                CrsBackend::Proj4rs => PROJ4RS_EXACT.contains(&id),
                CrsBackend::TerrioCrs => !TERRIO_MISSES.contains(&id),
            };
            match (expected, v == Verdict::Exact) {
                (true, false) => regressions.push(format!("{} {id}: {v:?}", b.name())),
                (false, true) => gains.push(format!("{} {id}", b.name())),
                _ => {}
            }
        }
        println!("| {id} | {} |", row.join(" | "));
    }
    if !gains.is_empty() {
        println!("newly exact (widen the lists): {gains:?}");
    }
    assert!(regressions.is_empty(), "{regressions:#?}");
}

/// The active engine's error names the engine and what it takes.
#[test]
fn refusal_names_the_engine() {
    let e = wxy_warp::check_crs("not a crs").unwrap_err().to_string();
    assert!(e.contains(CrsBackend::active().name()), "{e}");
}

/// A pole is one point whatever its longitude: (lon, -90) in EPSG:4326 is
/// (0, 0) in EPSG:3031 for PROJ, from every engine here.
#[test]
fn pole_is_a_point() {
    for &b in CrsBackend::available() {
        let (_, to) = crs_pair_with(b, "EPSG:4326", "EPSG:3031").unwrap();
        for lon in [-120.59, 0.0, 10.26, 180.0] {
            let (x, y) = to
                .convert(lon, -90.0)
                .unwrap_or_else(|| panic!("{} refused ({lon}, -90)", b.name()));
            assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "{}: ({x}, {y})", b.name());
        }
    }
}

/// A definition the engine keeps but cannot use is refused when checked,
/// not later when the warp needs it.
#[test]
fn unusable_wkt_is_refused() {
    for &b in CrsBackend::available() {
        assert!(
            wxy_warp::crs::check_crs_with(b, "GEOGCRS[\"x\"]").is_err(),
            "{}",
            b.name()
        );
    }
}

/// Points per second through each engine, EPSG:4326 to EPSG:3031 and
/// back. Not a gate: `cargo test --release -p wxy-warp --features
/// terrio-crs --test crs_backends -- --ignored --nocapture speed`.
#[test]
#[ignore]
fn speed() {
    let n = 1_000_000;
    let pts: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            (
                -180.0 + 360.0 * (i % 1000) as f64 / 1000.0,
                -89.0 + 39.0 * (i / 1000) as f64 / 1000.0,
            )
        })
        .collect();
    for &b in CrsBackend::available() {
        let (back, to) = crs_pair_with(b, "EPSG:4326", "EPSG:3031").unwrap();
        let t = std::time::Instant::now();
        let mut ok = 0;
        for &(x, y) in &pts {
            if let Some((px, py)) = to.convert(x, y) {
                ok += back.convert(px, py).is_some() as usize;
            }
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "{}: {n} round trips in {s:.2} s ({:.2} M/s), {ok} ok",
            b.name(),
            n as f64 / s / 1e6
        );
    }
}
