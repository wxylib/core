//! The VRT corpus (wxylib/fixtures, vrt/): every case is planned by this
//! crate, evaluated by the small reference reader below (raw source pixels,
//! nearest sampling as the plan says, the plan's operations), and compared
//! byte for byte with what GDAL read for the same request.
//!
//! The reader lives here, not in the crate: the crate plans, readers read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use vrt_model::plan::nearest_indices;
use vrt_model::vrtcon::{lower, SourceInfo};
use vrt_model::{plan_band_leaves, Dataset, Dsn, LeafRow, Op, Request, Resolver};

fn root() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("vrt")
}

/// Raw source pixels, as f64, band-major.
struct Raster {
    size: (usize, usize),
    bands: Vec<Vec<f64>>,
}

struct Corpus {
    dir: PathBuf,
    sources: HashMap<String, (Raster, Value)>,
}

fn load_raw(path: &Path, dtype: &str, n: usize, bands: usize) -> Vec<Vec<f64>> {
    let b = std::fs::read(path).unwrap();
    let v: Vec<f64> = match dtype {
        "uint8" | "Byte" => b.iter().map(|&x| x as f64).collect(),
        "int16" | "Int16" => b
            .chunks(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        "float32" | "Float32" => b
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        t => panic!("dtype {t}"),
    };
    assert_eq!(v.len(), n * bands);
    v.chunks(n).map(|c| c.to_vec()).collect()
}

impl Corpus {
    fn load() -> Option<(Corpus, Value)> {
        let dir = root();
        let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
        let m: Value = serde_json::from_str(&text).unwrap();
        let mut sources = HashMap::new();
        for (_, s) in m["sources"].as_object().unwrap() {
            let size = (
                s["size"][0].as_u64().unwrap() as usize,
                s["size"][1].as_u64().unwrap() as usize,
            );
            let bands = s["bands"].as_u64().unwrap() as usize;
            let raw = load_raw(
                &dir.join(s["raw"].as_str().unwrap()),
                s["dtype"].as_str().unwrap(),
                size.0 * size.1,
                bands,
            );
            sources.insert(
                s["file"].as_str().unwrap().to_string(),
                (Raster { size, bands: raw }, s.clone()),
            );
        }
        Some((Corpus { dir, sources }, m))
    }

    /// Corpus-relative key of a resolved source name.
    fn key(&self, dsn: &Dsn) -> String {
        let s = dsn.to_string();
        let d = self.dir.to_string_lossy().to_string();
        s.strip_prefix(&format!("{d}/")).unwrap_or(&s).to_string()
    }
}

impl Resolver for Corpus {
    fn size(&self, dsn: &Dsn) -> Option<(usize, usize)> {
        if dsn.is_virtual() {
            return self.open_vrt(dsn).map(|d| (d.x_size, d.y_size));
        }
        self.sources.get(&self.key(dsn)).map(|(r, _)| r.size)
    }
    fn open_vrt(&self, dsn: &Dsn) -> Option<Dataset> {
        let text = std::fs::read_to_string(dsn.to_string()).ok()?;
        Dataset::from_xml(&text).ok()
    }
}

fn apply(v: f64, ops: &[Op]) -> Option<f64> {
    let mut x = v;
    for op in ops {
        match op {
            Op::SkipNodata(n) => {
                if x == *n || (x.is_nan() && n.is_nan()) {
                    return None;
                }
            }
            Op::Linear { ratio, offset } => x = x * ratio + offset,
            Op::Lut(t) => {
                x = if x <= t[0].0 {
                    t[0].1
                } else if x >= t[t.len() - 1].0 {
                    t[t.len() - 1].1
                } else {
                    let i = t.iter().position(|(a, _)| *a >= x).unwrap();
                    let ((x0, y0), (x1, y1)) = (t[i - 1], t[i]);
                    if x == x1 {
                        y1
                    } else {
                        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    }
                };
            }
        }
    }
    Some(x)
}

/// Evaluate a plan into a buffer of `buf` pixels, initialised to `init`.
fn evaluate(c: &Corpus, rows: &[LeafRow], buf: (usize, usize), init: f64) -> Vec<f64> {
    let mut out = vec![init; buf.0 * buf.1];
    for row in rows {
        let (r, _) = c
            .sources
            .get(&c.key(&row.dsn))
            .unwrap_or_else(|| panic!("source {}", row.dsn));
        let (xs, ys) = nearest_indices(&row.window, r.size);
        let data = &r.bands[row.band.0 - 1];
        for (j, &sy) in ys.iter().enumerate() {
            for (i, &sx) in xs.iter().enumerate() {
                let mut v = Some(data[sy * r.size.0 + sx]);
                for level in row.ops.iter().rev() {
                    v = v.and_then(|x| apply(x, level));
                }
                if let Some(v) = v {
                    out[(row.window.out[1] + j) * buf.0 + row.window.out[0] + i] = v;
                }
            }
        }
    }
    out
}

fn to_bytes(v: &[f64], dtype: &str) -> Vec<u8> {
    match dtype {
        "Byte" => v
            .iter()
            .map(|x| x.round().clamp(0.0, 255.0) as u8)
            .collect(),
        "Int16" => v
            .iter()
            .flat_map(|x| (x.round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())
            .collect(),
        "Float32" => v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
        t => panic!("dtype {t}"),
    }
}

fn info_of(src: &Value) -> SourceInfo {
    let dt = match src["dtype"].as_str().unwrap() {
        "uint8" => "Byte",
        "int16" => "Int16",
        "float32" => "Float32",
        t => panic!("{t}"),
    };
    let n = src["bands"].as_u64().unwrap() as usize;
    let g: Vec<f64> = src["geotransform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    SourceInfo {
        x_size: src["size"][0].as_u64().unwrap() as usize,
        y_size: src["size"][1].as_u64().unwrap() as usize,
        band_types: vec![dt.to_string(); n],
        geotransform: Some([g[0], g[1], g[2], g[3], g[4], g[5]]),
        srs: Some("EPSG:4326".into()),
        nodata: vec![None; n],
    }
}

#[test]
fn corpus_matches_gdal() {
    let Some((c, m)) = Corpus::load() else {
        panic!(
            "VRT corpus not found at {} (set WXY_FIXTURES)",
            root().display()
        );
    };
    let base = c.dir.to_string_lossy().to_string();
    let mut checked = 0;
    let mut report = vec![];
    for case in m["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let dsn = Dsn::parse(case["dsn"].as_str().unwrap());
        let ds = match &dsn {
            Dsn::VrtString { source, options } => {
                let key = source.to_string();
                let (_, src) = &c.sources[&key];
                let abs = Dsn::parse(&format!("{base}/{key}"));
                lower(&abs, options, &info_of(src)).unwrap()
            }
            _ => Dataset::from_xml(&std::fs::read_to_string(c.dir.join(name)).unwrap()).unwrap(),
        };
        // The model writes back what it read.
        assert_eq!(
            Dataset::from_xml(&ds.to_xml()).unwrap(),
            ds,
            "{name}: XML round trip"
        );
        let dtype = case["dtype"].as_str().unwrap();
        for rq in case["requests"].as_array().unwrap() {
            let w: Vec<usize> = rq["window"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let b: Vec<usize> = rq["buf"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let r = Request::resampled(w[0], w[1], w[2], w[3], b[0], b[1]);
            let mut got = vec![];
            for band in 1..=ds.bands.len() {
                let rows = plan_band_leaves(&ds, band, &r, &base, &c).unwrap();
                let init = ds.bands[band - 1].nodata_value().unwrap_or(0.0);
                got.extend(to_bytes(&evaluate(&c, &rows, (b[0], b[1]), init), dtype));
            }
            let want = std::fs::read(c.dir.join(rq["file"].as_str().unwrap())).unwrap();
            if got != want {
                report.push(format!("{name} {:?} -> {:?}", w, b));
            }
            checked += 1;
        }
    }
    assert!(
        report.is_empty(),
        "{} of {checked} requests differ from GDAL:\n{}",
        report.len(),
        report.join("\n")
    );
    assert_eq!(checked, 32);
}
