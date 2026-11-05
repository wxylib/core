//! Read oracle-case fixture sets and compare values with their answers.
//!
//! A set (format `oracle-case/0`, defined in wxylib/fixtures `ORACLE.md`)
//! is a directory with a `cases.json` holding questions (open this dsn,
//! read this window) and one `answers/<oracle-id>/answers.json` per run of
//! a reference library, such as `gdal-3.8.4` or `gdal-3.14.0dev-...`. The
//! crate does no I/O beyond reading those files and knows nothing of
//! rasters: a test reads the case, asks its own implementation, and hands
//! the values to [`compare`].
//!
//! ```no_run
//! use oracle_case::{Set, compare};
//! let set = Set::open("fixtures/cog").unwrap();
//! let answers = set.answers_default().unwrap();
//! for case in set.cases_with_op("raster.read") {
//!     let want = answers.values_f64(&case.id).unwrap();
//!     let got: Vec<f64> = vec![]; // the implementation's read of case.dsn
//!     let outcome = compare(&got, &want, &case.compare);
//!     assert!(outcome.passed(), "{}: {outcome}", case.id);
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The format this crate reads.
pub const FORMAT: &str = "oracle-case/0";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path}: format {found:?}, this crate reads {FORMAT}")]
    Format { path: PathBuf, found: String },
    #[error("no answer set {0:?}")]
    NoAnswers(String),
    #[error("case {0:?} has no answer")]
    NoAnswer(String),
    #[error("case {0:?} was answered with an error: {1}")]
    Refused(String, String),
    #[error("case {0:?} has no values in its answer")]
    NoValues(String),
    #[error("{file}: {len} bytes, expected {want} for {dtype} {shape:?}")]
    Size {
        file: String,
        len: usize,
        want: usize,
        dtype: String,
        shape: Vec<usize>,
    },
    #[error("{file}: sha256 {got}, answers.json says {want}")]
    Checksum {
        file: String,
        got: String,
        want: String,
    },
    #[error("unknown type {0:?}")]
    Dtype(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.into(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.into(),
        source,
    })
}

/// What an implementation must reach for a case.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Tolerance {
    /// Equal values (NaN equals NaN).
    Exact,
    /// `|got - want| <= tol`.
    Abs { tol: f64 },
    /// `|got - want| <= tol * |want|`.
    Rel { tol: f64 },
}

/// A case's `compare`: a tolerance, a count of cells allowed to differ,
/// and why, if a difference is known.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Compare {
    #[serde(flatten)]
    pub tolerance: Tolerance,
    #[serde(default)]
    pub max_mismatch: usize,
    #[serde(default)]
    pub note: Option<String>,
}

impl Default for Compare {
    fn default() -> Self {
        Compare {
            tolerance: Tolerance::Exact,
            max_mismatch: 0,
            note: None,
        }
    }
}

/// One question. Fields an op does not use are `None`; fields this crate
/// does not know are kept in `extra`.
#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub id: String,
    pub op: String,
    pub dsn: String,
    #[serde(default)]
    pub open_options: Vec<String>,
    #[serde(default)]
    pub config: BTreeMap<String, String>,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub compare: Compare,
    /// raster.read: [xoff, yoff, xsize, ysize].
    #[serde(default)]
    pub window: Option<[u64; 4]>,
    /// raster.read: [xsize, ysize].
    #[serde(default)]
    pub buf: Option<[u64; 2]>,
    /// raster.read: 1-based band numbers.
    #[serde(default)]
    pub bands: Option<Vec<u32>>,
    #[serde(default)]
    pub resampling: Option<String>,
    /// mdim: the array's full name.
    #[serde(default)]
    pub array: Option<String>,
    #[serde(default)]
    pub start: Option<Vec<u64>>,
    #[serde(default)]
    pub count: Option<Vec<u64>>,
    #[serde(default)]
    pub step: Option<Vec<i64>>,
    /// vector.read
    #[serde(default)]
    pub layer: Option<String>,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default, rename = "where")]
    pub where_: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CasesFile {
    format: String,
    #[serde(default)]
    set: Option<String>,
    #[serde(default)]
    description: Option<String>,
    cases: Vec<Case>,
}

/// A fixture set: its directory and its questions.
#[derive(Debug, Clone)]
pub struct Set {
    pub dir: PathBuf,
    pub name: Option<String>,
    pub description: Option<String>,
    pub cases: Vec<Case>,
}

impl Set {
    /// Reads `<dir>/cases.json`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Set> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join("cases.json");
        let f: CasesFile = read_json(&path)?;
        if f.format != FORMAT {
            return Err(Error::Format {
                path,
                found: f.format,
            });
        }
        Ok(Set {
            dir,
            name: f.set,
            description: f.description,
            cases: f.cases,
        })
    }

    /// The set at `$var/<name>`, or `None` when the variable is unset or
    /// the set is absent, so a test can skip. Prints why it skipped.
    pub fn from_env(var: &str, name: &str) -> Option<Set> {
        let root = std::env::var_os(var)?;
        let dir = Path::new(&root).join(name);
        if !dir.join("cases.json").exists() {
            eprintln!("skipping: no oracle-case set at {}", dir.display());
            return None;
        }
        Some(Set::open(dir).expect("a readable oracle-case set"))
    }

    pub fn case(&self, id: &str) -> Option<&Case> {
        self.cases.iter().find(|c| c.id == id)
    }

    pub fn cases_with_op<'a>(&'a self, op: &'a str) -> impl Iterator<Item = &'a Case> + 'a {
        self.cases.iter().filter(move |c| c.op == op)
    }

    /// The oracle ids with an answer set, sorted.
    pub fn oracle_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(self.dir.join("answers"))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().join("answers.json").exists())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        ids
    }

    pub fn answers(&self, oracle_id: &str) -> Result<Answers> {
        let path = self
            .dir
            .join("answers")
            .join(oracle_id)
            .join("answers.json");
        if !path.exists() {
            return Err(Error::NoAnswers(oracle_id.into()));
        }
        let f: AnswersFile = read_json(&path)?;
        if f.format != FORMAT {
            return Err(Error::Format {
                path,
                found: f.format,
            });
        }
        Ok(Answers {
            dir: self.dir.clone(),
            id: oracle_id.into(),
            oracle: f.oracle,
            answers: f.answers,
        })
    }

    /// The answer set named by `ORACLE_ID` if set, else the newest GDAL
    /// version present (by version number; a dev build comes before its release).
    pub fn answers_default(&self) -> Result<Answers> {
        if let Ok(id) = std::env::var("ORACLE_ID") {
            return self.answers(&id);
        }
        let ids = self.oracle_ids();
        let best = ids
            .iter()
            .max_by(|a, b| version_key(a).cmp(&version_key(b)))
            .ok_or_else(|| Error::NoAnswers("any".into()))?;
        self.answers(best)
    }

    /// A case's dsn as GDAL received it: `{set}` replaced by the set
    /// directory, and a plain relative path joined to it. Other forms
    /// (`vrt://`, driver prefixes) are relative to the set directory as a
    /// working directory; resolving them is the implementation's job.
    pub fn resolve(&self, dsn: &str) -> String {
        let d = self.dir.to_string_lossy();
        let s = dsn.replace("{set}", &d);
        let plain =
            !s.starts_with('/') && !s.starts_with('<') && !s.contains("://") && !s.contains(':');
        if plain {
            self.dir.join(&s).to_string_lossy().into_owned()
        } else {
            s
        }
    }
}

fn version_key(id: &str) -> (Vec<u64>, bool) {
    let v = id
        .rsplit_once('-')
        .map(|(a, _)| a)
        .filter(|_| id.contains("dev"))
        .unwrap_or(id);
    let v = v.trim_start_matches(|c: char| !c.is_ascii_digit());
    let nums = v
        .split(|c: char| !c.is_ascii_digit())
        .take(3)
        .filter_map(|p| p.parse().ok())
        .collect();
    (nums, !id.contains("dev"))
}

#[derive(Debug, Deserialize)]
struct AnswersFile {
    format: String,
    #[serde(default)]
    oracle: serde_json::Value,
    answers: BTreeMap<String, Answer>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    Error,
}

/// A raw array file and how to read it.
#[derive(Debug, Clone, Deserialize)]
pub struct Values {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileRef {
    pub file: String,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub level: String,
    pub code: i64,
    pub msg: String,
}

/// One answer: values, features or info, or the error the oracle gave.
#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    pub status: Status,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub values: Option<Values>,
    #[serde(default)]
    pub features: Option<FileRef>,
    #[serde(default)]
    pub info: Option<serde_json::Value>,
    #[serde(default)]
    pub messages: Vec<Message>,
}

/// One oracle run's answers to a set.
#[derive(Debug, Clone)]
pub struct Answers {
    dir: PathBuf,
    pub id: String,
    /// The `oracle` block: name, version, how it was reached.
    pub oracle: serde_json::Value,
    pub answers: BTreeMap<String, Answer>,
}

impl Answers {
    pub fn get(&self, id: &str) -> Result<&Answer> {
        self.answers
            .get(id)
            .ok_or_else(|| Error::NoAnswer(id.into()))
    }

    fn ok(&self, id: &str) -> Result<&Answer> {
        let a = self.get(id)?;
        if a.status == Status::Error {
            return Err(Error::Refused(
                id.into(),
                a.error.clone().unwrap_or_default(),
            ));
        }
        Ok(a)
    }

    /// The answer's array file, checked against its size and checksum.
    pub fn values_bytes(&self, id: &str) -> Result<(Values, Vec<u8>)> {
        let v = self
            .ok(id)?
            .values
            .clone()
            .ok_or_else(|| Error::NoValues(id.into()))?;
        let path = self.dir.join(&v.file);
        let bytes = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
        let want = v.shape.iter().product::<usize>() * dtype_size(&v.dtype)?;
        if bytes.len() != want {
            return Err(Error::Size {
                file: v.file,
                len: bytes.len(),
                want,
                dtype: v.dtype,
                shape: v.shape,
            });
        }
        if let Some(sum) = &v.sha256 {
            let got = hex(&Sha256::digest(&bytes));
            if &got != sum {
                return Err(Error::Checksum {
                    file: v.file,
                    got,
                    want: sum.clone(),
                });
            }
        }
        Ok((v, bytes))
    }

    /// The answer's array as f64, in C order (shape in `values.shape`).
    pub fn values_f64(&self, id: &str) -> Result<Vec<f64>> {
        let (v, bytes) = self.values_bytes(id)?;
        decode_f64(&bytes, &v.dtype)
    }

    /// The answer's feature file (vector.read), as JSON.
    pub fn features(&self, id: &str) -> Result<serde_json::Value> {
        let f = self
            .ok(id)?
            .features
            .clone()
            .ok_or_else(|| Error::NoValues(id.into()))?;
        read_json(&self.dir.join(&f.file))
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Bytes per element of a numpy type name.
pub fn dtype_size(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "uint8" | "int8" => 1,
        "uint16" | "int16" => 2,
        "uint32" | "int32" | "float32" => 4,
        "uint64" | "int64" | "float64" => 8,
        _ => return Err(Error::Dtype(dtype.into())),
    })
}

/// Little-endian values of a numpy type name, as f64 (64-bit integers
/// beyond 2^53 lose precision; compare those as bytes).
pub fn decode_f64(bytes: &[u8], dtype: &str) -> Result<Vec<f64>> {
    let n = dtype_size(dtype)?;
    macro_rules! dec {
        ($t:ty) => {
            bytes
                .chunks_exact(n)
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect()
        };
    }
    Ok(match dtype {
        "uint8" => bytes.iter().map(|&b| b as f64).collect(),
        "int8" => bytes.iter().map(|&b| b as i8 as f64).collect(),
        "uint16" => dec!(u16),
        "int16" => dec!(i16),
        "uint32" => dec!(u32),
        "int32" => dec!(i32),
        "uint64" => dec!(u64),
        "int64" => dec!(i64),
        "float32" => dec!(f32),
        "float64" => dec!(f64),
        _ => return Err(Error::Dtype(dtype.into())),
    })
}

/// How an implementation's values met a case's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub len: usize,
    pub want_len: usize,
    pub mismatches: usize,
    pub max_abs: f64,
    /// Index (C order) of the first cell outside tolerance.
    pub first: Option<usize>,
    pub max_mismatch: usize,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.len == self.want_len && self.mismatches <= self.max_mismatch
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len != self.want_len {
            return write!(f, "{} values, the answer has {}", self.len, self.want_len);
        }
        write!(
            f,
            "{} of {} outside tolerance (allowed {}), max |diff| {}",
            self.mismatches, self.len, self.max_mismatch, self.max_abs
        )?;
        if let Some(i) = self.first {
            write!(f, ", first at {i}")?;
        }
        Ok(())
    }
}

/// Compares values cell by cell under a case's `compare`. NaN equals NaN;
/// NaN against a number is a mismatch.
pub fn compare(got: &[f64], want: &[f64], c: &Compare) -> Outcome {
    let mut o = Outcome {
        len: got.len(),
        want_len: want.len(),
        mismatches: 0,
        max_abs: 0.0,
        first: None,
        max_mismatch: c.max_mismatch,
    };
    if got.len() != want.len() {
        return o;
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ok = match (g.is_nan(), w.is_nan()) {
            (true, true) => true,
            (false, false) => {
                let d = (g - w).abs();
                if d > o.max_abs {
                    o.max_abs = d;
                }
                match c.tolerance {
                    Tolerance::Exact => g == w,
                    Tolerance::Abs { tol } => d <= tol,
                    Tolerance::Rel { tol } => d <= tol * w.abs(),
                }
            }
            _ => {
                o.max_abs = f64::NAN;
                false
            }
        };
        if !ok {
            o.mismatches += 1;
            o.first.get_or_insert(i);
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_order() {
        let mut ids = vec![
            "gdal-3.14.0dev-739fcf03bf",
            "gdal-3.8.4",
            "gdal-3.12.2",
            "gdal-3.14.0",
        ];
        ids.sort_by_key(|a| version_key(a));
        assert_eq!(
            ids,
            [
                "gdal-3.8.4",
                "gdal-3.12.2",
                "gdal-3.14.0dev-739fcf03bf",
                "gdal-3.14.0"
            ]
        );
    }

    #[test]
    fn tolerances() {
        let exact = Compare::default();
        assert!(compare(&[1.0, f64::NAN], &[1.0, f64::NAN], &exact).passed());
        let o = compare(&[1.0, 2.5], &[1.0, 2.0], &exact);
        assert_eq!((o.mismatches, o.first, o.max_abs), (1, Some(1), 0.5));
        let abs = Compare {
            tolerance: Tolerance::Abs { tol: 0.5 },
            ..exact.clone()
        };
        assert!(compare(&[1.0, 2.5], &[1.0, 2.0], &abs).passed());
        assert!(!compare(&[0.0], &[f64::NAN], &abs).passed());
        assert!(!compare(&[0.0], &[0.0, 1.0], &abs).passed());
        let one = Compare {
            max_mismatch: 1,
            ..exact
        };
        assert!(compare(&[1.0, 2.5], &[1.0, 2.0], &one).passed());
    }
}
