//! # wxy-cache
//!
//! A persistent byte-range cache (design m05, pilot C1): a reader wrapper
//! that keeps every range it fetches, still encoded, as a row
//! `(path, offset, size, data, hash, etag, fetched)`. The rows are a byte
//! reference table with the bytes in it; joined to an mdim-refs table on
//! `(path, offset, size)` they give back chunk keys, so the cache grows
//! into a store covering exactly what was read, in the source's own
//! encoding, without anyone choosing a format.
//!
//! The cache is a directory of append-only Parquet fragments, one per
//! [`Cache::flush`]; opening it reads them all, and nothing is ever
//! rewritten, so concurrent runs never contend. Rows are keyed on the
//! ranges a reader asks for, above any range merging the store does
//! below, so each TIFF tile is one row. A row is used only when its etag
//! matches the source's current one; a changed source simply misses.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{
    Array, BinaryArray, RecordBatch, StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_tiff::error::AsyncTiffResult;
use async_tiff::reader::AsyncFileReader;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use sha2::{Digest, Sha256};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parquet(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "cache: {e}"),
            Error::Parquet(e) => write!(f, "cache fragment: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

fn pq(e: impl std::fmt::Display) -> Error {
    Error::Parquet(e.to_string())
}

/// One held range.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    /// The bytes as fetched, never decoded.
    pub data: Bytes,
    /// sha256 of `data`, hex.
    pub hash: String,
    /// The source's etag when the bytes were read.
    pub etag: Option<String>,
    /// Milliseconds since the Unix epoch.
    pub fetched: i64,
}

type Key = (String, u64, u64);

/// Reads that went through a [`Cache`]: hits came from it, misses from
/// the inner reader.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub hits: u64,
    pub hit_bytes: u64,
    pub misses: u64,
    pub miss_bytes: u64,
}

/// A directory of cache fragments, loaded, plus the rows fetched since.
#[derive(Debug)]
pub struct Cache {
    dir: PathBuf,
    held: Mutex<HashMap<Key, Row>>,
    new: Mutex<Vec<Row>>,
    hits: AtomicU64,
    hit_bytes: AtomicU64,
    misses: AtomicU64,
    miss_bytes: AtomicU64,
}

/// The fragment schema.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("offset", DataType::UInt64, false),
        Field::new("size", DataType::UInt64, false),
        Field::new("data", DataType::Binary, false),
        Field::new("hash", DataType::Utf8, false),
        Field::new("etag", DataType::Utf8, true),
        Field::new(
            "fetched",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
    ]))
}

/// The fragments in `dir`, in name order.
pub fn fragments(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "parquet") {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

/// Every row of one fragment.
pub fn read_fragment(file: impl AsRef<Path>) -> Result<Vec<Row>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(file)?)
        .map_err(pq)?
        .build()
        .map_err(pq)?;
    let mut rows = Vec::new();
    for batch in reader {
        let b = batch.map_err(pq)?;
        let col = |name: &str| {
            b.column_by_name(name)
                .ok_or_else(|| Error::Parquet(format!("no column {name}")))
        };
        let down = |name: &'static str| Error::Parquet(format!("column {name} has the wrong type"));
        let path = col("path")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| down("path"))?;
        let offset = col("offset")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| down("offset"))?;
        let size = col("size")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| down("size"))?;
        let data = col("data")?
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| down("data"))?;
        let hash = col("hash")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| down("hash"))?;
        let etag = col("etag")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| down("etag"))?;
        let fetched = col("fetched")?
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| down("fetched"))?;
        for i in 0..b.num_rows() {
            rows.push(Row {
                path: path.value(i).into(),
                offset: offset.value(i),
                size: size.value(i),
                data: Bytes::copy_from_slice(data.value(i)),
                hash: hash.value(i).into(),
                etag: (!etag.is_null(i)).then(|| etag.value(i).into()),
                fetched: fetched.value(i),
            });
        }
    }
    Ok(rows)
}

fn write_fragment(file: &Path, rows: &[Row]) -> Result<()> {
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.path.as_str()),
            )),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.offset))),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.size))),
            Arc::new(BinaryArray::from_iter_values(
                rows.iter().map(|r| r.data.as_ref()),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.hash.as_str()),
            )),
            Arc::new(StringArray::from_iter(
                rows.iter().map(|r| r.etag.as_deref()),
            )),
            Arc::new(
                TimestampMillisecondArray::from_iter_values(rows.iter().map(|r| r.fetched))
                    .with_timezone("UTC"),
            ),
        ],
    )
    .map_err(pq)?;
    // written beside the fragments, then renamed in: a reader never sees
    // half a fragment
    let tmp = file.with_extension("parquet.tmp");
    let mut w = ArrowWriter::try_new(File::create(&tmp)?, batch.schema(), None).map_err(pq)?;
    w.write(&batch).map_err(pq)?;
    w.close().map_err(pq)?;
    std::fs::rename(&tmp, file)?;
    Ok(())
}

fn sha256(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

impl Cache {
    /// Open (creating if missing) the cache in `dir`, loading every
    /// fragment. A row whose bytes do not match its size is skipped.
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let mut held = HashMap::new();
        for f in fragments(&dir)? {
            for r in read_fragment(&f)? {
                if r.data.len() as u64 == r.size {
                    held.insert((r.path.clone(), r.offset, r.size), r);
                }
            }
        }
        Ok(Arc::new(Self {
            dir,
            held: Mutex::new(held),
            new: Mutex::new(Vec::new()),
            hits: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            miss_bytes: AtomicU64::new(0),
        }))
    }

    /// The directory the fragments live in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Rows held, from fragments and this run.
    pub fn len(&self) -> usize {
        self.held.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `range` of `path` is held under `etag`.
    pub fn contains(&self, path: &str, range: Range<u64>, etag: Option<&str>) -> bool {
        self.get(path, range, etag).is_some()
    }

    fn get(&self, path: &str, range: Range<u64>, etag: Option<&str>) -> Option<Bytes> {
        let held = self.held.lock().unwrap();
        let r = held.get(&(path.to_string(), range.start, range.end - range.start))?;
        (r.etag.as_deref() == etag).then(|| r.data.clone())
    }

    fn put(&self, path: &str, range: Range<u64>, etag: Option<&str>, data: Bytes) {
        let row = Row {
            path: path.into(),
            offset: range.start,
            size: range.end - range.start,
            hash: sha256(&data),
            data,
            etag: etag.map(String::from),
            fetched: now_ms(),
        };
        let key = (row.path.clone(), row.offset, row.size);
        self.new.lock().unwrap().push(row.clone());
        self.held.lock().unwrap().insert(key, row);
    }

    /// Hits and misses so far.
    pub fn stats(&self) -> Stats {
        Stats {
            hits: self.hits.load(Ordering::Relaxed),
            hit_bytes: self.hit_bytes.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            miss_bytes: self.miss_bytes.load(Ordering::Relaxed),
        }
    }

    /// Write the rows fetched since the last flush as one new fragment,
    /// and return its path (`None` when nothing was fetched).
    pub fn flush(&self) -> Result<Option<PathBuf>> {
        let rows = std::mem::take(&mut *self.new.lock().unwrap());
        if rows.is_empty() {
            return Ok(None);
        }
        let file = self
            .dir
            .join(format!("{:013}-{}.parquet", now_ms(), std::process::id()));
        write_fragment(&file, &rows)?;
        Ok(Some(file))
    }

    /// `inner`, reading `path` (the name rows are kept under) through this
    /// cache. `etag` identifies the source's current version.
    pub fn reader(
        self: &Arc<Self>,
        inner: Arc<dyn AsyncFileReader>,
        path: impl Into<String>,
        etag: Option<String>,
    ) -> Arc<dyn AsyncFileReader> {
        Arc::new(CachedReader {
            inner,
            cache: self.clone(),
            path: path.into(),
            etag,
        })
    }
}

/// A reader that answers from a [`Cache`] and keeps what it fetches.
#[derive(Debug)]
struct CachedReader {
    inner: Arc<dyn AsyncFileReader>,
    cache: Arc<Cache>,
    path: String,
    etag: Option<String>,
}

impl CachedReader {
    fn hit(&self, range: &Range<u64>) -> Option<Bytes> {
        let b = self
            .cache
            .get(&self.path, range.clone(), self.etag.as_deref())?;
        self.cache.hits.fetch_add(1, Ordering::Relaxed);
        self.cache
            .hit_bytes
            .fetch_add(b.len() as u64, Ordering::Relaxed);
        Some(b)
    }

    fn keep(&self, range: Range<u64>, b: &Bytes) {
        self.cache.misses.fetch_add(1, Ordering::Relaxed);
        self.cache
            .miss_bytes
            .fetch_add(b.len() as u64, Ordering::Relaxed);
        // a short read (past the end of the file) is not a held range
        if b.len() as u64 == range.end - range.start {
            self.cache
                .put(&self.path, range, self.etag.as_deref(), b.clone());
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncFileReader for CachedReader {
    async fn get_bytes(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        if let Some(b) = self.hit(&range) {
            return Ok(b);
        }
        let b = self.inner.get_bytes(range.clone()).await?;
        self.keep(range, &b);
        Ok(b)
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> AsyncTiffResult<Vec<Bytes>> {
        let mut out: Vec<Option<Bytes>> = ranges.iter().map(|r| self.hit(r)).collect();
        let missing: Vec<usize> = (0..ranges.len()).filter(|&i| out[i].is_none()).collect();
        if !missing.is_empty() {
            // the misses go down together, for the store to merge
            let got = self
                .inner
                .get_byte_ranges(missing.iter().map(|&i| ranges[i].clone()).collect())
                .await?;
            for (&i, b) in missing.iter().zip(got) {
                self.keep(ranges[i].clone(), &b);
                out[i] = Some(b);
            }
        }
        Ok(out.into_iter().map(Option::unwrap).collect())
    }
}
