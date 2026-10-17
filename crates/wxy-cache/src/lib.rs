//! # wxy-cache
//!
//! A persistent byte-range cache (design m05, pilot C1; keys and layout
//! per m06): every range read through it is kept, still encoded, as a row
//! `(source, offset, size, url, data, hash, validator, checked_by,
//! fetched)`. The rows are a byte reference table with the bytes in it;
//! joined to an mdim-refs table on `(path, offset, size)` they give back
//! chunk keys, so the cache grows into a store covering exactly what was
//! read, in the source's own encoding, without anyone choosing a format.
//!
//! Rows are keyed on the **source** (the path as a reference table names
//! it, before any remap to an access URL; for a bare file, its URL as
//! given), so copies of one file share rows; the URL actually fetched is
//! a plain column. A row is used only when its validator (an etag, or size
//! and modification time) matches the source's current one; `checked_by`
//! says which was used, or `none`.
//!
//! The cache is a directory of append-only Parquet fragments partitioned
//! by source (`<dir>/<source hash>/<run>.parquet`): a source's partition
//! is loaded the first time it is asked for, and nothing is ever
//! rewritten, so concurrent runs never contend. [`Cache::get`] and
//! [`Cache::put`] are the format-free interface; [`Cache::reader`] wraps
//! an async-tiff byte reader with it, keyed on the ranges the reader asks
//! for, above any range merging the store does below.

use std::collections::{HashMap, HashSet};
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

/// How a source's version was checked when its bytes were read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Validator {
    /// The server's etag.
    Etag(String),
    /// Size and last-modified time, as `size-mtime`.
    SizeMtime(String),
    /// Nothing to check by: rows are used whatever the source now holds.
    None,
}

impl Validator {
    fn value(&self) -> Option<&str> {
        match self {
            Validator::Etag(v) | Validator::SizeMtime(v) => Some(v),
            Validator::None => None,
        }
    }

    fn checked_by(&self) -> &'static str {
        match self {
            Validator::Etag(_) => "etag",
            Validator::SizeMtime(_) => "size_mtime",
            Validator::None => "none",
        }
    }

    fn from_columns(by: &str, value: Option<&str>) -> Self {
        match (by, value) {
            ("etag", Some(v)) => Validator::Etag(v.into()),
            ("size_mtime", Some(v)) => Validator::SizeMtime(v.into()),
            _ => Validator::None,
        }
    }
}

/// One held range.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The source as a reference table names it (before any remap), or a
    /// bare file's URL as given.
    pub source: String,
    pub offset: u64,
    pub size: u64,
    /// Where the bytes were actually fetched from.
    pub url: String,
    /// The bytes as fetched, never decoded.
    pub data: Bytes,
    /// sha256 of `data`, hex.
    pub hash: String,
    /// The source's version when the bytes were read.
    pub validator: Validator,
    /// Milliseconds since the Unix epoch.
    pub fetched: i64,
}

type Key = (String, u64, u64);

/// Reads that went through a [`Cache`]: hits came from it, misses from
/// the source.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub hits: u64,
    pub hit_bytes: u64,
    pub misses: u64,
    pub miss_bytes: u64,
}

/// A cache directory: the partitions loaded so far, plus the rows put
/// since the last flush.
#[derive(Debug)]
pub struct Cache {
    dir: PathBuf,
    loaded: Mutex<HashSet<String>>,
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
        Field::new("source", DataType::Utf8, false),
        Field::new("offset", DataType::UInt64, false),
        Field::new("size", DataType::UInt64, false),
        Field::new("url", DataType::Utf8, false),
        Field::new("data", DataType::Binary, false),
        Field::new("hash", DataType::Utf8, false),
        Field::new("validator", DataType::Utf8, true),
        Field::new("checked_by", DataType::Utf8, false),
        Field::new(
            "fetched",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
    ]))
}

/// The partition directory of `source`: the first 16 hex digits of its
/// sha256.
pub fn partition(dir: impl AsRef<Path>, source: &str) -> PathBuf {
    dir.as_ref().join(&sha256(source.as_bytes())[..16])
}

/// The fragments of one partition directory, in name order.
fn partition_fragments(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if dir.is_dir() {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x == "parquet") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Every fragment in a cache directory, partition by partition.
pub fn fragments(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut parts: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    parts.sort();
    for p in parts {
        out.extend(partition_fragments(&p)?);
    }
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
        let text = |name: &'static str| -> Result<&StringArray> {
            col(name)?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| down(name))
        };
        let int = |name: &'static str| -> Result<&UInt64Array> {
            col(name)?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| down(name))
        };
        let (source, url, hash) = (text("source")?, text("url")?, text("hash")?);
        let (validator, checked_by) = (text("validator")?, text("checked_by")?);
        let (offset, size) = (int("offset")?, int("size")?);
        let data = col("data")?
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| down("data"))?;
        let fetched = col("fetched")?
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| down("fetched"))?;
        for i in 0..b.num_rows() {
            rows.push(Row {
                source: source.value(i).into(),
                offset: offset.value(i),
                size: size.value(i),
                url: url.value(i).into(),
                data: Bytes::copy_from_slice(data.value(i)),
                hash: hash.value(i).into(),
                validator: Validator::from_columns(
                    checked_by.value(i),
                    (!validator.is_null(i)).then(|| validator.value(i)),
                ),
                fetched: fetched.value(i),
            });
        }
    }
    Ok(rows)
}

fn write_fragment(file: &Path, rows: &[Row]) -> Result<()> {
    let strings = |f: fn(&Row) -> &str| Arc::new(StringArray::from_iter_values(rows.iter().map(f)));
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            strings(|r| &r.source),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.offset))),
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r| r.size))),
            strings(|r| &r.url),
            Arc::new(BinaryArray::from_iter_values(
                rows.iter().map(|r| r.data.as_ref()),
            )),
            strings(|r| &r.hash),
            Arc::new(StringArray::from_iter(
                rows.iter().map(|r| r.validator.value()),
            )),
            strings(|r| r.validator.checked_by()),
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
    /// Open (creating if missing) the cache in `dir`. Partitions are read
    /// when their source is first asked for.
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Arc::new(Self {
            dir,
            loaded: Mutex::new(HashSet::new()),
            held: Mutex::new(HashMap::new()),
            new: Mutex::new(Vec::new()),
            hits: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            miss_bytes: AtomicU64::new(0),
        }))
    }

    /// The directory the partitions live in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Rows held in the partitions loaded so far.
    pub fn len(&self) -> usize {
        self.held.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Load `source`'s partition, once. A fragment that cannot be read is
    /// skipped (the source is read again), as is a row whose bytes do not
    /// match its size.
    fn load(&self, source: &str) {
        if !self.loaded.lock().unwrap().insert(source.to_string()) {
            return;
        }
        let files = partition_fragments(&partition(&self.dir, source)).unwrap_or_default();
        let mut held = self.held.lock().unwrap();
        for f in files {
            for r in read_fragment(&f).unwrap_or_default() {
                if r.source == source && r.data.len() as u64 == r.size {
                    held.insert((r.source.clone(), r.offset, r.size), r);
                }
            }
        }
    }

    /// The bytes of `range` of `source`, when held under `validator`.
    pub fn get(&self, source: &str, range: Range<u64>, validator: &Validator) -> Option<Bytes> {
        self.load(source);
        let held = self.held.lock().unwrap();
        let r = held.get(&(source.to_string(), range.start, range.end - range.start))?;
        (r.validator == *validator).then(|| r.data.clone())
    }

    /// Whether `range` of `source` is held under `validator`.
    pub fn contains(&self, source: &str, range: Range<u64>, validator: &Validator) -> bool {
        self.get(source, range, validator).is_some()
    }

    /// Keep `data`, read from `url`, as `range` of `source`.
    pub fn put(
        &self,
        source: &str,
        url: &str,
        range: Range<u64>,
        validator: &Validator,
        data: Bytes,
    ) {
        self.load(source);
        let row = Row {
            source: source.into(),
            offset: range.start,
            size: range.end - range.start,
            url: url.into(),
            hash: sha256(&data),
            data,
            validator: validator.clone(),
            fetched: now_ms(),
        };
        let key = (row.source.clone(), row.offset, row.size);
        self.new.lock().unwrap().push(row.clone());
        self.held.lock().unwrap().insert(key, row);
    }

    /// Hits and misses so far, through [`Cache::reader`].
    pub fn stats(&self) -> Stats {
        Stats {
            hits: self.hits.load(Ordering::Relaxed),
            hit_bytes: self.hit_bytes.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            miss_bytes: self.miss_bytes.load(Ordering::Relaxed),
        }
    }

    /// Write the rows put since the last flush, one new fragment in each
    /// source's partition, and return their paths.
    pub fn flush(&self) -> Result<Vec<PathBuf>> {
        let rows = std::mem::take(&mut *self.new.lock().unwrap());
        let mut by_source: HashMap<&str, Vec<Row>> = HashMap::new();
        for r in &rows {
            by_source.entry(&r.source).or_default().push(r.clone());
        }
        let run = format!("{:013}-{}.parquet", now_ms(), std::process::id());
        let mut out = Vec::new();
        for (source, rows) in by_source {
            let part = partition(&self.dir, source);
            std::fs::create_dir_all(&part)?;
            let file = part.join(&run);
            write_fragment(&file, &rows)?;
            out.push(file);
        }
        out.sort();
        Ok(out)
    }

    /// `inner`, the bytes of `source` at `url`, read through this cache.
    pub fn reader(
        self: &Arc<Self>,
        inner: Arc<dyn AsyncFileReader>,
        source: impl Into<String>,
        url: impl Into<String>,
        validator: Validator,
    ) -> Arc<dyn AsyncFileReader> {
        Arc::new(CachedReader {
            inner,
            cache: self.clone(),
            source: source.into(),
            url: url.into(),
            validator,
        })
    }
}

/// A byte reader that answers from a [`Cache`] and keeps what it fetches.
#[derive(Debug)]
struct CachedReader {
    inner: Arc<dyn AsyncFileReader>,
    cache: Arc<Cache>,
    source: String,
    url: String,
    validator: Validator,
}

impl CachedReader {
    fn hit(&self, range: &Range<u64>) -> Option<Bytes> {
        let b = self
            .cache
            .get(&self.source, range.clone(), &self.validator)?;
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
                .put(&self.source, &self.url, range, &self.validator, b.clone());
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
