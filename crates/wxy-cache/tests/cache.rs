//! The cache answers what it holds, keeps what it fetches, persists it as
//! fragments, and ignores rows read under another etag.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_tiff::error::AsyncTiffResult;
use async_tiff::reader::AsyncFileReader;
use bytes::Bytes;
use wxy_cache::{fragments, read_fragment, Cache, Validator};

/// A file of bytes `0, 1, 2, ...` that counts the ranges asked of it.
#[derive(Debug, Default)]
struct Counting(AtomicU64);

#[async_trait::async_trait]
impl AsyncFileReader for Counting {
    async fn get_bytes(&self, range: Range<u64>) -> AsyncTiffResult<Bytes> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(range.map(|i| i as u8).collect())
    }
}

fn reads(c: &Counting) -> u64 {
    c.0.load(Ordering::Relaxed)
}

#[tokio::test]
async fn second_run_reads_nothing_from_the_source() {
    let dir = std::env::temp_dir().join(format!("wxy-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ranges = vec![0..10, 100..164, 4000..4096];

    // first run: everything from the source, kept
    let inner = Arc::new(Counting::default());
    let cache = Cache::open(&dir).unwrap();
    let v1 = Validator::SizeMtime("v1".into());
    let r = cache.reader(inner.clone(), "s3://a.tif", "file:///a.tif", v1.clone());
    let first = r.get_byte_ranges(ranges.clone()).await.unwrap();
    assert_eq!(reads(&inner), 3);
    assert_eq!(cache.stats().misses, 3);
    let files = cache.flush().unwrap();
    assert_eq!(files.len(), 1);
    assert!(cache.flush().unwrap().is_empty(), "nothing new to write");
    let rows = read_fragment(&files[0]).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1].offset, 100);
    assert_eq!(rows[1].size, 64);
    assert_eq!(rows[1].hash.len(), 64);
    assert_eq!(rows[1].source, "s3://a.tif");
    assert_eq!(rows[1].url, "file:///a.tif");
    assert_eq!(rows[1].validator, v1);

    // second run, a new process in effect, fetching from another copy of
    // the same source: all from the fragments
    let inner = Arc::new(Counting::default());
    let cache = Cache::open(&dir).unwrap();
    assert_eq!(cache.len(), 0, "partitions load on first use");
    let r = cache.reader(
        inner.clone(),
        "s3://a.tif",
        "https://mirror/a.tif",
        v1.clone(),
    );
    let second = r.get_byte_ranges(ranges.clone()).await.unwrap();
    assert_eq!(second, first);
    assert_eq!(reads(&inner), 0);
    assert_eq!(cache.stats().hits, 3);
    assert_eq!(cache.stats().hit_bytes, 10 + 64 + 96);

    // a mix: only the miss goes to the source
    let mixed = r.get_byte_ranges(vec![0..10, 10..20]).await.unwrap();
    assert_eq!(mixed[1], (10..20).map(|i| i as u8).collect::<Bytes>());
    assert_eq!(reads(&inner), 1);
    assert!(cache.contains("s3://a.tif", 10..20, &v1));
    assert_eq!(cache.len(), 4);

    // another version of the source, or another source, misses
    let inner = Arc::new(Counting::default());
    let r = cache.reader(
        inner.clone(),
        "s3://a.tif",
        "file:///a.tif",
        Validator::Etag("x".into()),
    );
    r.get_bytes(0..10).await.unwrap();
    let r = cache.reader(inner.clone(), "s3://b.tif", "file:///b.tif", v1.clone());
    r.get_bytes(0..10).await.unwrap();
    assert_eq!(reads(&inner), 2);

    // append only, partitioned by source: this flush adds a fragment to
    // a.tif's partition and starts b.tif's
    assert_eq!(cache.flush().unwrap().len(), 2);
    assert_eq!(fragments(&dir).unwrap().len(), 3);
    let parts = std::fs::read_dir(&dir).unwrap().count();
    assert_eq!(parts, 2);

    // no validator: cached and used, marked as unchecked
    let cache = Cache::open(&dir).unwrap();
    let inner = Arc::new(Counting::default());
    let r = cache.reader(
        inner.clone(),
        "s3://c.tif",
        "file:///c.tif",
        Validator::None,
    );
    r.get_bytes(0..8).await.unwrap();
    r.get_bytes(0..8).await.unwrap();
    assert_eq!(reads(&inner), 1);
    let rows = read_fragment(&cache.flush().unwrap()[0]).unwrap();
    assert_eq!(rows[0].validator, Validator::None);
    std::fs::remove_dir_all(&dir).unwrap();
}
