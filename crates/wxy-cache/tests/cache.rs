//! The cache answers what it holds, keeps what it fetches, persists it as
//! fragments, and ignores rows read under another etag.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_tiff::error::AsyncTiffResult;
use async_tiff::reader::AsyncFileReader;
use bytes::Bytes;
use wxy_cache::{fragments, read_fragment, Cache};

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
    let r = cache.reader(inner.clone(), "file:///a.tif", Some("v1".into()));
    let first = r.get_byte_ranges(ranges.clone()).await.unwrap();
    assert_eq!(reads(&inner), 3);
    assert_eq!(cache.stats().misses, 3);
    let file = cache.flush().unwrap().unwrap();
    assert_eq!(cache.flush().unwrap(), None, "nothing new to write");
    let rows = read_fragment(&file).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1].offset, 100);
    assert_eq!(rows[1].size, 64);
    assert_eq!(rows[1].hash.len(), 64);

    // second run, a new process in effect: all from the fragments
    let inner = Arc::new(Counting::default());
    let cache = Cache::open(&dir).unwrap();
    assert_eq!(cache.len(), 3);
    let r = cache.reader(inner.clone(), "file:///a.tif", Some("v1".into()));
    let second = r.get_byte_ranges(ranges.clone()).await.unwrap();
    assert_eq!(second, first);
    assert_eq!(reads(&inner), 0);
    assert_eq!(cache.stats().hits, 3);
    assert_eq!(cache.stats().hit_bytes, 10 + 64 + 96);

    // a mix: only the miss goes to the source
    let mixed = r.get_byte_ranges(vec![0..10, 10..20]).await.unwrap();
    assert_eq!(mixed[1], (10..20).map(|i| i as u8).collect::<Bytes>());
    assert_eq!(reads(&inner), 1);
    assert!(cache.contains("file:///a.tif", 10..20, Some("v1")));

    // another version of the source, or another file, misses
    let inner = Arc::new(Counting::default());
    let r = cache.reader(inner.clone(), "file:///a.tif", Some("v2".into()));
    r.get_bytes(0..10).await.unwrap();
    let r = cache.reader(inner.clone(), "file:///b.tif", Some("v1".into()));
    r.get_bytes(0..10).await.unwrap();
    assert_eq!(reads(&inner), 2);

    // append only: each flush is a new fragment
    cache.flush().unwrap();
    assert_eq!(fragments(&dir).unwrap().len(), 2);
    std::fs::remove_dir_all(&dir).unwrap();
}
