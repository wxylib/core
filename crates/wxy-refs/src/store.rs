//! Virtual arrays as zarrs storage, read straight from their references.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::StreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use wxy_cache::{Cache, Validator};
use wxy_source::{Error, Result};
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{
    AsyncMaybeBytesIterator, AsyncReadableStorageTraits, MaybeBytes, StorageError, StoreKey,
};

use crate::VirtualArray;

/// What a store key holds.
enum Entry {
    /// Encoded bytes held in the table (metadata, inline chunks).
    Held(Bytes),
    /// A byte range of a source, read on demand.
    Ref {
        source: String,
        url: String,
        offset: u64,
        length: u64,
    },
}

/// Reads through the store so far: ranges the cache answered, and ranges
/// read from their sources.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefsStats {
    pub hits: u64,
    pub hit_bytes: u64,
    pub misses: u64,
    pub miss_bytes: u64,
}

/// Zarr storage over [`VirtualArray`]s: `zarr.json` for the root group and
/// each array from its metadata, chunks by their keys (`<array>/c/0/1/...`,
/// the default encoding the table's metadata names). A referenced chunk is
/// fetched as one byte range from its URL (`file://`, `http(s)://`, or
/// `s3://bucket/key` with credentials, endpoint and addressing style from
/// the `AWS_*` environment, see [`crate::location`]);
/// a key the table does not hold is missing, so zarrs reads the fill value.
///
/// With a cache, each range is looked up first under the source's
/// validator (etag, else size and modification time, from the file or a
/// HEAD request, once per source) and kept after a fetch.
pub struct RefsStore {
    entries: HashMap<String, Entry>,
    cache: Option<Arc<Cache>>,
    stores: Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
    validators: tokio::sync::Mutex<HashMap<String, Validator>>,
    hits: AtomicU64,
    hit_bytes: AtomicU64,
    misses: AtomicU64,
    miss_bytes: AtomicU64,
}

/// The store key of chunk `key` of the array at `path`.
pub fn chunk_key(path: &str, key: &[u64]) -> String {
    let mut k = format!("{}/c", path.trim_start_matches('/'));
    for i in key {
        k.push('/');
        k.push_str(&i.to_string());
    }
    k
}

impl RefsStore {
    /// A store over `arrays`, reading through `cache` when given.
    pub fn new(arrays: &[VirtualArray], cache: Option<Arc<Cache>>) -> Result<Self> {
        let group = Bytes::from_static(br#"{"zarr_format":3,"node_type":"group"}"#);
        let mut entries = HashMap::new();
        entries.insert("zarr.json".to_string(), Entry::Held(group.clone()));
        for a in arrays {
            let path = a.path.trim_start_matches('/');
            // groups above a nested array
            let parts: Vec<&str> = path.split('/').collect();
            for i in 1..parts.len() {
                entries
                    .entry(format!("{}/zarr.json", parts[..i].join("/")))
                    .or_insert_with(|| Entry::Held(group.clone()));
            }
            let meta = serde_json::to_vec(&a.metadata)
                .map_err(|e| Error::Decode(format!("{}: {e}", a.path)))?;
            entries.insert(format!("{path}/zarr.json"), Entry::Held(meta.into()));
            for (k, bytes) in &a.inline {
                entries.insert(chunk_key(path, k), Entry::Held(bytes.clone().into()));
            }
            for r in &a.refs {
                entries.insert(
                    chunk_key(path, &r.key),
                    Entry::Ref {
                        source: r.source.clone(),
                        url: r.url.clone(),
                        offset: r.offset,
                        length: r.length,
                    },
                );
            }
        }
        Ok(RefsStore {
            entries,
            cache,
            stores: Mutex::new(HashMap::new()),
            validators: tokio::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            miss_bytes: AtomicU64::new(0),
        })
    }

    /// The cache this store reads through, if any.
    pub fn cache(&self) -> Option<&Arc<Cache>> {
        self.cache.as_ref()
    }

    /// Reads so far.
    pub fn stats(&self) -> RefsStats {
        RefsStats {
            hits: self.hits.load(Ordering::Relaxed),
            hit_bytes: self.hit_bytes.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            miss_bytes: self.miss_bytes.load(Ordering::Relaxed),
        }
    }

    /// The byte range stored for a chunk key (`<array>/c/...`), when the
    /// table references one: what a plan reads for that chunk.
    pub fn stored_range(&self, key: &str) -> Option<std::ops::Range<u64>> {
        match self.entries.get(key)? {
            Entry::Ref { offset, length, .. } => Some(*offset..offset + length),
            Entry::Held(_) => None,
        }
    }

    /// For a chunk key the table references: whether the cache holds its
    /// range under the source's current validator, and the range's size.
    pub async fn cached(&self, key: &str) -> Result<Option<(bool, u64)>> {
        let (
            Some(cache),
            Some(Entry::Ref {
                source,
                url,
                offset,
                length,
            }),
        ) = (&self.cache, self.entries.get(key))
        else {
            return Ok(None);
        };
        let v = self.validator(source, url).await?;
        Ok(Some((
            cache.contains(source, *offset..offset + length, &v),
            *length,
        )))
    }

    /// The object store and location for a URL.
    fn locate(&self, url: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath)> {
        let (base, location) = crate::location::split(url)?;
        let mut stores = self.stores.lock().unwrap();
        if let Some(s) = stores.get(&base) {
            return Ok((s.clone(), location));
        }
        let store = crate::location::store(&base)?;
        stores.insert(base, store.clone());
        Ok((store, location))
    }

    /// The validator for `source`, read from `url` the first time.
    async fn validator(&self, source: &str, url: &str) -> Result<Validator> {
        let mut held = self.validators.lock().await;
        if let Some(v) = held.get(source) {
            return Ok(v.clone());
        }
        let local = url::Url::parse(url)
            .ok()
            .filter(|u| u.scheme() == "file")
            .and_then(|u| u.to_file_path().ok());
        let v = if let Some(p) = local {
            let m = std::fs::metadata(&p).map_err(|e| Error::Fetch(format!("{url}: {e}")))?;
            let ns = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            Validator::SizeMtime(format!("{}-{ns}", m.len()))
        } else {
            let (store, location) = self.locate(url)?;
            let m = store
                .head(&location)
                .await
                .map_err(|e| Error::Fetch(format!("{url}: {e}")))?;
            match m.e_tag {
                Some(e) => Validator::Etag(e),
                None => Validator::SizeMtime(format!(
                    "{}-{}",
                    m.size,
                    m.last_modified.timestamp_nanos_opt().unwrap_or(0)
                )),
            }
        };
        held.insert(source.to_string(), v.clone());
        Ok(v)
    }

    async fn fetch(&self, source: &str, url: &str, offset: u64, length: u64) -> Result<Bytes> {
        let range = offset..offset + length;
        let validator = match &self.cache {
            Some(cache) => {
                let v = self.validator(source, url).await?;
                if let Some(b) = cache.get(source, range.clone(), &v) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    self.hit_bytes.fetch_add(length, Ordering::Relaxed);
                    return Ok(b);
                }
                Some(v)
            }
            None => None,
        };
        let (store, location) = self.locate(url)?;
        let bytes = store
            .get_range(&location, range.clone())
            .await
            .map_err(|e| Error::Fetch(format!("{url} {range:?}: {e}")))?;
        if bytes.len() as u64 != length {
            return Err(Error::Fetch(format!(
                "{url} {range:?}: short read of {} bytes",
                bytes.len()
            )));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.miss_bytes.fetch_add(length, Ordering::Relaxed);
        if let (Some(cache), Some(v)) = (&self.cache, validator) {
            cache.put(source, url, range, &v, bytes.clone());
        }
        Ok(bytes)
    }

    async fn value(&self, key: &StoreKey) -> Result<Option<Bytes>> {
        match self.entries.get(key.as_str()) {
            None => Ok(None),
            Some(Entry::Held(b)) => Ok(Some(b.clone())),
            Some(Entry::Ref {
                source,
                url,
                offset,
                length,
            }) => Ok(Some(self.fetch(source, url, *offset, *length).await?)),
        }
    }
}

fn storage(e: Error) -> StorageError {
    StorageError::Other(e.to_string())
}

#[async_trait::async_trait]
impl AsyncReadableStorageTraits for RefsStore {
    async fn get(&self, key: &StoreKey) -> std::result::Result<MaybeBytes, StorageError> {
        self.value(key).await.map_err(storage)
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> std::result::Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        // a chunk is one range of its source; slice the whole of it
        let Some(value) = self.value(key).await.map_err(storage)? else {
            return Ok(None);
        };
        let size = value.len() as u64;
        let parts: Vec<_> = byte_ranges
            .map(|r| {
                let r = r.to_range_usize(size);
                Ok(value.slice(r))
            })
            .collect();
        Ok(Some(futures::stream::iter(parts).boxed()))
    }

    async fn size_key(&self, key: &StoreKey) -> std::result::Result<Option<u64>, StorageError> {
        Ok(match self.entries.get(key.as_str()) {
            None => None,
            Some(Entry::Held(b)) => Some(b.len() as u64),
            Some(Entry::Ref { length, .. }) => Some(*length),
        })
    }

    fn supports_get_partial(&self) -> bool {
        false
    }
}
