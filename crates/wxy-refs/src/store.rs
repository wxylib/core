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

/// One referenced chunk, by its linear index in its array's chunk grid
/// (C order): the source it is read from (an index into
/// [`RefsStore::sources`]) and its byte range there. 32 bytes a chunk,
/// so the 22.8M references of the whole GHRSST MUR series take about
/// 730 MB (m06 C4).
#[derive(Debug, Clone, Copy)]
struct Slot {
    index: u64,
    source: u32,
    offset: u64,
    length: u64,
}

/// The referenced chunks of one array, sorted by linear index.
struct ArrayRefs {
    counts: Vec<u64>,
    slots: Vec<Slot>,
}

impl ArrayRefs {
    fn find(&self, key: &[u64]) -> Option<&Slot> {
        if key.len() != self.counts.len() || key.iter().zip(&self.counts).any(|(k, c)| k >= c) {
            return None;
        }
        let i = linear(key, &self.counts);
        let at = self.slots.binary_search_by_key(&i, |s| s.index).ok()?;
        Some(&self.slots[at])
    }
}

/// The position of `key` in C order over `counts`.
fn linear(key: &[u64], counts: &[u64]) -> u64 {
    key.iter().zip(counts).fold(0, |i, (k, c)| i * c + k)
}

/// What a store key holds.
enum Entry<'a> {
    /// Encoded bytes held in the table (metadata, inline chunks).
    Held(&'a Bytes),
    /// A byte range of a source, read on demand.
    Ref {
        source: &'a str,
        url: &'a str,
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
/// References are held compactly: each source's name and URL once, and
/// per chunk only its grid position, source number and byte range, looked
/// up by binary search. [`RefsStore::from_arrays`] frees each array's
/// [`ChunkRef`](crate::ChunkRef)s as it indexes them.
///
/// With a cache, each range is looked up first under the source's
/// validator (etag, else size and modification time, from the file or a
/// HEAD request, once per source) and kept after a fetch.
pub struct RefsStore {
    held: HashMap<String, Bytes>,
    arrays: HashMap<String, ArrayRefs>,
    sources: Vec<(Arc<str>, Arc<str>)>,
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

/// A chunk store key split into its array's path and grid position.
fn split_chunk_key(key: &str) -> Option<(&str, Vec<u64>)> {
    let (path, rest) = if let Some(rest) = key.strip_prefix("c/") {
        ("", rest)
    } else {
        let at = key.rfind("/c/")?;
        (&key[..at], &key[at + 3..])
    };
    let pos = rest
        .split('/')
        .map(|p| p.parse::<u64>().ok())
        .collect::<Option<Vec<_>>>()?;
    Some((path, pos))
}

/// The chunk grid shape of a Zarr v3 array's metadata.
fn chunk_counts(path: &str, meta: &serde_json::Value) -> Result<Vec<u64>> {
    let ints = |v: &serde_json::Value| -> Option<Vec<u64>> {
        v.as_array()?.iter().map(|x| x.as_u64()).collect()
    };
    let bad = || Error::Decode(format!("{path}: no regular chunk grid in its metadata"));
    let shape = ints(&meta["shape"]).ok_or_else(bad)?;
    let chunks = ints(&meta["chunk_grid"]["configuration"]["chunk_shape"]).ok_or_else(bad)?;
    if shape.len() != chunks.len() || chunks.contains(&0) {
        return Err(bad());
    }
    Ok(shape
        .iter()
        .zip(&chunks)
        .map(|(s, c)| s.div_ceil(*c))
        .collect())
}

/// Builds a store's tables one array at a time.
struct Builder {
    held: HashMap<String, Bytes>,
    arrays: HashMap<String, ArrayRefs>,
    sources: Vec<(Arc<str>, Arc<str>)>,
    ids: HashMap<(Arc<str>, Arc<str>), u32>,
}

impl Builder {
    fn new() -> Self {
        let mut held = HashMap::new();
        held.insert(
            "zarr.json".to_string(),
            Bytes::from_static(br#"{"zarr_format":3,"node_type":"group"}"#),
        );
        Builder {
            held,
            arrays: HashMap::new(),
            sources: Vec::new(),
            ids: HashMap::new(),
        }
    }

    fn add(
        &mut self,
        a: &VirtualArray,
        refs: impl ExactSizeIterator<Item = crate::ChunkRef>,
    ) -> Result<()> {
        let group = Bytes::from_static(br#"{"zarr_format":3,"node_type":"group"}"#);
        let path = a.path.trim_start_matches('/');
        // groups above a nested array
        let parts: Vec<&str> = path.split('/').collect();
        for i in 1..parts.len() {
            self.held
                .entry(format!("{}/zarr.json", parts[..i].join("/")))
                .or_insert_with(|| group.clone());
        }
        let meta = serde_json::to_vec(&a.metadata)
            .map_err(|e| Error::Decode(format!("{}: {e}", a.path)))?;
        self.held.insert(format!("{path}/zarr.json"), meta.into());
        for (k, bytes) in &a.inline {
            self.held
                .insert(chunk_key(path, k), Bytes::copy_from_slice(bytes));
        }
        if refs.len() == 0 {
            return Ok(());
        }
        let counts = chunk_counts(&a.path, &a.metadata)?;
        let mut slots = Vec::with_capacity(refs.len());
        for r in refs {
            if r.key.len() != counts.len() || r.key.iter().zip(&counts).any(|(k, c)| k >= c) {
                return Err(Error::Decode(format!(
                    "{}: chunk {:?} is outside the grid",
                    a.path, r.key
                )));
            }
            let next = self.sources.len() as u32;
            let source = *self
                .ids
                .entry((r.source.clone(), r.url.clone()))
                .or_insert_with(|| {
                    self.sources.push((r.source.clone(), r.url.clone()));
                    next
                });
            slots.push(Slot {
                index: linear(&r.key, &counts),
                source,
                offset: r.offset,
                length: r.length,
            });
        }
        // stable, so of two rows for one chunk the later stands, as the
        // table's last word on it
        slots.sort_by_key(|s| s.index);
        let mut kept: Vec<Slot> = Vec::with_capacity(slots.len());
        for s in slots {
            match kept.last_mut() {
                Some(last) if last.index == s.index => *last = s,
                _ => kept.push(s),
            }
        }
        kept.shrink_to_fit();
        // an inline chunk is held, not referenced
        let inline: std::collections::HashSet<u64> =
            a.inline.iter().map(|(k, _)| linear(k, &counts)).collect();
        if !inline.is_empty() {
            kept.retain(|s| !inline.contains(&s.index));
        }
        self.arrays.insert(
            path.to_string(),
            ArrayRefs {
                counts,
                slots: kept,
            },
        );
        Ok(())
    }

    fn finish(self, cache: Option<Arc<Cache>>) -> RefsStore {
        RefsStore {
            held: self.held,
            arrays: self.arrays,
            sources: self.sources,
            cache,
            stores: Mutex::new(HashMap::new()),
            validators: tokio::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            miss_bytes: AtomicU64::new(0),
        }
    }
}

impl RefsStore {
    /// A store over `arrays`, reading through `cache` when given.
    pub fn new(arrays: &[VirtualArray], cache: Option<Arc<Cache>>) -> Result<Self> {
        let mut b = Builder::new();
        for a in arrays {
            b.add(a, a.refs.iter().cloned())?;
        }
        Ok(b.finish(cache))
    }

    /// A store over `arrays`, as [`RefsStore::new`], freeing each array's
    /// references once they are indexed: what a large table needs.
    pub fn from_arrays(arrays: Vec<VirtualArray>, cache: Option<Arc<Cache>>) -> Result<Self> {
        let mut b = Builder::new();
        for mut a in arrays {
            let refs = std::mem::take(&mut a.refs);
            b.add(&a, refs.into_iter())?;
        }
        Ok(b.finish(cache))
    }

    /// What `key` holds, if anything.
    fn entry(&self, key: &str) -> Option<Entry<'_>> {
        if let Some(b) = self.held.get(key) {
            return Some(Entry::Held(b));
        }
        let (path, pos) = split_chunk_key(key)?;
        let s = self.arrays.get(path)?.find(&pos)?;
        let (source, url) = &self.sources[s.source as usize];
        Some(Entry::Ref {
            source,
            url,
            offset: s.offset,
            length: s.length,
        })
    }

    /// References held, over all arrays.
    pub fn len(&self) -> usize {
        self.arrays.values().map(|a| a.slots.len()).sum()
    }

    /// Whether the store holds no references.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        match self.entry(key)? {
            Entry::Ref { offset, length, .. } => Some(offset..offset + length),
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
        ) = (&self.cache, self.entry(key))
        else {
            return Ok(None);
        };
        let v = self.validator(source, url).await?;
        Ok(Some((
            cache.contains(source, offset..offset + length, &v),
            length,
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
        match self.entry(key.as_str()) {
            None => Ok(None),
            Some(Entry::Held(b)) => Ok(Some(b.clone())),
            Some(Entry::Ref {
                source,
                url,
                offset,
                length,
            }) => Ok(Some(self.fetch(source, url, offset, length).await?)),
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
        Ok(match self.entry(key.as_str()) {
            None => None,
            Some(Entry::Held(b)) => Some(b.len() as u64),
            Some(Entry::Ref { length, .. }) => Some(length),
        })
    }

    fn supports_get_partial(&self) -> bool {
        false
    }
}
