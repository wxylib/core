//! A chunk source over a [`RefsStore`] that knows each chunk's stored
//! range from the table, so a plan reports the bytes it reads before any
//! read (m06 9a).

use std::sync::Arc;

use async_trait::async_trait;
use wxy_domain::DataType;
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result};

use crate::store::{chunk_key, RefsStore};

/// `inner` (the array at `path` in `store`, as opened by
/// `wxy_zarr::open`) with each chunk's stored range taken from the table:
/// a referenced chunk reads its range, an inline or absent chunk reads
/// nothing from a source.
pub struct RefsSource<S> {
    inner: S,
    store: Arc<RefsStore>,
    path: String,
}

impl<S> RefsSource<S> {
    pub fn new(inner: S, store: Arc<RefsStore>, path: &str) -> Self {
        RefsSource {
            inner,
            store,
            path: path.trim_start_matches('/').to_string(),
        }
    }

    /// The store the chunks are read through.
    pub fn store(&self) -> &Arc<RefsStore> {
        &self.store
    }
}

#[async_trait]
impl<S: ChunkSource> ChunkSource for RefsSource<S> {
    fn grid(&self) -> &ChunkGrid {
        self.inner.grid()
    }

    fn dtype(&self) -> DataType {
        self.inner.dtype()
    }

    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
        self.inner.fetch(keys).await
    }

    fn decode(&self, encoded: Encoded) -> Result<Chunk> {
        self.inner.decode(encoded)
    }

    fn stored_ranges(&self, key: &ChunkKey) -> Option<Vec<std::ops::Range<u64>>> {
        Some(
            self.store
                .stored_range(&chunk_key(&self.path, &key.0))
                .into_iter()
                .collect(),
        )
    }
}
