//! # wxy-refs
//!
//! Byte-reference tables read directly (design doc 03, S3c; m06 section 3).
//! A table holds one row per chunk: where its bytes are (path, offset,
//! size), or the bytes themselves. [`table`] turns the mdim-refs export
//! into [`VirtualArray`]s, and [`RefsStore`] serves those to zarrs as
//! storage: metadata from the table, present chunks fetched as byte ranges
//! through `object_store` (local files, HTTP(S), S3), inline chunks from
//! the table, absent chunks reported missing so zarrs fills them. No
//! Icechunk repository is written first; `wxy_virtual::write_icechunk`
//! stays a sink for the same arrays.
//!
//! The byte-range cache ([`wxy_cache`]) sits inside the store: the store
//! holds each chunk's row when it fetches, so it asks the cache first and
//! keeps what it fetched, keyed on the source as the table names it
//! before any remap.

pub mod location;
pub mod store;
pub mod table;

pub use store::{chunk_key, RefsStats, RefsStore};
pub use table::{from_refs_table, from_refs_table_at};

/// A reference to one stored chunk: its key in the Zarr chunk grid, the
/// source it was scanned from (as the table names it, before any remap),
/// the URL it is read from, and the bytes that hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub key: Vec<u64>,
    pub source: String,
    pub url: String,
    pub offset: u64,
    pub length: u64,
}

/// One virtual array: where it goes in the hierarchy, its Zarr v3
/// metadata (`zarr.json`), its chunk references, and any chunks small
/// enough to store inline (encoded bytes by chunk key, such as coordinate
/// values). Chunks with neither read as the fill value.
#[derive(Debug, Clone)]
pub struct VirtualArray {
    pub path: String,
    pub metadata: serde_json::Value,
    pub refs: Vec<ChunkRef>,
    pub inline: Vec<(Vec<u64>, Vec<u8>)>,
}
