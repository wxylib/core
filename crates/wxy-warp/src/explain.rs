//! Plans as values (m04, "EXPLAIN for maps"): what a warp, point read or
//! zonal summary will fetch, before it fetches anything.
//!
//! [`WarpPlan::explain`], [`PointPlan::explain`] and [`ZonePlan::explain`]
//! describe a plan as an [`Explain`]: the source, every chunk the run
//! fetches (in the order it fetches them), the bytes that costs where the
//! source knows, and what the operation planned on top (GDAL's warp
//! chunks and their source windows, the batches of leading slices, the
//! point method after GDAL's downgrade, the zonal spans). An `Explain` is
//! plain data and round-trips through serde.
//!
//! A plan itself is not deserialised: its transformer and geolocation
//! arrays are derived from the source and options, and planning again
//! from the same inputs (all serialisable: [`GridSpec`], [`WarpOptions`],
//! [`PointOptions`], [`ZoneCells`]) gives the same plan.

use serde::{Deserialize, Serialize};
use wxy_source::{ChunkKey, ChunkSource, Window};

use crate::{GridSpec, PointMethod, PointOptions, PointPlan, WarpOptions, WarpPlan, ZoneCells};
use crate::{Route, ZonePlan};

/// What a plan will do, as data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explain {
    pub source: SourceSummary,
    pub fetch: Fetch,
    #[serde(flatten)]
    pub detail: Detail,
}

/// The source a plan reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceSummary {
    /// Where it was opened from, when the caller says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Resolution level read (0 is full resolution), when the caller says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<usize>,
    pub shape: Vec<u64>,
    pub chunk: Vec<u64>,
    pub dtype: String,
}

/// The chunks a plan fetches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fetch {
    /// Distinct chunks.
    pub chunks: usize,
    /// Chunk fetches, counting a chunk again each time it is fetched again
    /// (a warp batch boundary inside a chunk along a leading dim).
    pub fetches: usize,
    /// Of the source's chunks.
    pub of: u64,
    /// Bytes decoded, a whole chunk per fetch.
    pub decoded_bytes: u64,
    /// Bytes read, when the source knows its chunks' stored sizes.
    pub stored_bytes: Option<u64>,
    /// Each fetch's chunk key, in fetch order.
    pub keys: Vec<Vec<u64>>,
}

/// A window `offset`, `size` over some dims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub offset: Vec<u64>,
    pub size: Vec<u64>,
}

/// One of GDAL's warp chunks: a target window and the source window it
/// reads, both as `(x, y)` (column, row) offset and size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarpChunk {
    pub dst_offset: [i32; 2],
    pub dst_size: [i32; 2],
    pub src_offset: [i32; 2],
    pub src_size: [i32; 2],
}

/// What the operation planned.
// one per plan, so the warp variant's size does not matter
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "lowercase")]
pub enum Detail {
    Warp {
        target: GridSpec,
        options: WarpOptions,
        /// `chunked` (GDAL's warper), `plain` (rwarp in one pass), or
        /// `empty` (the target does not see the source).
        route: String,
        /// The source region every slice reads, over `(y, x)`.
        region: Option<Region>,
        /// GDAL's warp chunks (the chunked route), empty ones left out.
        warp_chunks: Vec<WarpChunk>,
        /// Windows over the leading dims, one read and warp each.
        batches: Vec<Region>,
    },
    Points {
        options: PointOptions,
        /// The method used, after GDAL's downgrade for small sources.
        method: PointMethod,
        points: usize,
        on_source: usize,
    },
    Zonal {
        cells: ZoneCells,
        zones: usize,
        /// Row spans held, split at chunk columns.
        spans: usize,
    },
}

fn summary<S: ChunkSource + ?Sized>(source: &S) -> SourceSummary {
    let g = source.grid();
    SourceSummary {
        path: None,
        level: None,
        shape: g.shape.clone(),
        chunk: g.chunk.clone(),
        dtype: format!("{:?}", source.dtype()).to_lowercase(),
    }
}

fn fetch<S: ChunkSource + ?Sized>(source: &S, keys: Vec<ChunkKey>) -> Fetch {
    let g = source.grid();
    let item = source.dtype().size().unwrap_or(0) as u64;
    let per_chunk = item * g.chunk.iter().product::<u64>();
    let stored = keys
        .iter()
        .map(|k| source.stored_bytes(k))
        .sum::<Option<u64>>();
    let distinct = keys.iter().collect::<std::collections::HashSet<_>>().len();
    Fetch {
        chunks: distinct,
        fetches: keys.len(),
        of: g.counts().iter().product(),
        decoded_bytes: per_chunk * keys.len() as u64,
        stored_bytes: stored,
        keys: keys.into_iter().map(|k| k.0).collect(),
    }
}

fn region(w: &Window) -> Region {
    Region {
        offset: w.offset.clone(),
        size: w.size.clone(),
    }
}

impl WarpPlan {
    /// The chunks this warp fetches from `source` in batches of at most
    /// `max_bytes` (as [`WarpPlan::batches`]), and how it is chunked.
    pub fn explain<S: ChunkSource + ?Sized>(&self, source: &S, max_bytes: u64) -> Explain {
        let batches = self.batches(&source.grid().chunk, max_bytes);
        let mut keys = Vec::new();
        if let Some(([y0, x0], [ys, xs])) = self.region() {
            for b in &batches {
                let mut off = b.offset.clone();
                let mut size = b.size.clone();
                off.extend([y0, x0]);
                size.extend([ys, xs]);
                // a planned region is inside the source, so this cannot fail
                keys.extend(
                    source
                        .grid()
                        .plan(&Window::new(off, size))
                        .unwrap_or_default(),
                );
            }
        }
        let (route, warp_chunks) = match &self.route {
            Route::Empty => ("empty", Vec::new()),
            Route::Plain(_) => ("plain", Vec::new()),
            Route::Chunked { chunks, .. } => (
                "chunked",
                chunks
                    .iter()
                    .filter(|c| !c.is_empty())
                    .map(|c| WarpChunk {
                        dst_offset: c.dst_off,
                        dst_size: c.dst_size,
                        src_offset: c.src_off,
                        src_size: c.src_size,
                    })
                    .collect(),
            ),
        };
        Explain {
            source: summary(source),
            fetch: fetch(source, keys),
            detail: Detail::Warp {
                target: self.dst.clone(),
                options: self.opts,
                route: route.into(),
                region: self.region().map(|(o, s)| Region {
                    offset: o.to_vec(),
                    size: s.to_vec(),
                }),
                warp_chunks,
                batches: batches.iter().map(region).collect(),
            },
        }
    }
}

impl PointPlan {
    /// The chunks these point reads fetch from `source`; `options` are the
    /// ones the plan was made with.
    pub fn explain<S: ChunkSource + ?Sized>(&self, source: &S, options: &PointOptions) -> Explain {
        Explain {
            source: summary(source),
            fetch: fetch(source, self.keys()),
            detail: Detail::Points {
                options: *options,
                method: self.method,
                points: self.positions.len(),
                on_source: self.positions.iter().filter(|p| p.is_some()).count(),
            },
        }
    }
}

impl ZonePlan {
    /// The chunks this zonal summary fetches from `source`.
    pub fn explain<S: ChunkSource + ?Sized>(&self, source: &S) -> Explain {
        Explain {
            source: summary(source),
            fetch: fetch(source, self.keys()),
            detail: Detail::Zonal {
                cells: self.cells,
                zones: self.zones,
                spans: self.spans(),
            },
        }
    }
}
