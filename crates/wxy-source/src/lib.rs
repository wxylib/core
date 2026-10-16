//! # wxy-source
//!
//! Where a field's numbers come from. Every read is three steps (design doc
//! 02, section 5, decision DM6):
//!
//! 1. **plan**: a [`Window`] in index space becomes a list of [`ChunkKey`]s
//!    through the source's [`ChunkGrid`];
//! 2. **fetch**: I/O-bound and async, [`ChunkSource::fetch`] returns
//!    [`Encoded`] chunks without decoding them;
//! 3. **decode**: CPU-bound and synchronous, [`ChunkSource::decode`] turns one
//!    encoded chunk into a [`Chunk`] of typed values. It can run on any
//!    thread, or in wasm.
//!
//! [`read_window`] runs all three and assembles the chunks into one
//! [`Buffer`]. Callers that want to schedule I/O themselves (merge requests
//! across a whole warp plan, cache encoded chunks, decode in parallel) call
//! the steps directly.

use std::any::Any;
use std::fmt;

use async_trait::async_trait;
use wxy_domain::DataType;

/// Errors from planning, fetching or decoding.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("window {0:?} is outside the source")]
    OutOfBounds(Window),
    #[error("window has {got} dims, source has {want}")]
    Rank { got: usize, want: usize },
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("fetch failed: {0}")]
    Fetch(String),
    #[error("decode failed: {0}")]
    Decode(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A box in index space: an offset and a size per dim.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Window {
    pub offset: Vec<u64>,
    pub size: Vec<u64>,
}

impl Window {
    pub fn new(offset: Vec<u64>, size: Vec<u64>) -> Self {
        assert_eq!(offset.len(), size.len());
        Self { offset, size }
    }

    pub fn len(&self) -> u64 {
        self.size.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The overlap of two windows, or `None` if they do not meet.
    pub fn intersect(&self, other: &Window) -> Option<Window> {
        let mut offset = Vec::with_capacity(self.offset.len());
        let mut size = Vec::with_capacity(self.offset.len());
        for d in 0..self.offset.len() {
            let a = self.offset[d].max(other.offset[d]);
            let b = (self.offset[d] + self.size[d]).min(other.offset[d] + other.size[d]);
            if b <= a {
                return None;
            }
            offset.push(a);
            size.push(b - a);
        }
        Some(Window { offset, size })
    }
}

/// The key of one chunk: its position in the chunk grid.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey(pub Vec<u64>);

/// Regular chunking of index space. Chunk `k` along dim `d` covers indices
/// `k * chunk[d] .. min((k + 1) * chunk[d], shape[d])`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkGrid {
    pub shape: Vec<u64>,
    pub chunk: Vec<u64>,
}

impl ChunkGrid {
    pub fn new(shape: Vec<u64>, chunk: Vec<u64>) -> Self {
        assert_eq!(shape.len(), chunk.len());
        assert!(chunk.iter().all(|&c| c > 0), "chunk sizes must be positive");
        Self { shape, chunk }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Chunks per dim.
    pub fn counts(&self) -> Vec<u64> {
        self.shape
            .iter()
            .zip(&self.chunk)
            .map(|(s, c)| s.div_ceil(*c))
            .collect()
    }

    /// The window a chunk covers, clipped to the shape.
    pub fn window(&self, key: &ChunkKey) -> Window {
        let offset: Vec<u64> = key.0.iter().zip(&self.chunk).map(|(k, c)| k * c).collect();
        let size = offset
            .iter()
            .zip(&self.chunk)
            .zip(&self.shape)
            .map(|((o, c), s)| (*c).min(s - o))
            .collect();
        Window { offset, size }
    }

    /// Plan: the chunks a window touches, in C order (last dim fastest).
    pub fn plan(&self, window: &Window) -> Result<Vec<ChunkKey>> {
        if window.offset.len() != self.rank() {
            return Err(Error::Rank {
                got: window.offset.len(),
                want: self.rank(),
            });
        }
        for d in 0..self.rank() {
            if window.size[d] == 0 || window.offset[d] + window.size[d] > self.shape[d] {
                return Err(Error::OutOfBounds(window.clone()));
            }
        }
        let lo: Vec<u64> = (0..self.rank())
            .map(|d| window.offset[d] / self.chunk[d])
            .collect();
        let hi: Vec<u64> = (0..self.rank())
            .map(|d| (window.offset[d] + window.size[d] - 1) / self.chunk[d])
            .collect();
        let mut keys = Vec::new();
        let mut cur = lo.clone();
        loop {
            keys.push(ChunkKey(cur.clone()));
            let mut d = self.rank();
            loop {
                if d == 0 {
                    return Ok(keys);
                }
                d -= 1;
                if cur[d] < hi[d] {
                    cur[d] += 1;
                    break;
                }
                cur[d] = lo[d];
            }
        }
    }
}

/// Typed values in C order.
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

macro_rules! each_values {
    ($v:expr, $x:ident => $body:expr) => {
        match $v {
            Values::U8($x) => $body,
            Values::U16($x) => $body,
            Values::U32($x) => $body,
            Values::U64($x) => $body,
            Values::I8($x) => $body,
            Values::I16($x) => $body,
            Values::I32($x) => $body,
            Values::I64($x) => $body,
            Values::F32($x) => $body,
            Values::F64($x) => $body,
        }
    };
}

impl Values {
    pub fn len(&self) -> usize {
        each_values!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Values::U8(_) => DataType::U8,
            Values::U16(_) => DataType::U16,
            Values::U32(_) => DataType::U32,
            Values::U64(_) => DataType::U64,
            Values::I8(_) => DataType::I8,
            Values::I16(_) => DataType::I16,
            Values::I32(_) => DataType::I32,
            Values::I64(_) => DataType::I64,
            Values::F32(_) => DataType::F32,
            Values::F64(_) => DataType::F64,
        }
    }

    /// A zeroed buffer of `n` values of `dtype`.
    pub fn zeros(dtype: DataType, n: usize) -> Result<Values> {
        Ok(match dtype {
            DataType::U8 | DataType::Bool => Values::U8(vec![0; n]),
            DataType::U16 => Values::U16(vec![0; n]),
            DataType::U32 => Values::U32(vec![0; n]),
            DataType::U64 => Values::U64(vec![0; n]),
            DataType::I8 => Values::I8(vec![0; n]),
            DataType::I16 => Values::I16(vec![0; n]),
            DataType::I32 => Values::I32(vec![0; n]),
            DataType::I64 => Values::I64(vec![0; n]),
            DataType::F32 => Values::F32(vec![0.0; n]),
            DataType::F64 => Values::F64(vec![0.0; n]),
            other => return Err(Error::Unsupported(format!("buffer of {other:?}"))),
        })
    }

    /// Every value as `f64`, for checks and simple consumers.
    pub fn to_f64(&self) -> Vec<f64> {
        each_values!(self, v => v.iter().map(|&x| x as f64).collect())
    }

    /// Values as little-endian bytes, for comparison with oracle files.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        each_values!(self, v => v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
}

/// Values for a window, in C order over the window's dims.
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    pub window: Window,
    pub values: Values,
}

/// A decoded chunk. `values` cover `stored` (the chunk's full stored extent,
/// which for TIFF edge tiles is larger than the image); `window` is the part
/// inside the source's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub key: ChunkKey,
    pub window: Window,
    pub stored: Vec<u64>,
    pub values: Values,
}

/// An encoded chunk, as fetched. The payload is whatever the source needs to
/// decode it later (bytes, or a format crate's own tile type).
pub struct Encoded {
    pub key: ChunkKey,
    pub payload: Box<dyn Any + Send>,
}

impl fmt::Debug for Encoded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoded")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// A source of chunks for one field.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait ChunkSource: Send + Sync {
    /// Chunking of the field's index space.
    fn grid(&self) -> &ChunkGrid;
    /// The element type `decode` produces.
    fn dtype(&self) -> DataType;
    /// I/O bound: fetch encoded chunks, in the order given.
    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>>;
    /// CPU bound: decode one chunk.
    fn decode(&self, encoded: Encoded) -> Result<Chunk>;
}

/// Plan, fetch and decode a window, and assemble the chunks into one buffer.
pub async fn read_window<S: ChunkSource + ?Sized>(source: &S, window: &Window) -> Result<Buffer> {
    let keys = source.grid().plan(window)?;
    let encoded = source.fetch(&keys).await?;
    let mut out = Values::zeros(source.dtype(), window.len() as usize)?;
    for e in encoded {
        let chunk = source.decode(e)?;
        copy_into(&chunk, window, &mut out)?;
    }
    Ok(Buffer {
        window: window.clone(),
        values: out,
    })
}

/// Copy the part of a chunk that falls in `window` into `out` (C order over
/// the window).
fn copy_into(chunk: &Chunk, window: &Window, out: &mut Values) -> Result<()> {
    let Some(overlap) = chunk.window.intersect(window) else {
        return Ok(());
    };
    let rank = window.offset.len();
    let strides = |shape: &[u64]| {
        let mut s = vec![1u64; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    };
    let src_strides = strides(&chunk.stored);
    let dst_strides = strides(&window.size);
    let inner = overlap.size[rank - 1] as usize;
    let outer: Vec<u64> = overlap.size[..rank - 1].to_vec();
    let n_outer: u64 = outer.iter().product();
    for o in 0..n_outer {
        // position within the overlap for all but the last dim
        let mut rem = o;
        let mut src_off = 0u64;
        let mut dst_off = 0u64;
        for d in (0..rank - 1).rev() {
            let i = rem % outer[d];
            rem /= outer[d];
            let g = overlap.offset[d] + i;
            src_off += (g - chunk.window.offset[d]) * src_strides[d];
            dst_off += (g - window.offset[d]) * dst_strides[d];
        }
        let g = overlap.offset[rank - 1];
        src_off += g - chunk.window.offset[rank - 1];
        dst_off += g - window.offset[rank - 1];
        let (s, t) = (src_off as usize, dst_off as usize);
        macro_rules! cp {
            ($a:ident, $b:ident) => {
                $b[t..t + inner].copy_from_slice(&$a[s..s + inner])
            };
        }
        match (&chunk.values, &mut *out) {
            (Values::U8(a), Values::U8(b)) => cp!(a, b),
            (Values::U16(a), Values::U16(b)) => cp!(a, b),
            (Values::U32(a), Values::U32(b)) => cp!(a, b),
            (Values::U64(a), Values::U64(b)) => cp!(a, b),
            (Values::I8(a), Values::I8(b)) => cp!(a, b),
            (Values::I16(a), Values::I16(b)) => cp!(a, b),
            (Values::I32(a), Values::I32(b)) => cp!(a, b),
            (Values::I64(a), Values::I64(b)) => cp!(a, b),
            (Values::F32(a), Values::F32(b)) => cp!(a, b),
            (Values::F64(a), Values::F64(b)) => cp!(a, b),
            _ => return Err(Error::Decode("chunk type differs from source type".into())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_covers_touched_chunks_in_c_order() {
        let g = ChunkGrid::new(vec![1, 300, 200], vec![1, 128, 128]);
        assert_eq!(g.counts(), vec![1, 3, 2]);
        let keys = g
            .plan(&Window::new(vec![0, 100, 100], vec![1, 60, 60]))
            .unwrap();
        let k: Vec<Vec<u64>> = keys.into_iter().map(|k| k.0).collect();
        assert_eq!(
            k,
            vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]]
        );
        assert_eq!(g.window(&ChunkKey(vec![0, 2, 1])).size, vec![1, 44, 72]);
    }

    #[test]
    fn plan_rejects_out_of_bounds() {
        let g = ChunkGrid::new(vec![10, 10], vec![4, 4]);
        assert!(g.plan(&Window::new(vec![8, 0], vec![4, 1])).is_err());
    }

    /// A source whose value at (r, c) is r * 1000 + c, chunked 3 x 4.
    struct Synthetic(ChunkGrid);

    #[async_trait]
    impl ChunkSource for Synthetic {
        fn grid(&self) -> &ChunkGrid {
            &self.0
        }
        fn dtype(&self) -> DataType {
            DataType::I32
        }
        async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
            Ok(keys
                .iter()
                .map(|k| Encoded {
                    key: k.clone(),
                    payload: Box::new(()),
                })
                .collect())
        }
        fn decode(&self, e: Encoded) -> Result<Chunk> {
            let w = self.0.window(&e.key);
            let stored = self.0.chunk.clone();
            let mut v = Vec::new();
            for r in 0..stored[0] {
                for c in 0..stored[1] {
                    v.push(((w.offset[0] + r) * 1000 + w.offset[1] + c) as i32);
                }
            }
            Ok(Chunk {
                key: e.key,
                window: w,
                stored,
                values: Values::I32(v),
            })
        }
    }

    #[tokio::test]
    async fn read_window_assembles_across_chunks_and_edges() {
        let src = Synthetic(ChunkGrid::new(vec![10, 10], vec![3, 4]));
        let w = Window::new(vec![2, 3], vec![8, 7]);
        let b = read_window(&src, &w).await.unwrap();
        let Values::I32(v) = b.values else { panic!() };
        for r in 0..8 {
            for c in 0..7 {
                assert_eq!(v[r * 7 + c], ((2 + r) * 1000 + 3 + c) as i32);
            }
        }
    }
}
