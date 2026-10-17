//! Planned point extraction (stage S4c; doc 05, E1): points become source
//! cells through the same transformer a warp uses, cells are grouped by
//! chunk, and each chunk is fetched and decoded once, however many points
//! fall in it.
//!
//! A point's cell is the one it falls in, as GDAL's `gdallocationinfo`
//! finds it: reproject the point to the source CRS, apply the inverse
//! geotransform (or invert the geolocation arrays, for a curvilinear
//! source) and take the floor. Every slice of the leading dims is read at
//! that cell.

use std::collections::BTreeMap;

use wxy_domain::Domain;
use wxy_source::{ChunkKey, ChunkSource, Error, Result, Values};

use crate::{transformer, GridSpec, ResampleAlg, SourceGrid, WarpOptions, IDENTITY};

/// Chunks fetched per request.
const FETCH_BATCH: usize = 64;

/// Where each point falls in a source, and the chunks that hold them.
#[derive(Debug, Clone)]
pub struct PointPlan {
    /// Source shape; the last two dims are y and x.
    pub shape: Vec<u64>,
    /// `(row, col)` of each point's cell, or `None` off the source.
    pub cells: Vec<Option<[u64; 2]>>,
    /// Points by the spatial chunk (`[chunk row, chunk col]`) they fall in.
    pub groups: BTreeMap<[u64; 2], Vec<usize>>,
    chunk: Vec<u64>,
}

/// Values at each point, for every slice of the leading dims.
#[derive(Debug, Clone, PartialEq)]
pub struct PointValues {
    /// The leading dims' sizes; `values` is `leading x n` in C order.
    pub leading: Vec<u64>,
    pub values: Values,
    /// Whether each point fell on the source (its values are zero if not).
    pub found: Vec<bool>,
}

/// Plan reads of points `(x, y)` given in `crs` from a source with
/// `domain`, chunked as `chunk` (the source grid's chunk shape). With
/// `wrap_globe`, longitudes wrap into a lon/lat source spanning 360
/// degrees (as [`WarpOptions::wrap_globe`]); without it, a point beyond
/// the source's edge is off it, as in GDAL.
pub fn plan_points(
    domain: &Domain,
    chunk: &[u64],
    crs: &str,
    x: &[f64],
    y: &[f64],
    wrap_globe: bool,
) -> Result<PointPlan> {
    if x.len() != y.len() {
        return Err(Error::Unsupported("x and y differ in length".into()));
    }
    let src = SourceGrid::from_domain(domain)?;
    if chunk.len() != src.shape.len() {
        return Err(Error::Rank {
            got: chunk.len(),
            want: src.shape.len(),
        });
    }
    // The points as a target "grid" whose pixels are their coordinates;
    // every point exact, as gdallocationinfo does.
    let points = GridSpec {
        crs: crs.into(),
        geotransform: IDENTITY,
        size: [x.len().max(1), 1],
    };
    let opts = WarpOptions {
        resampling: ResampleAlg::NearestNeighbour,
        error_threshold: 0.0,
        wrap_globe,
        ..Default::default()
    };
    let t = transformer(&src, &points, &opts)?;
    let (mut px, mut py) = (x.to_vec(), y.to_vec());
    let ok = rwarp::Transformer::transform(&t, true, &mut px, &mut py);
    let (h, w) = (src.shape[src.y_dim], src.shape[src.x_dim]);
    let (ch, cw) = (chunk[src.y_dim], chunk[src.x_dim]);
    let mut cells = Vec::with_capacity(x.len());
    let mut groups: BTreeMap<[u64; 2], Vec<usize>> = BTreeMap::new();
    for i in 0..x.len() {
        let (c, r) = (px[i].floor(), py[i].floor());
        let on = ok[i] && c >= 0.0 && r >= 0.0 && c < w as f64 && r < h as f64;
        if on {
            let (r, c) = (r as u64, c as u64);
            cells.push(Some([r, c]));
            groups.entry([r / ch, c / cw]).or_default().push(i);
        } else {
            cells.push(None);
        }
    }
    Ok(PointPlan {
        shape: src.shape,
        cells,
        groups,
        chunk: chunk.to_vec(),
    })
}

impl PointPlan {
    /// Every chunk the points need, each once, spatial group by spatial
    /// group (all leading chunks of one group together).
    pub fn keys(&self) -> Vec<ChunkKey> {
        let lead = self.shape.len() - 2;
        let counts: Vec<u64> = (0..lead)
            .map(|d| self.shape[d].div_ceil(self.chunk[d]))
            .collect();
        let n_lead: u64 = counts.iter().product();
        let mut keys = Vec::with_capacity(self.groups.len() * n_lead as usize);
        for g in self.groups.keys() {
            for l in 0..n_lead {
                let mut k = vec![0; lead + 2];
                let mut rem = l;
                for d in (0..lead).rev() {
                    k[d] = rem % counts[d];
                    rem /= counts[d];
                }
                k[lead] = g[0];
                k[lead + 1] = g[1];
                keys.push(ChunkKey(k));
            }
        }
        keys
    }
}

/// Read the planned points from `source`: each chunk fetched once.
pub async fn extract_points<S: ChunkSource + ?Sized>(
    source: &S,
    plan: &PointPlan,
) -> Result<PointValues> {
    if source.grid().shape != plan.shape || source.grid().chunk != plan.chunk {
        return Err(Error::Unsupported(
            "source grid differs from the planned one".into(),
        ));
    }
    let n = plan.cells.len();
    let lead = plan.shape.len() - 2;
    let leading = plan.shape[..lead].to_vec();
    let n_lead: u64 = leading.iter().product();
    let mut out = Values::zeros(source.dtype(), n_lead as usize * n)?;
    let keys = plan.keys();
    for batch in keys.chunks(FETCH_BATCH) {
        for e in source.fetch(batch).await? {
            let chunk = source.decode(e)?;
            let k = &chunk.key.0;
            let points = &plan.groups[&[k[lead], k[lead + 1]]];
            let mut strides = vec![1u64; lead + 2];
            for d in (0..lead + 1).rev() {
                strides[d] = strides[d + 1] * chunk.stored[d + 1];
            }
            // every leading index this chunk holds
            let lw = &chunk.window;
            let n_here: u64 = lw.size[..lead].iter().product();
            for l in 0..n_here {
                let (mut rem, mut src_lead, mut dst_lead) = (l, 0u64, 0u64);
                let mut idx = vec![0u64; lead];
                for d in (0..lead).rev() {
                    idx[d] = lw.offset[d] + rem % lw.size[d];
                    rem /= lw.size[d];
                    src_lead += (idx[d] - lw.offset[d]) * strides[d];
                }
                for (d, &i) in idx.iter().enumerate() {
                    dst_lead = dst_lead * leading[d] + i;
                }
                for &p in points {
                    let [r, c] = plan.cells[p].expect("grouped points are on the source");
                    let s = src_lead
                        + (r - lw.offset[lead]) * strides[lead]
                        + (c - lw.offset[lead + 1]);
                    put(
                        &mut out,
                        (dst_lead * n as u64) as usize + p,
                        &chunk.values,
                        s as usize,
                    )?;
                }
            }
        }
    }
    Ok(PointValues {
        leading,
        values: out,
        found: plan.cells.iter().map(Option::is_some).collect(),
    })
}

/// `out[t] = from[s]`, both of one type.
fn put(out: &mut Values, t: usize, from: &Values, s: usize) -> Result<()> {
    macro_rules! each {
        ($($v:ident),*) => {
            match (out, from) {
                $((Values::$v(o), Values::$v(f)) => o[t] = f[s],)*
                _ => return Err(Error::Decode("chunk type differs from source type".into())),
            }
        };
    }
    each!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64);
    Ok(())
}
