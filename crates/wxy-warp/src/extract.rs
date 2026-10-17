//! Planned point extraction (stage S4c; doc 05, E1): points become source
//! positions through the same transformer a warp uses, the cells each
//! point needs are grouped by chunk, and each chunk is fetched and decoded
//! once, however many points fall in it.
//!
//! Positions are found as GDAL finds them: reproject the point to the
//! source CRS, then apply the inverse geotransform (or invert the
//! geolocation arrays, for a curvilinear source). Every slice of the
//! leading dims is read there.
//!
//! - [`PointMethod::Nearest`] reads the cell the point falls in (the
//!   floor), as `gdallocationinfo` does, nodata included.
//! - The interpolated methods are GDAL's `InterpolateAtPoint` (GDAL 3.10,
//!   `gdal_interpolateatpoint.cpp`): kernels on cell centres, edge cells
//!   repeated beyond the edge, no value where a bilinear kernel touches
//!   nodata, and nodata left out of a cubic kernel's weights.

use std::collections::{BTreeMap, BTreeSet};

use wxy_domain::Domain;
use wxy_source::{Chunk, ChunkKey, ChunkSource, Error, Result, Values};

use crate::{transformer, GridSpec, ResampleAlg, SourceGrid, WarpOptions, IDENTITY};

/// Chunks fetched per request.
const FETCH_BATCH: usize = 64;

/// How a value is read at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointMethod {
    /// The cell the point falls in, in the source's type.
    #[default]
    Nearest,
    /// GDAL's bilinear `InterpolateAtPoint`, as f64.
    Bilinear,
    /// GDAL's cubic (Catmull-Rom) `InterpolateAtPoint`, as f64.
    Cubic,
    /// GDAL's cubic B-spline `InterpolateAtPoint`, as f64.
    CubicSpline,
}

/// Options for [`plan_points`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointOptions {
    pub method: PointMethod,
    /// Wrap longitudes into a lon/lat source spanning 360 degrees (as
    /// [`WarpOptions::wrap_globe`]), and let kernels wrap across its edge.
    /// Without it a point beyond the source's edge is off it, as in GDAL.
    pub wrap_globe: bool,
}

impl Default for PointOptions {
    fn default() -> Self {
        Self {
            method: PointMethod::Nearest,
            wrap_globe: true,
        }
    }
}

/// Where each point falls in a source, and the chunks that hold the cells
/// it needs.
#[derive(Debug, Clone)]
pub struct PointPlan {
    /// Source shape; the last two dims are y and x.
    pub shape: Vec<u64>,
    /// `(pixel, line)` of each point (0 at the outer edge of the first
    /// cell), or `None` off the source.
    pub positions: Vec<Option<[f64; 2]>>,
    /// Points by the spatial chunks (`[chunk row, chunk col]`) their cells
    /// fall in; a point whose kernel crosses chunks is in each.
    pub groups: BTreeMap<[u64; 2], Vec<usize>>,
    /// The method after GDAL's downgrade for small sources.
    pub method: PointMethod,
    /// Each point's kernel origin, `(row, col)` before clamping or wrapping.
    origins: Vec<[i64; 2]>,
    wrap_x: bool,
    chunk: Vec<u64>,
}

/// Values at each point, for every slice of the leading dims.
#[derive(Debug, Clone, PartialEq)]
pub struct PointValues {
    /// The leading dims' sizes; `values` is `leading x n` in C order.
    pub leading: Vec<u64>,
    /// In the source's type for [`PointMethod::Nearest`] (zero where not
    /// found); f64 otherwise (NaN where GDAL has no value).
    pub values: Values,
    /// Whether each point fell on the source.
    pub found: Vec<bool>,
}

impl PointMethod {
    fn kernel(self) -> usize {
        match self {
            PointMethod::Nearest => 1,
            PointMethod::Bilinear => 2,
            PointMethod::Cubic | PointMethod::CubicSpline => 4,
        }
    }
}

/// Plan reads of points `(x, y)` given in `crs` from a source with
/// `domain`, chunked as `chunk` (the source grid's chunk shape).
pub fn plan_points(
    domain: &Domain,
    chunk: &[u64],
    crs: &str,
    x: &[f64],
    y: &[f64],
    options: &PointOptions,
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
    // every point exact, as GDAL transforms a single point.
    let points = GridSpec {
        crs: crs.into(),
        geotransform: IDENTITY,
        size: [x.len().max(1), 1],
    };
    let opts = WarpOptions {
        resampling: ResampleAlg::NearestNeighbour,
        error_threshold: 0.0,
        wrap_globe: options.wrap_globe,
        ..Default::default()
    };
    let t = transformer(&src, &points, &opts)?;
    let (mut px, mut py) = (x.to_vec(), y.to_vec());
    let ok = rwarp::Transformer::transform(&t, true, &mut px, &mut py);
    plan_from(src, chunk, &px, &py, &ok, options)
}

/// Plan reads at source positions given directly: `pixel` and `line` as
/// GDAL counts them (0 at the outer edge of the first cell).
pub fn plan_pixels(
    domain: &Domain,
    chunk: &[u64],
    pixel: &[f64],
    line: &[f64],
    options: &PointOptions,
) -> Result<PointPlan> {
    if pixel.len() != line.len() {
        return Err(Error::Unsupported("pixel and line differ in length".into()));
    }
    let src = SourceGrid::from_domain(domain)?;
    if chunk.len() != src.shape.len() {
        return Err(Error::Rank {
            got: chunk.len(),
            want: src.shape.len(),
        });
    }
    let ok: Vec<bool> = pixel
        .iter()
        .zip(line)
        .map(|(p, l)| p.is_finite() && l.is_finite())
        .collect();
    plan_from(src, chunk, pixel, line, &ok, options)
}

fn plan_from(
    src: SourceGrid,
    chunk: &[u64],
    px: &[f64],
    py: &[f64],
    ok: &[bool],
    options: &PointOptions,
) -> Result<PointPlan> {
    let (h, w) = (src.shape[src.y_dim], src.shape[src.x_dim]);
    let (ch, cw) = (chunk[src.y_dim], chunk[src.x_dim]);
    let wrap_x = options.wrap_globe && src.globe()?.is_some();

    // GDAL's downgrade for sources too small for the kernel
    let mut method = options.method;
    if (w < 4 || h < 4) && matches!(method, PointMethod::Cubic | PointMethod::CubicSpline) {
        method = PointMethod::Bilinear;
    }
    if (w < 2 || h < 2) && method == PointMethod::Bilinear {
        method = PointMethod::Nearest;
    }
    let k = method.kernel() as i64;

    let mut positions = Vec::with_capacity(px.len());
    let mut origins = Vec::with_capacity(px.len());
    let mut groups: BTreeMap<[u64; 2], Vec<usize>> = BTreeMap::new();
    let (wf, hf) = (w as f64, h as f64);
    for i in 0..px.len() {
        let (p, l) = (px[i], py[i]);
        let on = ok[i]
            && if method == PointMethod::Nearest {
                p >= 0.0 && l >= 0.0 && p < wf && l < hf
            } else {
                p >= 0.0 && l >= 0.0 && p <= wf && l <= hf
            };
        if !on {
            positions.push(None);
            origins.push([0, 0]);
            continue;
        }
        positions.push(Some([p, l]));
        let o = if method == PointMethod::Nearest {
            [l.floor() as i64, p.floor() as i64]
        } else {
            // kernels sit on cell centres; a 4 x 4 kernel starts one before
            let back = (k - 2) / 2;
            [
                (l - 0.5).floor() as i64 - back,
                (p - 0.5).floor() as i64 - back,
            ]
        };
        origins.push(o);
        let rows: BTreeSet<u64> = (0..k).map(|j| clamp(o[0] + j, h) / ch).collect();
        let cols: BTreeSet<u64> = (0..k).map(|j| col(o[1] + j, w, wrap_x) / cw).collect();
        for &r in &rows {
            for &c in &cols {
                groups.entry([r, c]).or_default().push(i);
            }
        }
    }
    Ok(PointPlan {
        shape: src.shape,
        positions,
        groups,
        method,
        origins,
        wrap_x,
        chunk: chunk.to_vec(),
    })
}

/// A kernel row clamped to the source: edge cells repeat beyond the edge.
fn clamp(i: i64, n: u64) -> u64 {
    i.clamp(0, n as i64 - 1) as u64
}

/// A kernel column: wrapped around a 360 degree source, else clamped.
fn col(i: i64, n: u64, wrap: bool) -> u64 {
    if wrap {
        i.rem_euclid(n as i64) as u64
    } else {
        clamp(i, n)
    }
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
/// `nodata` is the source's (it matters to the interpolated methods).
pub async fn extract_points<S: ChunkSource + ?Sized>(
    source: &S,
    plan: &PointPlan,
    nodata: Option<f64>,
) -> Result<PointValues> {
    if source.grid().shape != plan.shape || source.grid().chunk != plan.chunk {
        return Err(Error::Unsupported(
            "source grid differs from the planned one".into(),
        ));
    }
    let n = plan.positions.len();
    let lead = plan.shape.len() - 2;
    let leading = plan.shape[..lead].to_vec();
    let n_lead = leading.iter().product::<u64>() as usize;
    let (h, w) = (plan.shape[lead], plan.shape[lead + 1]);
    let k = plan.method.kernel();
    let kk = k * k;
    let nearest = plan.method == PointMethod::Nearest;
    let mut out = if nearest {
        Values::zeros(source.dtype(), n_lead * n)?
    } else {
        Values::F64(Vec::new())
    };
    // each point's kernel values, per leading slice, filled chunk by chunk
    let mut kernels = if nearest {
        Vec::new()
    } else {
        vec![f64::NAN; n_lead * n * kk]
    };
    let keys = plan.keys();
    for batch in keys.chunks(FETCH_BATCH) {
        for e in source.fetch(batch).await? {
            let chunk = source.decode(e)?;
            let key = &chunk.key.0;
            let lw = &chunk.window;
            let (r0, c0) = (lw.offset[lead], lw.offset[lead + 1]);
            let (r1, c1) = (r0 + lw.size[lead], c0 + lw.size[lead + 1]);
            for (lead_src, l) in leading_slices(&chunk, lead, &leading) {
                for &p in &plan.groups[&[key[lead], key[lead + 1]]] {
                    let o = plan.origins[p];
                    for i in 0..k {
                        let r = clamp(o[0] + i as i64, h);
                        if r < r0 || r >= r1 {
                            continue;
                        }
                        for j in 0..k {
                            let c = col(o[1] + j as i64, w, plan.wrap_x);
                            if c < c0 || c >= c1 {
                                continue;
                            }
                            let s = lead_src + (r - r0) * stride(&chunk, lead) + (c - c0);
                            if nearest {
                                put(&mut out, l * n + p, &chunk.values, s as usize)?;
                            } else {
                                kernels[(l * n + p) * kk + i * k + j] =
                                    get(&chunk.values, s as usize);
                            }
                        }
                    }
                }
            }
        }
    }
    if !nearest {
        let mut v = vec![f64::NAN; n_lead * n];
        for (q, out) in v.iter_mut().enumerate() {
            if let Some([px, py]) = plan.positions[q % n] {
                let a = &kernels[q * kk..][..kk];
                let delta = [
                    (px - 0.5) - (px - 0.5).floor(),
                    (py - 0.5) - (py - 0.5).floor(),
                ];
                if let Some(x) = interpolate(plan.method, delta, a, nodata) {
                    *out = x;
                }
            }
        }
        out = Values::F64(v);
    }
    Ok(PointValues {
        leading,
        values: out,
        found: plan.positions.iter().map(Option::is_some).collect(),
    })
}

/// The stride of the y dim in a chunk's stored values.
fn stride(chunk: &Chunk, lead: usize) -> u64 {
    chunk.stored[lead + 1]
}

/// Each leading index a chunk holds: its offset into the chunk's values,
/// and its flat index over all leading slices.
fn leading_slices(chunk: &Chunk, lead: usize, leading: &[u64]) -> Vec<(u64, usize)> {
    let mut strides = vec![1u64; lead + 2];
    for d in (0..lead + 1).rev() {
        strides[d] = strides[d + 1] * chunk.stored[d + 1];
    }
    let lw = &chunk.window;
    let n_here: u64 = lw.size[..lead].iter().product();
    (0..n_here)
        .map(|l| {
            let (mut rem, mut src, mut dst) = (l, 0u64, 0u64);
            let mut idx = vec![0u64; lead];
            for d in (0..lead).rev() {
                idx[d] = lw.offset[d] + rem % lw.size[d];
                rem /= lw.size[d];
                src += (idx[d] - lw.offset[d]) * strides[d];
            }
            for (d, &i) in idx.iter().enumerate() {
                dst = dst * leading[d] + i;
            }
            (src, dst as usize)
        })
        .collect()
}

/// GDAL's kernels on `a` (`k x k`, row by row), `delta` the point's offset
/// from the first centre cell it falls between (x, y).
fn interpolate(
    method: PointMethod,
    delta: [f64; 2],
    a: &[f64],
    nodata: Option<f64>,
) -> Option<f64> {
    let is_nodata = |v: f64| nodata.is_some_and(|nd| are_real_equal(nd, v));
    match method {
        PointMethod::Nearest => unreachable!(),
        PointMethod::Bilinear => {
            if a.iter().any(|&v| is_nodata(v)) {
                return None;
            }
            let d1 = [1.0 - delta[0], 1.0 - delta[1]];
            let xz1 = a[0] * d1[0] + a[1] * delta[0];
            let xz2 = a[2] * d1[0] + a[3] * delta[0];
            Some(xz1 * d1[1] + xz2 * delta[1])
        }
        PointMethod::Cubic | PointMethod::CubicSpline => {
            let kernel = if method == PointMethod::Cubic {
                cubic
            } else {
                cubic_spline
            };
            let (mut sum, mut weight) = (0.0, 0.0);
            for ki in 0..4 {
                for kj in 0..4 {
                    let fx = (kj as i32 - 1) as f64 - delta[0];
                    let fy = (ki as i32 - 1) as f64 - delta[1];
                    let wgt = kernel(fx) * kernel(fy);
                    let v = a[kj + ki * 4];
                    if is_nodata(v) {
                        continue;
                    }
                    sum += v * wgt;
                    weight += wgt;
                }
            }
            (weight != 0.0).then(|| sum / weight)
        }
    }
}

/// GDAL's `ARE_REAL_EQUAL` for doubles (float epsilon, 2 ulp).
fn are_real_equal(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() < f32::EPSILON as f64 * (a + b).abs() * 2.0
}

/// GDAL's `CubicKernel` (Catmull-Rom).
fn cubic(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 1.0 {
        let x2 = x * x;
        x2 * (1.5 * ax - 2.5) + 1.0
    } else if ax <= 2.0 {
        let x2 = x * x;
        x2 * (-0.5 * ax + 2.5) - 4.0 * ax + 2.0
    } else {
        0.0
    }
}

/// GDAL's `CubicSplineKernel` (cubic B-spline).
fn cubic_spline(v: f64) -> f64 {
    if v > 2.0 {
        return 0.0;
    }
    let (xm1, xp1, xp2) = (v - 1.0, v + 1.0, v + 2.0);
    let a = if xp2 <= 0.0 { 0.0 } else { xp2 * xp2 * xp2 };
    let b = if xp1 <= 0.0 { 0.0 } else { xp1 * xp1 * xp1 };
    let c = if v <= 0.0 { 0.0 } else { v * v * v };
    let d = if xm1 <= 0.0 { 0.0 } else { xm1 * xm1 * xm1 };
    0.166_666_666_666_666_67 * (a - (4.0 * b) + (6.0 * c) - (4.0 * d))
}

/// One value as f64.
fn get(v: &Values, s: usize) -> f64 {
    match v {
        Values::U8(a) => a[s] as f64,
        Values::U16(a) => a[s] as f64,
        Values::U32(a) => a[s] as f64,
        Values::U64(a) => a[s] as f64,
        Values::I8(a) => a[s] as f64,
        Values::I16(a) => a[s] as f64,
        Values::I32(a) => a[s] as f64,
        Values::I64(a) => a[s] as f64,
        Values::F32(a) => a[s] as f64,
        Values::F64(a) => a[s],
    }
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
