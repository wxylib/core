//! # wxy-warp
//!
//! Warp a wxylib source to a target grid, through
//! [rwarp](https://github.com/hypertidy/rwarp) (design doc 03, A2 and S4).
//! The steps are rwarp's, and each is visible:
//!
//! 1. **plan**: build the transformer (exact, wrapped in rwarp's
//!    approximate transformer at GDAL's default 0.125 pixel error) and
//!    compute the source window the target needs ([`plan`]);
//! 2. **read**: read that window from any [`ChunkSource`] (COG, Zarr,
//!    virtual Zarr);
//! 3. **warp**: resample with rwarp's kernel, all bands in one pass.
//!
//! By default (stage S5, [`WarpOptions::chunked`]) nearest, bilinear and
//! cubic warps run as GDAL's warp operation does instead: chunk by chunk,
//! with the source windows, kernel scales and kernels of GDAL's warper
//! (wxy-resample), so that a target coarser than the source is smoothed as
//! GDAL smooths it and the output matches GDAL's cell for cell. The steps
//! above remain as the plain route.
//!
//! Stage S4 warps domains whose spatial dims carry an affine mapping in a
//! CRS frame. CRS work uses rwarp's pure-Rust `proj4rs` backend (EPSG
//! codes and proj strings, no datum grids); PROJ itself waits on
//! rust-proj-capability.

use rwarp::approx::ApproxTransformer;
use rwarp::source_window::{build_source_grid, compute_source_window, SourceWindow};
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::warp_resample_t;
pub use rwarp::warp::ResampleAlg;
use wxy_domain::{Domain, Frame, Positions};
use wxy_source::{read_window, Buffer, ChunkSource, Error, Result, Values, Window};

mod chunked;

/// A target grid: CRS, GDAL geotransform and size `[ncol, nrow]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    pub crs: String,
    pub geotransform: [f64; 6],
    pub size: [usize; 2],
}

impl GridSpec {
    /// A grid covering `extent` (`[xmin, xmax, ymin, ymax]`) with `size`
    /// cells.
    pub fn from_extent(crs: impl Into<String>, extent: [f64; 4], size: [usize; 2]) -> Self {
        Self {
            crs: crs.into(),
            geotransform: wxy_domain::Affine::from_extent(extent, size).0,
            size,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WarpOptions {
    pub resampling: ResampleAlg,
    /// Maximum error, in pixels, of the approximate transformer. `0`
    /// transforms every pixel exactly.
    pub error_threshold: f64,
    /// Nodata, overriding the source's.
    pub nodata: Option<f64>,
    /// When the source is lon/lat and spans 360 degrees, wrap source
    /// longitudes into it, so that a target cell at exactly 180 (or just
    /// past it) reads from the other edge; when it also spans -90 to 90,
    /// a cell centred on a pole reads the edge row it sits on. GDAL's
    /// warper leaves such cells empty; `false` matches it.
    pub wrap_globe: bool,
    /// Warp as GDAL's warper does (`true`, the default): chunk by chunk,
    /// each chunk's kernel stretched to the source cells a target cell
    /// covers. `false` warps in one pass with rwarp's kernel at 1:1, which
    /// samples rather than smooths a coarser target (bilinear and cubic
    /// then differ from GDAL there).
    pub chunked: bool,
}

impl Default for WarpOptions {
    fn default() -> Self {
        Self {
            resampling: ResampleAlg::NearestNeighbour,
            error_threshold: 0.125,
            nodata: None,
            wrap_globe: true,
            chunked: true,
        }
    }
}

/// Where a source domain sits: its CRS, geotransform, and which dims are
/// y and x (any other dim is a band dim, read whole).
#[derive(Debug, Clone)]
pub struct SourceGrid {
    pub crs: String,
    pub geotransform: [f64; 6],
    pub y_dim: usize,
    pub x_dim: usize,
    pub shape: Vec<u64>,
}

impl SourceGrid {
    /// The primary affine mapping of `domain`. Domains of rank 2, or rank 3
    /// with one leading band dim.
    pub fn from_domain(domain: &Domain) -> Result<Self> {
        let shape = domain.shape();
        if !(shape.len() == 2 || shape.len() == 3) {
            return Err(Error::Unsupported("warp needs a 2D or 3D domain".into()));
        }
        let x = shape.len() - 1;
        let m = domain
            .primary_for(x)
            .ok_or_else(|| Error::Unsupported("no mapping for the x dim".into()))?;
        let Positions::Affine(a) = &m.positions else {
            return Err(Error::Unsupported(format!(
                "warp from {:?} positions (S4 warps affine grids)",
                m.rung()
            )));
        };
        if m.dims != [x - 1, x] {
            return Err(Error::Unsupported(
                "spatial dims must be the last two, as (y, x)".into(),
            ));
        }
        let Frame::Crs(crs) = &m.frame else {
            return Err(Error::Unsupported(format!(
                "warp needs a CRS frame, not {:?}",
                m.frame
            )));
        };
        Ok(Self {
            crs: crs.clone(),
            geotransform: a.0,
            y_dim: x - 1,
            x_dim: x,
            shape,
        })
    }

    fn size(&self) -> [i32; 2] {
        [self.shape[self.x_dim] as i32, self.shape[self.y_dim] as i32]
    }

    /// For an unrotated lon/lat source 360 degrees wide, its width in
    /// columns, and its height in rows when it runs from pole to pole.
    fn globe(&self) -> Result<Option<(f64, Option<f64>)>> {
        let g = self.geotransform;
        let [w, h] = self.size().map(|n| n as f64);
        // Within a hundredth of a cell: steps read from float32 coordinates
        // (0.25 stored as 0.2500000x) miss 360 by far more than rounding.
        let near = |a: f64, b: f64, step: f64| (a - b).abs() < 0.01 * step.abs();
        if g[2] != 0.0 || g[4] != 0.0 || !near((g[1] * w).abs(), 360.0, g[1]) {
            return Ok(None);
        }
        if !proj(&self.crs)?.is_latlong() {
            return Ok(None);
        }
        let (top, bottom) = (g[3], g[3] + g[5] * h);
        let poles = near(top.max(bottom), 90.0, g[5]) && near(top.min(bottom), -90.0, g[5]);
        Ok(Some((w, poles.then_some(h))))
    }

    fn bands(&self) -> u64 {
        if self.shape.len() == 3 {
            self.shape[0]
        } else {
            1
        }
    }
}

/// rwarp's approximate transformer, with two shims for rwarp issues found
/// against GDAL here (fixes are with rwarp's author; remove these when a
/// release carries them):
///
/// - a subdivided scanline can leave one point untransformed and marked
///   failed; such points are transformed exactly, as GDAL does;
/// - its nearest kernel truncates source indices towards zero, so a point
///   at `-1 < x < 0` reads column 0; with `nearest`, points left of or
///   above the source are moved to -2 (still outside after truncation),
///   which is what GDAL's floor gives.
struct Approx {
    /// `None` when the error threshold is 0: every point exact, as GDAL
    /// does without its approximate transformer.
    approx: Option<ApproxTransformer<Globe>>,
    exact: Globe,
    nearest: bool,
    /// The same wrap, applied again after interpolation: for a -180..180
    /// source, the only wrap, catching points just past its edge.
    globe: Option<(f64, Option<f64>)>,
}

/// The exact transformer, with source longitudes wrapped when the source
/// covers the globe but does not start at -180. This is where GDAL wraps
/// (its `+lon_wrap` for sources outside -180..180), under the approximate
/// transformer, so interpolation sees the wrap's jump and subdivides across
/// it: a 0..360 source then matches GDAL at its 0/360 seam. A -180..180
/// source is not wrapped here, so its approximation stays GDAL's.
struct Globe {
    inner: GenImgProjTransformer,
    globe: Option<(f64, Option<f64>)>,
}

impl rwarp::Transformer for Globe {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let ok = self.inner.transform(dst_to_src, x, y);
        if dst_to_src {
            wrap(self.globe, x, y, &ok);
        }
        ok
    }
}

/// Wrap source columns into `[0, w)` and clamp pole rows to the source,
/// for `globe = (w, h)` (see [`SourceGrid::globe`]).
fn wrap(globe: Option<(f64, Option<f64>)>, x: &mut [f64], y: &mut [f64], ok: &[bool]) {
    let Some((w, h)) = globe else { return };
    for i in (0..ok.len()).filter(|&i| ok[i]) {
        x[i] = x[i].rem_euclid(w);
        if let Some(h) = h {
            // A pole maps to row 0 or row h exactly: the edge, not past it.
            y[i] = y[i].clamp(0.0, h - 1e-9);
        }
    }
}

impl rwarp::Transformer for Approx {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (x0, y0) = (x.to_vec(), y.to_vec());
        let mut ok = match &self.approx {
            Some(a) => a.transform(dst_to_src, x, y),
            None => self.exact.transform(dst_to_src, x, y),
        };
        let failed: Vec<usize> = (0..ok.len()).filter(|&i| !ok[i]).collect();
        if !failed.is_empty() {
            let mut fx: Vec<f64> = failed.iter().map(|&i| x0[i]).collect();
            let mut fy: Vec<f64> = failed.iter().map(|&i| y0[i]).collect();
            let fok = self.exact.transform(dst_to_src, &mut fx, &mut fy);
            for (k, &i) in failed.iter().enumerate() {
                x[i] = fx[k];
                y[i] = fy[k];
                ok[i] = fok[k];
            }
        }
        if dst_to_src {
            wrap(self.globe, x, y, &ok);
        }
        if self.nearest && dst_to_src {
            for v in x.iter_mut().chain(y.iter_mut()) {
                if *v < 0.0 && *v > -1.0 {
                    *v = -2.0;
                }
            }
        }
        ok
    }
}

fn transformer(src: &SourceGrid, dst: &GridSpec, opts: &WarpOptions) -> Result<Approx> {
    let (threshold, alg) = (opts.error_threshold, opts.resampling);
    let globe = if opts.wrap_globe { src.globe()? } else { None };
    let west =
        src.geotransform[0].min(src.geotransform[0] + src.geotransform[1] * src.size()[0] as f64);
    let under = globe.filter(|_| (west + 180.0).abs() > 0.01 * src.geotransform[1].abs());
    let make = || -> Result<Globe> {
        let inner =
            GenImgProjTransformer::new(&src.crs, src.geotransform, &dst.crs, dst.geotransform)
                .map_err(Error::Unsupported)?;
        Ok(Globe {
            inner,
            globe: under,
        })
    };
    Ok(Approx {
        approx: if threshold > 0.0 {
            Some(ApproxTransformer::new(make()?, threshold))
        } else {
            None
        },
        exact: make()?,
        // The chunked warp checks source offsets as GDAL does and needs
        // the positions as they are.
        nearest: alg == ResampleAlg::NearestNeighbour && !opts.chunked,
        globe,
    })
}

/// Parse `def` as the CRS backend does: an `EPSG:NNNN` code or a PROJ
/// string (proj4rs: no WKT, PROJJSON or other authorities yet).
pub fn check_crs(def: &str) -> Result<()> {
    proj(def).map(|_| ())
}

fn proj(def: &str) -> Result<proj4rs::proj::Proj> {
    proj4rs::proj::Proj::from_user_string(def)
        .map_err(|e| Error::Unsupported(format!("CRS {def:?}: {e} (EPSG:NNNN or a PROJ string)")))
}

/// The source window the whole target needs, or `None` when the target
/// does not see the source.
///
/// rwarp's backward probe samples the target's edges, which misses source
/// regions inside the target that its edges never reach (a pole inside a
/// polar target; rwarp does not yet do GDAL's special-point check). A
/// forward probe of the source (GDAL's
/// `ComputeSourceWindowStartingFromSource`, rwarp's
/// [`build_source_grid`]) widens the window to cover them. A wider window
/// changes no output values, only how much is read.
pub fn plan(src: &SourceGrid, dst: &GridSpec, opts: &WarpOptions) -> Result<Option<SourceWindow>> {
    let t = transformer(src, dst, opts)?;
    let dst_size = [dst.size[0] as i32, dst.size[1] as i32];
    let size = src.size();
    let back = compute_source_window(&t, [0, 0], dst_size, size, opts.resampling.radius());

    let grid = build_source_grid(&t, size, 21);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for i in 0..grid.dst_x.len() {
        let (dx, dy) = (grid.dst_x[i], grid.dst_y[i]);
        if grid.success[i]
            && dx >= 0.0
            && dx <= dst_size[0] as f64
            && dy >= 0.0
            && dy <= dst_size[1] as f64
        {
            x0 = x0.min(grid.src_x[i]);
            x1 = x1.max(grid.src_x[i]);
            y0 = y0.min(grid.src_y[i]);
            y1 = y1.max(grid.src_y[i]);
        }
    }
    let mut w = match back {
        Some(w) if w.xsize > 0 && w.ysize > 0 => w,
        _ if x0 <= x1 => SourceWindow {
            xoff: 0,
            yoff: 0,
            xsize: 0,
            ysize: 0,
            fill_ratio: 0.0,
            n_failed: 0,
            n_samples: 0,
        },
        _ => return Ok(None),
    };
    if x0 <= x1 {
        // Grid points are sample positions; pad by a cell plus the kernel.
        let pad = opts.resampling.radius() as f64 + 1.0;
        let fx0 = ((x0 - pad).floor().max(0.0)) as i32;
        let fy0 = ((y0 - pad).floor().max(0.0)) as i32;
        let fx1 = ((x1 + pad).ceil().min(size[0] as f64)) as i32;
        let fy1 = ((y1 + pad).ceil().min(size[1] as f64)) as i32;
        let (wx0, wy0) = if w.xsize > 0 {
            (w.xoff.min(fx0), w.yoff.min(fy0))
        } else {
            (fx0, fy0)
        };
        let wx1 = (w.xoff + w.xsize).max(fx1);
        let wy1 = (w.yoff + w.ysize).max(fy1);
        w.xoff = wx0;
        w.yoff = wy0;
        w.xsize = wx1 - wx0;
        w.ysize = wy1 - wy0;
    }
    Ok(Some(w))
}

/// Warp `source` (whose index space is `domain`) to `dst`. The result is
/// `(band, y, x)` (or `(y, x)` for a 2D domain), filled with nodata (or
/// zero) where nothing maps.
pub async fn warp<S: ChunkSource + ?Sized>(
    source: &S,
    domain: &Domain,
    dst: &GridSpec,
    opts: &WarpOptions,
    nodata: Option<f64>,
) -> Result<Buffer> {
    let src = SourceGrid::from_domain(domain)?;
    let nodata = opts.nodata.or(nodata);
    let (ncol, nrow) = (dst.size[0], dst.size[1]);
    let bands = src.bands();
    let out_shape: Vec<u64> = if src.shape.len() == 3 {
        vec![bands, nrow as u64, ncol as u64]
    } else {
        vec![nrow as u64, ncol as u64]
    };
    let out_window = Window::new(vec![0; out_shape.len()], out_shape.clone());
    if opts.chunked {
        if let Some(method) = method(opts.resampling) {
            return warp_chunked(source, &src, dst, opts, nodata, method, out_window).await;
        }
    }

    // The transformer is not Send (PROJ handles), so it is rebuilt after
    // the read rather than held across it.
    let Some(sw) = plan(&src, dst, opts)? else {
        let n = (bands as usize) * ncol * nrow;
        return Ok(Buffer {
            window: out_window,
            values: fill(source.dtype(), n, nodata)?,
        });
    };
    let (xoff, yoff, xs, ys) = (
        sw.xoff as u64,
        sw.yoff as u64,
        sw.xsize as u64,
        sw.ysize as u64,
    );
    let read = if src.shape.len() == 3 {
        Window::new(vec![0, yoff, xoff], vec![bands, ys, xs])
    } else {
        Window::new(vec![yoff, xoff], vec![ys, xs])
    };
    let buf = read_window(source, &read).await?;

    let t = transformer(&src, dst, opts)?;
    let b = bands as usize;
    let (xs, ys) = (xs as usize, ys as usize);
    macro_rules! run {
        ($v:expr, $variant:ident, $t:ty) => {{
            let inter = interleave($v, b, ys * xs);
            let nd = nodata.map(|n| n as $t);
            let out = warp_resample_t::<$t>(
                &t,
                &inter,
                xs,
                ys,
                b,
                xoff as usize,
                yoff as usize,
                ncol,
                nrow,
                nd,
                None,
                opts.resampling,
            );
            Values::$variant(deinterleave(&out, b, ncol * nrow))
        }};
    }
    let values = match &buf.values {
        Values::U8(v) => run!(v, U8, u8),
        Values::I8(v) => run!(v, I8, i8),
        Values::U16(v) => run!(v, U16, u16),
        Values::I16(v) => run!(v, I16, i16),
        Values::U32(v) => run!(v, U32, u32),
        Values::I32(v) => run!(v, I32, i32),
        Values::F32(v) => run!(v, F32, f32),
        Values::F64(v) => run!(v, F64, f64),
        Values::U64(_) | Values::I64(_) => {
            return Err(Error::Unsupported("warping 64-bit integers".into()))
        }
    };
    Ok(Buffer {
        window: out_window,
        values,
    })
}

fn method(alg: ResampleAlg) -> Option<wxy_resample::Method> {
    match alg {
        ResampleAlg::NearestNeighbour => Some(wxy_resample::Method::Nearest),
        ResampleAlg::Bilinear => Some(wxy_resample::Method::Bilinear),
        ResampleAlg::Cubic => Some(wxy_resample::Method::Cubic),
        _ => None,
    }
}

/// Target pixel positions of lon 0 at latitudes -89.9999 and 89.9999 in
/// the source's geographic CRS (GDAL's special points: when one lands in
/// the target, source windows are found by grid sampling). For a projected
/// source the geographic CRS is taken as WGS84.
fn special_points(src: &SourceGrid, t: &Approx) -> Vec<(f64, f64)> {
    let Ok(p) = proj(&src.crs) else {
        return Vec::new();
    };
    let g = src.geotransform;
    let det = g[1] * g[5] - g[2] * g[4];
    if det == 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for lat in [-89.9999f64, 89.9999] {
        let (gx, gy) = if p.is_latlong() {
            (0.0, lat)
        } else {
            let Ok(ll) = proj4rs::proj::Proj::from_user_string("+proj=longlat +datum=WGS84") else {
                continue;
            };
            let mut pt = (0.0f64, lat.to_radians(), 0.0f64);
            if proj4rs::transform::transform(&ll, &p, &mut pt).is_err() {
                continue;
            }
            (pt.0, pt.1)
        };
        let (dx, dy) = (gx - g[0], gy - g[3]);
        let mut x = [(g[5] * dx - g[2] * dy) / det];
        let mut y = [(g[1] * dy - g[4] * dx) / det];
        if rwarp::Transformer::transform(t, false, &mut x, &mut y)[0] {
            out.push((x[0], y[0]));
        }
    }
    out
}

/// The chunked warp (S5): GDAL's chunks, windows and kernels. Bands are
/// warped one at a time, as single-band GDAL warps.
async fn warp_chunked<S: ChunkSource + ?Sized>(
    source: &S,
    src: &SourceGrid,
    dst: &GridSpec,
    opts: &WarpOptions,
    nodata: Option<f64>,
    method: wxy_resample::Method,
    out_window: Window,
) -> Result<Buffer> {
    let (ncol, nrow) = (dst.size[0], dst.size[1]);
    let bands = src.bands() as usize;
    let dtype = source.dtype();
    let bits = 8 * dtype
        .size()
        .ok_or_else(|| Error::Unsupported(format!("warping {dtype:?}")))?;
    // The transformer is not Send, so it is built again after the read.
    let chunks = {
        let t = transformer(src, dst, opts)?;
        let special = special_points(src, &t);
        let planner = chunked::Planner::new(
            &t,
            src.size(),
            [ncol as i32, nrow as i32],
            method,
            special,
            src.globe()?,
            bits,
            nodata.is_some(),
        );
        planner.chunks()
    };
    let live: Vec<&chunked::Chunk> = chunks.iter().filter(|c| !c.is_empty()).collect();
    if live.is_empty() {
        return Ok(Buffer {
            window: out_window,
            values: fill(dtype, bands * ncol * nrow, nodata)?,
        });
    }
    let x0 = live.iter().map(|c| c.src_off[0]).min().unwrap();
    let y0 = live.iter().map(|c| c.src_off[1]).min().unwrap();
    let x1 = live
        .iter()
        .map(|c| c.src_off[0] + c.src_size[0])
        .max()
        .unwrap();
    let y1 = live
        .iter()
        .map(|c| c.src_off[1] + c.src_size[1])
        .max()
        .unwrap();
    let (xs, ys) = ((x1 - x0) as u64, (y1 - y0) as u64);
    let read = if src.shape.len() == 3 {
        Window::new(vec![0, y0 as u64, x0 as u64], vec![bands as u64, ys, xs])
    } else {
        Window::new(vec![y0 as u64, x0 as u64], vec![ys, xs])
    };
    let buf = read_window(source, &read).await?;

    let t = transformer(src, dst, opts)?;
    let (n_src, n_dst) = ((xs * ys) as usize, ncol * nrow);
    let region = ([x0, y0], [x1 - x0, y1 - y0]);
    macro_rules! run {
        ($v:expr, $variant:ident, $t:ty) => {{
            let init = nodata.map(|n| n as $t).unwrap_or_default();
            let mut out = vec![init; bands * n_dst];
            for b in 0..bands {
                chunked::warp_band::<$t>(
                    &t,
                    &chunks,
                    method,
                    &$v[b * n_src..(b + 1) * n_src],
                    region.0,
                    region.1,
                    &mut out[b * n_dst..(b + 1) * n_dst],
                    ncol,
                    nodata,
                );
            }
            Values::$variant(out)
        }};
    }
    let values = match &buf.values {
        Values::U8(v) => run!(v, U8, u8),
        Values::I8(v) => run!(v, I8, i8),
        Values::U16(v) => run!(v, U16, u16),
        Values::I16(v) => run!(v, I16, i16),
        Values::U32(v) => run!(v, U32, u32),
        Values::I32(v) => run!(v, I32, i32),
        Values::U64(v) => run!(v, U64, u64),
        Values::I64(v) => run!(v, I64, i64),
        Values::F32(v) => run!(v, F32, f32),
        Values::F64(v) => run!(v, F64, f64),
    };
    Ok(Buffer {
        window: out_window,
        values,
    })
}

/// Band-major `(band, pixel)` to pixel-interleaved `(pixel, band)`.
fn interleave<T: Copy>(v: &[T], bands: usize, n: usize) -> Vec<T> {
    if bands == 1 {
        return v.to_vec();
    }
    let mut out = Vec::with_capacity(v.len());
    for p in 0..n {
        for b in 0..bands {
            out.push(v[b * n + p]);
        }
    }
    out
}

fn deinterleave<T: Copy>(v: &[T], bands: usize, n: usize) -> Vec<T> {
    if bands == 1 {
        return v.to_vec();
    }
    let mut out = Vec::with_capacity(v.len());
    for b in 0..bands {
        for p in 0..n {
            out.push(v[p * bands + b]);
        }
    }
    out
}

fn fill(dtype: wxy_domain::DataType, n: usize, nodata: Option<f64>) -> Result<Values> {
    let mut v = Values::zeros(dtype, n)?;
    if let Some(nd) = nodata {
        macro_rules! set {
            ($x:expr, $t:ty) => {
                $x.iter_mut().for_each(|e| *e = nd as $t)
            };
        }
        match &mut v {
            Values::U8(x) => set!(x, u8),
            Values::I8(x) => set!(x, i8),
            Values::U16(x) => set!(x, u16),
            Values::I16(x) => set!(x, i16),
            Values::U32(x) => set!(x, u32),
            Values::I32(x) => set!(x, i32),
            Values::U64(x) => set!(x, u64),
            Values::I64(x) => set!(x, i64),
            Values::F32(x) => set!(x, f32),
            Values::F64(x) => set!(x, f64),
        }
    }
    Ok(v)
}
