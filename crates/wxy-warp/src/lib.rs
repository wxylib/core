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
//! 3. **warp**: resample with rwarp's kernel, one slice (band) at a time.
//!
//! By default (stage S5, [`WarpOptions::chunked`]) nearest, bilinear and
//! cubic warps run as GDAL's warp operation does instead: chunk by chunk,
//! with the source windows, kernel scales and kernels of GDAL's warper
//! (wxy-resample), so that a target coarser than the source is smoothed as
//! GDAL smooths it and the output matches GDAL's cell for cell. The steps
//! above remain as the plain route.
//!
//! Any dims before y and x (time, depth, band) are leading dims (stage
//! S4b): [`WarpPlan`] plans once from the spatial dims and warps the
//! leading slices in batches of whole source chunks, so a 4D store
//! streams through without being loaded, each slice warped as its own 2D
//! warp would be.
//!
//! Stage S4 warps domains whose spatial dims carry an affine mapping in a
//! CRS frame; stage S5b adds curvilinear ones (2D x and y arrays at cell
//! centres), through GDAL's geolocation-array transformer ported in
//! `geoloc`. CRS work uses rwarp's pure-Rust `proj4rs` backend (EPSG
//! codes and proj strings, no datum grids); PROJ itself waits on
//! rust-proj-capability.

use rwarp::approx::ApproxTransformer;
use rwarp::source_window::{build_source_grid, compute_source_window, SourceWindow};
use rwarp::transform::GenImgProjTransformer;
use rwarp::warp::warp_resample_t;
pub use rwarp::warp::ResampleAlg;
use std::sync::Arc;

use wxy_domain::{CellConvention, CoordArray, Domain, Frame, Mapping, Positions};
use wxy_source::{read_window, Buffer, ChunkSource, Error, Result, Values, Window};

mod chunked;
mod extract;
mod geoloc;

pub use extract::{
    extract_points, plan_pixels, plan_points, PointMethod, PointOptions, PointPlan, PointValues,
};

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
    /// Which resolution level [`warp_pyramid`] reads (gdalwarp's `-ovr`).
    /// [`warp`] reads the one source it is given.
    pub overview: Overview,
}

/// Overview choice for [`warp_pyramid`], as gdalwarp's `-ovr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overview {
    /// Full resolution only (`-ovr NONE`).
    None,
    /// The coarsest level still at least as fine as the target, to within
    /// a tenth of a cell (`-ovr AUTO`, gdalwarp's default).
    Auto,
    /// `n` levels finer than `Auto` picks (`-ovr AUTO-n`).
    AutoFiner(usize),
    /// Overview `k` (0 is the first overview, not full resolution), or
    /// the coarsest when there are fewer (`-ovr k`).
    Level(usize),
}

impl Default for WarpOptions {
    fn default() -> Self {
        Self {
            resampling: ResampleAlg::NearestNeighbour,
            error_threshold: 0.125,
            nodata: None,
            wrap_globe: true,
            chunked: true,
            overview: Overview::Auto,
        }
    }
}

/// Where a source domain sits: its CRS, geotransform, and which dims are
/// y and x (the last two; any dims before them are leading dims, warped
/// slice by slice).
#[derive(Debug, Clone)]
pub struct SourceGrid {
    pub crs: String,
    /// The affine mapping; for a curvilinear source, the identity (its
    /// positions come from the geolocation arrays instead).
    pub geotransform: [f64; 6],
    pub y_dim: usize,
    pub x_dim: usize,
    pub shape: Vec<u64>,
    /// GDAL's geolocation-array transformer, for a curvilinear source.
    geoloc: Option<Arc<geoloc::GeoLoc>>,
}

/// GDAL's geotransform for "pixel and georeferenced coordinates are the
/// same": what the geolocation transformer's pixels pass through.
const IDENTITY: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

impl SourceGrid {
    /// The primary affine mapping of `domain`, of rank 2 or more.
    pub fn from_domain(domain: &Domain) -> Result<Self> {
        let shape = domain.shape();
        if shape.len() < 2 {
            return Err(Error::Unsupported("warp needs y and x dims".into()));
        }
        let x = shape.len() - 1;
        let m = domain
            .primary_for(x)
            .ok_or_else(|| Error::Unsupported("no mapping for the x dim".into()))?;
        if m.dims == [x] {
            return Self::from_domain(&axes_to_grid(domain, x)?);
        }
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
        let (geotransform, geoloc) = match &m.positions {
            Positions::Affine(a) => (a.0, None),
            Positions::Curvilinear(arrays) => {
                // centres: one value per cell (GDAL's PIXEL_CENTER);
                // edges: the cell corners, one more row and column
                // (TOP_LEFT_CORNER, which then reaches the far edges)
                let edge = usize::from(m.cell == CellConvention::Edge);
                let (ny, nx) = (shape[x - 1] as usize + edge, shape[x] as usize + edge);
                let [lon, lat] = arrays.as_slice() else {
                    return Err(Error::Unsupported(
                        "a curvilinear warp needs two arrays, x then y".into(),
                    ));
                };
                if lon.shape != [ny, nx] || lat.shape != [ny, nx] {
                    return Err(Error::Unsupported(format!(
                        "curvilinear arrays at cell {:?} must be shaped ({ny}, {nx})",
                        m.cell
                    )));
                }
                let g = geoloc::GeoLoc::new(
                    nx,
                    ny,
                    lon.values.clone(),
                    lat.values.clone(),
                    None,
                    edge == 1,
                    proj(crs)?.is_latlong(),
                )
                .map_err(Error::Unsupported)?;
                (IDENTITY, Some(Arc::new(g)))
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "warp from {:?} positions (affine and curvilinear grids so far)",
                    other.rung()
                )))
            }
        };
        Ok(Self {
            crs: crs.clone(),
            geotransform,
            y_dim: x - 1,
            x_dim: x,
            shape,
            geoloc,
        })
    }

    fn size(&self) -> [i32; 2] {
        [self.shape[self.x_dim] as i32, self.shape[self.y_dim] as i32]
    }

    /// For an unrotated lon/lat source 360 degrees wide, its width in
    /// columns, and its height in rows when it runs from pole to pole.
    fn globe(&self) -> Result<Option<(f64, Option<f64>)>> {
        if self.geoloc.is_some() {
            return Ok(None);
        }
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
    let same = same_crs(&src.crs, &dst.crs);
    let make = || -> Result<Globe> {
        let inner = if let Some(g) = &src.geoloc {
            let (to_src, to_dst) = if same {
                (None, None)
            } else {
                let (a, b) =
                    rwarp::crs::crs_pair(&src.crs, &dst.crs).map_err(Error::Unsupported)?;
                (Some(a), Some(b))
            };
            GenImgProjTransformer::with_backend(
                IDENTITY,
                dst.geotransform,
                Box::new(GeolocToSource {
                    geoloc: g.clone(),
                    reproject: to_src,
                }),
                Box::new(GeolocToTarget {
                    geoloc: g.clone(),
                    reproject: to_dst,
                }),
            )
        } else if same {
            GenImgProjTransformer::with_backend(
                src.geotransform,
                dst.geotransform,
                Box::new(Identity),
                Box::new(Identity),
            )
        } else {
            GenImgProjTransformer::new(&src.crs, src.geotransform, &dst.crs, dst.geotransform)
        }
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

/// Source and target in the same CRS: GDAL then skips reprojection and
/// maps pixel to pixel through the two geotransforms alone. Going through
/// the CRS backend instead (degrees to radians and back) moves positions
/// by rounding error, enough to tip a value rounding at one half.
fn same_crs(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

struct Identity;

impl rwarp::CrsTransform for Identity {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// Target CRS positions to curvilinear source pixels, as GDAL's
/// `GDALGenImgProjTransform` does for a geolocation source: reproject to
/// the arrays' CRS (unless it is the target's), then invert the arrays.
/// It sits where a CRS transform goes, under an identity source
/// geotransform, so the approximate transformer wraps it unchanged.
struct GeolocToSource {
    geoloc: Arc<geoloc::GeoLoc>,
    reproject: Option<Box<dyn rwarp::CrsTransform>>,
}

impl rwarp::CrsTransform for GeolocToSource {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let (x, y) = match &self.reproject {
            Some(r) => r.convert(x, y)?,
            None => (x, y),
        };
        let (mut xs, mut ys, mut ok) = ([x], [y], [true]);
        self.geoloc.inverse(&mut xs, &mut ys, &mut ok);
        ok[0].then_some((xs[0], ys[0]))
    }
}

/// Curvilinear source pixels to target CRS positions.
struct GeolocToTarget {
    geoloc: Arc<geoloc::GeoLoc>,
    reproject: Option<Box<dyn rwarp::CrsTransform>>,
}

impl rwarp::CrsTransform for GeolocToTarget {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let (mut xs, mut ys, mut ok) = ([x], [y], [true]);
        self.geoloc.forward(&mut xs, &mut ys, &mut ok);
        if !ok[0] {
            return None;
        }
        match &self.reproject {
            Some(r) => r.convert(xs[0], ys[0]),
            None => Some((xs[0], ys[0])),
        }
    }
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

/// Warp `source` (whose index space is `domain`) to `dst`. The result has
/// the source's leading dims, then `(y, x)`: `(y, x)` for a 2D domain,
/// `(band, y, x)` for a 3D one, and so on. It is filled with nodata (or
/// zero) where nothing maps. Every slice is read and warped in one go; to
/// stream a large store slice by slice, use [`WarpPlan`].
pub async fn warp<S: ChunkSource + ?Sized>(
    source: &S,
    domain: &Domain,
    dst: &GridSpec,
    opts: &WarpOptions,
    nodata: Option<f64>,
) -> Result<Buffer> {
    let plan = WarpPlan::new(domain, source.dtype(), dst, opts, nodata)?;
    let lead = plan.leading().to_vec();
    plan.warp_slices(source, &Window::new(vec![0; lead.len()], lead))
        .await
}

/// Memory a batch of [`WarpPlan::batches`] aims to stay under by default:
/// the source region and the target, over all slices in the batch.
pub const BATCH_BYTES: u64 = 256 << 20;

/// A warp planned once and applied to every slice of the source's leading
/// (non-spatial) dims (stage S4b): a time x depth x y x x store warps one
/// batch of slices at a time, each slice as the 2D warp of that slice.
///
/// Planning (the transformer, GDAL's chunks and source windows, or the
/// plain route's source window) depends only on the spatial dims, so it is
/// done once; each batch reads the same spatial region across its slices.
#[derive(Debug, Clone)]
pub struct WarpPlan {
    src: SourceGrid,
    dst: GridSpec,
    opts: WarpOptions,
    nodata: Option<f64>,
    dtype: wxy_domain::DataType,
    route: Route,
}

#[derive(Debug, Clone)]
enum Route {
    /// The target does not see the source.
    Empty,
    /// GDAL's chunked warp (S5), over source region `(offset, size)`.
    Chunked {
        method: wxy_resample::Method,
        chunks: Vec<chunked::Chunk>,
        region: ([i32; 2], [i32; 2]),
    },
    /// rwarp in one pass, over this source window.
    Plain(SourceWindow),
}

impl WarpPlan {
    /// Plan the warp of a source of `dtype` over `domain` to `dst`.
    pub fn new(
        domain: &Domain,
        dtype: wxy_domain::DataType,
        dst: &GridSpec,
        opts: &WarpOptions,
        nodata: Option<f64>,
    ) -> Result<Self> {
        let src = SourceGrid::from_domain(domain)?;
        let nodata = opts.nodata.or(nodata);
        let route = match method(opts.resampling).filter(|_| opts.chunked) {
            Some(method) => {
                let bits = 8 * dtype
                    .size()
                    .ok_or_else(|| Error::Unsupported(format!("warping {dtype:?}")))?;
                let t = transformer(&src, dst, opts)?;
                let special = special_points(&src, dst);
                let chunks = chunked::Planner::new(
                    &t,
                    src.size(),
                    [dst.size[0] as i32, dst.size[1] as i32],
                    method,
                    special,
                    src.globe()?,
                    bits,
                    nodata.is_some(),
                    src.geoloc.is_none(),
                )
                .chunks();
                let live: Vec<&chunked::Chunk> = chunks.iter().filter(|c| !c.is_empty()).collect();
                if live.is_empty() {
                    Route::Empty
                } else {
                    let lo = |i: usize| live.iter().map(|c| c.src_off[i]).min().unwrap();
                    let hi = |i: usize| {
                        live.iter()
                            .map(|c| c.src_off[i] + c.src_size[i])
                            .max()
                            .unwrap()
                    };
                    let (x0, y0) = (lo(0), lo(1));
                    let region = ([x0, y0], [hi(0) - x0, hi(1) - y0]);
                    drop(live);
                    Route::Chunked {
                        method,
                        chunks,
                        region,
                    }
                }
            }
            None => match plan(&src, dst, opts)? {
                Some(w) => Route::Plain(w),
                None => Route::Empty,
            },
        };
        Ok(Self {
            src,
            dst: dst.clone(),
            opts: *opts,
            nodata,
            dtype,
            route,
        })
    }

    /// The shape of the source's leading dims (empty for a 2D source).
    pub fn leading(&self) -> &[u64] {
        &self.src.shape[..self.src.y_dim]
    }

    /// The source region every slice reads, as `(offset, size)` over
    /// `(y, x)`, or `None` when the target does not see the source.
    pub fn region(&self) -> Option<([u64; 2], [u64; 2])> {
        let (off, size) = match &self.route {
            Route::Empty => return None,
            Route::Chunked { region, .. } => *region,
            Route::Plain(w) => ([w.xoff, w.yoff], [w.xsize, w.ysize]),
        };
        Some((
            [off[1] as u64, off[0] as u64],
            [size[1] as u64, size[0] as u64],
        ))
    }

    /// Windows over the leading dims, in C order, that together cover
    /// them once. Each batch spans whole source chunks along the leading
    /// dims (`chunk` is the source's chunk shape, all dims), so a chunk
    /// holding several slices is read once, unless that would take the
    /// batch's source region and target over `max_bytes`; then the batch
    /// is halved along its longest dim until it fits (down to one slice).
    pub fn batches(&self, chunk: &[u64], max_bytes: u64) -> Vec<Window> {
        let lead = self.leading();
        let item = self.dtype.size().unwrap_or(8) as u64;
        let region = self.region().map_or(0, |(_, s)| s[0] * s[1]);
        let per_slice = item * (region + (self.dst.size[0] * self.dst.size[1]) as u64);
        let mut block: Vec<u64> = lead
            .iter()
            .enumerate()
            .map(|(i, &n)| chunk.get(i).copied().unwrap_or(1).clamp(1, n.max(1)))
            .collect();
        while block.iter().product::<u64>() * per_slice > max_bytes {
            let (i, &b) = block
                .iter()
                .enumerate()
                .max_by_key(|(_, &b)| b)
                .unwrap_or((0, &1));
            if b <= 1 {
                break;
            }
            block[i] = b.div_ceil(2);
        }
        let counts: Vec<u64> = lead
            .iter()
            .zip(&block)
            .map(|(&n, &b)| n.div_ceil(b))
            .collect();
        if counts.contains(&0) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut idx = vec![0u64; lead.len()];
        loop {
            let offset: Vec<u64> = idx.iter().zip(&block).map(|(i, b)| i * b).collect();
            let size = offset
                .iter()
                .zip(&block)
                .zip(lead)
                .map(|((o, b), n)| (*b).min(n - o))
                .collect();
            out.push(Window::new(offset, size));
            let mut d = lead.len();
            loop {
                if d == 0 {
                    return out;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < counts[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Read and warp the slices in `lead` (a window over the leading
    /// dims; rank 0 for a 2D source). The result's window is `lead`
    /// followed by the whole target `(y, x)`.
    pub async fn warp_slices<S: ChunkSource + ?Sized>(
        &self,
        source: &S,
        lead: &Window,
    ) -> Result<Buffer> {
        let (ncol, nrow) = (self.dst.size[0], self.dst.size[1]);
        if lead.offset.len() != self.leading().len() {
            return Err(Error::Unsupported(format!(
                "a window over {} leading dims, not {}",
                self.leading().len(),
                lead.offset.len()
            )));
        }
        let n_slices = lead.len() as usize;
        let mut offset = lead.offset.clone();
        let mut size = lead.size.clone();
        offset.extend([0, 0]);
        size.extend([nrow as u64, ncol as u64]);
        let out_window = Window::new(offset, size);
        let nodata = self.nodata;
        let Some(([y0, x0], [ys, xs])) = self.region() else {
            return Ok(Buffer {
                window: out_window,
                values: fill(self.dtype, n_slices * ncol * nrow, nodata)?,
            });
        };
        let mut read_off = lead.offset.clone();
        let mut read_size = lead.size.clone();
        read_off.extend([y0, x0]);
        read_size.extend([ys, xs]);
        let buf = read_window(source, &Window::new(read_off, read_size)).await?;

        // The transformer is not Send (PROJ handles), so it is rebuilt
        // after the read rather than held across it.
        let t = transformer(&self.src, &self.dst, &self.opts)?;
        let (n_src, n_dst) = ((xs * ys) as usize, ncol * nrow);
        let b = n_slices;
        let values = match &self.route {
            Route::Empty => unreachable!("an empty route has no region"),
            Route::Chunked { method, chunks, .. } => {
                let origin = [x0 as i32, y0 as i32];
                let extent = [xs as i32, ys as i32];
                macro_rules! run {
                    ($v:expr, $variant:ident, $t:ty) => {{
                        let init = nodata.map(|n| n as $t).unwrap_or_default();
                        let mut out = vec![init; b * n_dst];
                        for k in 0..b {
                            chunked::warp_band::<$t>(
                                &t,
                                chunks,
                                *method,
                                &$v[k * n_src..(k + 1) * n_src],
                                origin,
                                extent,
                                &mut out[k * n_dst..(k + 1) * n_dst],
                                ncol,
                                nodata,
                            );
                        }
                        Values::$variant(out)
                    }};
                }
                match &buf.values {
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
                }
            }
            Route::Plain(_) => {
                let (xs, ys) = (xs as usize, ys as usize);
                macro_rules! run {
                    ($v:expr, $variant:ident, $t:ty) => {{
                        let nd = nodata.map(|n| n as $t);
                        let mut out = Vec::with_capacity(b * n_dst);
                        for k in 0..b {
                            out.extend(warp_resample_t::<$t>(
                                &t,
                                &$v[k * n_src..(k + 1) * n_src],
                                xs,
                                ys,
                                1,
                                x0 as usize,
                                y0 as usize,
                                ncol,
                                nrow,
                                nd,
                                None,
                                self.opts.resampling,
                            ));
                        }
                        Values::$variant(out)
                    }};
                }
                match &buf.values {
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
                }
            }
        };
        Ok(Buffer {
            window: out_window,
            values,
        })
    }
}

/// The overview [`warp_pyramid`] reads for `dst`: an index into the
/// overviews (0 is the first overview), or `None` for full resolution.
/// `widths` are the overviews' widths in cells, finest first.
///
/// gdalwarp's choice with an explicit target extent and size: the target
/// grid's corners and a 10 by 10 lattice are transformed exactly to the
/// full-resolution source; the smaller of the x and y source spans per
/// target cell is the ratio to meet, and the coarsest level whose width
/// ratio stays below it plus a tenth wins.
pub fn choose_overview(
    full: &Domain,
    widths: &[u64],
    dst: &GridSpec,
    opts: &WarpOptions,
) -> Result<Option<usize>> {
    let n = widths.len();
    let auto_finer = match opts.overview {
        Overview::None => return Ok(None),
        _ if n == 0 => return Ok(None),
        Overview::Level(k) => return Ok(Some(k.min(n - 1))),
        Overview::Auto => 0,
        Overview::AutoFiner(f) => f,
    };
    let src = SourceGrid::from_domain(full)?;
    let exact = WarpOptions {
        error_threshold: 0.0,
        ..*opts
    };
    let t = transformer(&src, dst, &exact)?;
    let (w, h) = (dst.size[0] as f64, dst.size[1] as f64);
    let (mut xs, mut ys) = (Vec::with_capacity(100), Vec::with_capacity(100));
    for ix in 0..10 {
        for iy in 0..10 {
            xs.push(w * ix as f64 / 9.0);
            ys.push(h * iy as f64 / 9.0);
        }
    }
    let ok = rwarp::Transformer::transform(&t, true, &mut xs, &mut ys);
    let (mut x0, mut x1, mut y0, mut y1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for i in (0..ok.len()).filter(|&i| ok[i]) {
        x0 = x0.min(xs[i]);
        x1 = x1.max(xs[i]);
        y0 = y0.min(ys[i]);
        y1 = y1.max(ys[i]);
    }
    let rx = if x1 > x0 { (x1 - x0) / w } else { 0.0 };
    let ry = if y1 > y0 { (y1 - y0) / h } else { 0.0 };
    let target = rx.min(ry);
    if target <= 1.0 {
        return Ok(None);
    }
    const EPSILON: f64 = 0.1;
    let full_width = src.shape[src.x_dim] as f64;
    let (mut best, mut best_ratio) = (-1i64, 0.0);
    for k in -1..n as i64 {
        let ratio = if k < 0 {
            1.0
        } else {
            full_width / widths[k as usize] as f64
        };
        if ratio >= target + EPSILON || ratio <= best_ratio {
            continue;
        }
        best = k;
        best_ratio = ratio;
        if (target - ratio).abs() < EPSILON {
            break;
        }
    }
    let k = best - auto_finer as i64;
    Ok((k >= 0).then_some(k as usize))
}

/// [`warp`] from a resolution pyramid: `levels[0]` at full resolution,
/// then overviews finest to coarsest, each with its domain. The level is
/// chosen by [`WarpOptions::overview`] (see [`choose_overview`]).
pub async fn warp_pyramid<S: ChunkSource>(
    levels: &[S],
    domains: &[Domain],
    dst: &GridSpec,
    opts: &WarpOptions,
    nodata: Option<f64>,
) -> Result<Buffer> {
    if levels.is_empty() || levels.len() != domains.len() {
        return Err(Error::Unsupported(
            "warp_pyramid needs one domain per level, at least one".into(),
        ));
    }
    let widths: Vec<u64> = domains[1..]
        .iter()
        .map(|d| *d.shape().last().unwrap_or(&0))
        .collect();
    let k = choose_overview(&domains[0], &widths, dst, opts)?.map_or(0, |k| k + 1);
    warp(&levels[k], &domains[k], dst, opts, nodata).await
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
/// the target, source windows are found by grid sampling). As GDAL's
/// `GDALTransformLonLatToDestGenImgProjTransformer` does: none without a
/// reprojection (source and target CRS the same); for a lon/lat source the
/// lon/lat values go through the target geotransform as they are, without
/// reprojection; a projected source's geographic CRS is taken as WGS84.
/// A domain whose y and x dims carry 1D coordinates (CF, xarray) as one
/// 2D mapping: an affine when both axes are evenly spaced (the
/// normalisation default), otherwise 2D x and y arrays (at cell centres,
/// or edges when an axis is stored as edges), so
/// the warp goes through the geolocation transformer as GDAL's does for
/// irregular axes.
fn axes_to_grid(domain: &Domain, x: usize) -> Result<Domain> {
    let mut d = domain.clone();
    wxy_domain::normalise::normalise_domain(&mut d);
    if let Some((a, frame)) = wxy_domain::normalise::affine_from_axes(&d, x - 1, x) {
        let k = d.add_mapping(Mapping::new(vec![x - 1, x], Positions::Affine(a), frame));
        d.set_primary(k);
        return Ok(d);
    }
    // edges if either axis is stored as edges (a regular axis gives either)
    let edge = [x - 1, x].iter().any(|&d| {
        domain.primary_for(d).is_some_and(|m| {
            m.cell == CellConvention::Edge && matches!(m.positions, Positions::Rectilinear(_))
        })
    });
    let cell = if edge {
        CellConvention::Edge
    } else {
        CellConvention::Centre
    };
    let (ys, fy) = axis(domain, x - 1, cell)?;
    let (xs, fx) = axis(domain, x, cell)?;
    if fy != fx {
        return Err(Error::Unsupported(
            "y and x coordinates in different frames".into(),
        ));
    }
    let (ny, nx) = (ys.len(), xs.len());
    let mut lon = Vec::with_capacity(ny * nx);
    let mut lat = Vec::with_capacity(ny * nx);
    for y in &ys {
        lon.extend_from_slice(&xs);
        lat.extend(std::iter::repeat_n(*y, nx));
    }
    let k = d.add_mapping(
        Mapping::new(
            vec![x - 1, x],
            Positions::Curvilinear(vec![
                CoordArray::new(vec![ny, nx], lon),
                CoordArray::new(vec![ny, nx], lat),
            ]),
            fx,
        )
        .with_cell(cell),
    );
    d.set_primary(k);
    Ok(d)
}

/// Coordinates of dim `d` from its 1D mapping, at cell centres (`n`) or
/// edges (`n + 1`).
fn axis(domain: &Domain, d: usize, cell: CellConvention) -> Result<(Vec<f64>, Frame)> {
    let n = domain.shape()[d] as usize;
    let m = domain
        .primary_for(d)
        .filter(|m| m.dims == [d])
        .ok_or_else(|| Error::Unsupported(format!("no 1D mapping for dim {d}")))?;
    let v = match (&m.positions, cell) {
        (Positions::Regular1 { origin, step }, CellConvention::Centre) => {
            (0..n).map(|i| origin + (i as f64 + 0.5) * step).collect()
        }
        (Positions::Regular1 { origin, step }, CellConvention::Edge) => {
            (0..=n).map(|i| origin + i as f64 * step).collect()
        }
        (Positions::Rectilinear(v), c) if v.len() == 1 && m.cell == c => v[0].clone(),
        (other, _) => {
            return Err(Error::Unsupported(format!(
                "warp from 1D {:?} positions at {:?} with the other axis at {:?}",
                other.rung(),
                m.cell,
                cell
            )))
        }
    };
    Ok((v, m.frame.clone()))
}

fn special_points(src: &SourceGrid, dst: &GridSpec) -> Vec<(f64, f64)> {
    if same_crs(&src.crs, &dst.crs) {
        return Vec::new();
    }
    let Ok(p) = proj(&src.crs) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if p.is_latlong() {
        let g = dst.geotransform;
        let det = g[1] * g[5] - g[2] * g[4];
        if det != 0.0 {
            for lat in [-89.9999f64, 89.9999] {
                let (dx, dy) = (0.0 - g[0], lat - g[3]);
                out.push(((g[5] * dx - g[2] * dy) / det, (g[1] * dy - g[4] * dx) / det));
            }
        }
        return out;
    }
    let Ok(ll) = proj4rs::proj::Proj::from_user_string("+proj=longlat +datum=WGS84") else {
        return Vec::new();
    };
    // Source CRS to target pixels: rwarp's pipeline with the source's
    // positions taken as its pixels.
    let Ok(to_dst) = GenImgProjTransformer::new(&src.crs, IDENTITY, &dst.crs, dst.geotransform)
    else {
        return Vec::new();
    };
    for lat in [-89.9999f64, 89.9999] {
        let mut pt = (0.0f64, lat.to_radians(), 0.0f64);
        if proj4rs::transform::transform(&ll, &p, &mut pt).is_err() {
            continue;
        }
        let (mut x, mut y) = ([pt.0], [pt.1]);
        if rwarp::Transformer::transform(&to_dst, false, &mut x, &mut y)[0] {
            out.push((x[0], y[0]));
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
