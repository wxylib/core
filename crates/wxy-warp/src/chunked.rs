//! The chunked warp: GDAL's warp operation, chunk by chunk, with the
//! wxy-resample kernels (design doc 03, S5).
//!
//! Nearest, bilinear and cubic run wxy-resample's kernels (GDAL 3.12's);
//! lanczos, average and mode run rwarp's `warp` on each chunk, with the
//! chunk's source window and the kernel scale GDAL 3.12 gives the chunk.
//!
//! GDAL splits the target into chunks (`CollectChunkList`), computes each
//! chunk's source window (`ComputeSourceWindow`) and, from the window's
//! size, how much to stretch the kernel (`PerformWarp`). The chunk
//! boundaries matter to the output: they set the kernel scale, the window
//! edges where kernels are clipped, and the scanline extents the
//! approximate transformer interpolates over. This module ports those
//! steps so the output matches GDAL's.
//!
//! Not ported: GDAL's retry with `CHECK_WITH_INVERT_PROJ` when a window
//! comes back with wild values (a PROJ behaviour the proj4rs backend has no
//! switch for), destination alpha, cutlines and source masks other than
//! nodata.

use rwarp::warp::{KernelScaling, ResampleAlg, SourceBuffer};
use rwarp::Transformer;
use wxy_resample::{write_density, Kernel, Method, Sample, Scale, Window, BAND_DENSITY_THRESHOLD};

use crate::{Approx, WarpOptions};

/// The kernels a chunked warp runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ChunkKernel {
    /// wxy-resample's port of GDAL 3.12's kernels.
    Resample(Method),
    /// rwarp's `warp`, one call per chunk.
    Rwarp(ResampleAlg),
}

/// GDAL's `DEFAULT_STEP_COUNT`.
const STEPS: usize = 21;
/// GDAL's default warp memory limit, 64 MB.
pub(crate) const MEMORY_LIMIT: f64 = 64.0 * 1024.0 * 1024.0;

/// One chunk of the target and the source window it reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Chunk {
    pub dst_off: [i32; 2],
    pub dst_size: [i32; 2],
    pub src_off: [i32; 2],
    pub src_size: [i32; 2],
    /// Kernel margin included in `src_size` (GDAL's `dfSrcXExtraSize`).
    pub extra: [f64; 2],
    /// The kernel scale, target cells per source cell along x and y.
    pub scale: [f64; 2],
}

impl Chunk {
    pub fn is_empty(&self) -> bool {
        self.src_size[0] <= 0 || self.src_size[1] <= 0
    }
}

/// A source window and its fill ratio (GDAL's `ComputeSourceWindow`).
struct SourceWindow {
    off: [i32; 2],
    size: [i32; 2],
    extra: [f64; 2],
    fill: f64,
}

const EMPTY: SourceWindow = SourceWindow {
    off: [0, 0],
    size: [0, 0],
    extra: [0.0, 0.0],
    fill: 0.0,
};

/// A source grid point forward transformed: (src x, src y, dst x, dst y,
/// ok).
type GridPoint = (f64, f64, f64, f64, bool);

/// Plans the chunks of one warp.
pub(crate) struct Planner<'a> {
    t: &'a Approx,
    src: [i32; 2],
    dst: [i32; 2],
    radius: i32,
    /// Target pixel positions of points that force grid sampling when they
    /// land in the target (GDAL: lon 0 at lat -89.9999 and 89.9999).
    special: Vec<(f64, f64)>,
    /// The source grid of `ComputeSourceWindowStartingFromSource`, forward
    /// transformed once.
    source_grid: std::cell::OnceCell<Vec<GridPoint>>,
    /// Bits per source and target cell, masks included (GDAL's
    /// `GetWorkingMemoryForWindow`).
    src_bits: f64,
    dst_bits: f64,
    /// For a global lon/lat source: its width in cells, and its height
    /// when it runs pole to pole (for [`Planner::round_trip`]).
    globe_width: Option<f64>,
    pole_rows: Option<f64>,
    /// Whether the forward probe round-trips its points (affine sources:
    /// proj4rs's antipode). A curvilinear source's edges can wrap onto
    /// each other, so its points go back elsewhere; GDAL checks none.
    check_round_trip: bool,
    /// Bytes a chunk's working memory may take before it is split
    /// (GDAL's `dfWarpMemoryLimit`, gdalwarp's `-wm`).
    pub memory_limit: f64,
    /// The kernel scale given for every chunk ([`WarpOptions::kernel_scale`]).
    pub kernel_scale: Option<[f64; 2]>,
}

impl<'a> Planner<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        t: &'a Approx,
        src: [i32; 2],
        dst: [i32; 2],
        radius: i32,
        special: Vec<(f64, f64)>,
        globe: Option<(f64, Option<f64>)>,
        dtype_bits: usize,
        nodata: bool,
        check_round_trip: bool,
    ) -> Self {
        let mask = if nodata { 1.0 } else { 0.0 };
        Self {
            t,
            src,
            dst,
            radius,
            special,
            source_grid: std::cell::OnceCell::new(),
            src_bits: dtype_bits as f64 + mask,
            dst_bits: dtype_bits as f64 + mask,
            globe_width: globe.map(|g| g.0),
            pole_rows: globe.and_then(|g| g.1),
            check_round_trip,
            memory_limit: MEMORY_LIMIT,
            kernel_scale: None,
        }
    }

    /// GDAL's `CollectChunkList` for the whole target, sorted top to
    /// bottom, then left to right. GDAL's in-memory target has one-row
    /// blocks, so its size optimisation never applies and chunks halve
    /// along their longer side.
    pub fn chunks(&self) -> Vec<Chunk> {
        let mut out = Vec::new();
        self.collect([0, 0], self.dst, &mut out);
        for c in out.iter_mut().filter(|c| !c.is_empty()) {
            c.scale = chunk_scale(self.t, c, self.kernel_scale);
        }
        out.sort_by(|a, b| (a.dst_off[1], a.dst_off[0]).cmp(&(b.dst_off[1], b.dst_off[0])));
        out
    }

    fn collect(&self, off: [i32; 2], size: [i32; 2], out: &mut Vec<Chunk>) {
        let w = self.source_window(off, size);
        let memory = (self.src_bits * w.size[0] as f64 * w.size[1] as f64
            + self.dst_bits * size[0] as f64 * size[1] as f64)
            / 8.0;
        let split = (memory > self.memory_limit && (size[0] > 2 || size[1] > 2))
            || (w.fill > 0.0 && w.fill < 0.5 && (size[0] > 100 || size[1] > 100));
        if split {
            if size[0] > size[1] {
                let a = size[0] / 2;
                self.collect(off, [a, size[1]], out);
                self.collect([off[0] + a, off[1]], [size[0] - a, size[1]], out);
            } else {
                let a = size[1] / 2;
                self.collect(off, [size[0], a], out);
                self.collect([off[0], off[1] + a], [size[0], size[1] - a], out);
            }
            return;
        }
        out.push(Chunk {
            dst_off: off,
            dst_size: size,
            src_off: w.off,
            src_size: w.size,
            extra: w.extra,
            scale: [1.0, 1.0],
        });
    }

    /// GDAL's `ComputeSourceWindowTransformPoints`: transform sample points
    /// of the target window back to the source, widening `b`
    /// (min x, min y, max x, max y, which carries over between calls as in
    /// GDAL). Returns (samples, failed).
    fn sample(
        &self,
        off: [i32; 2],
        size: [i32; 2],
        grid: bool,
        all: bool,
        b: &mut [f64; 4],
    ) -> (usize, usize) {
        let (ox, oy) = (off[0] as f64, off[1] as f64);
        let (sx, sy) = (size[0] as f64, size[1] as f64);
        let step = if all { 0.0 } else { 1.0 / (STEPS - 1) as f64 };
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let mut push = |x: f64, y: f64| {
            xs.push(x);
            ys.push(y);
        };
        if grid {
            if all {
                for iy in 0..=size[1] {
                    for ix in 0..=size[0] {
                        push(ox + ix as f64, oy + iy as f64);
                    }
                }
            } else {
                // GDAL uses the x size for both near-edge ratios.
                let ratio = |i: usize| {
                    if i == 0 {
                        0.5 / sx
                    } else if i <= STEPS {
                        (i - 1) as f64 * step
                    } else {
                        1.0 - 0.5 / sx
                    }
                };
                for iy in 0..STEPS + 2 {
                    for ix in 0..STEPS + 2 {
                        push(ratio(ix) * sx + ox, ratio(iy) * sy + oy);
                    }
                }
            }
        } else if all {
            for ix in 0..=size[0] {
                push(ox + ix as f64, oy);
                push(ox + ix as f64, oy + sy);
            }
            for iy in 1..size[1] {
                push(ox, oy + iy as f64);
                push(ox + sx, oy + iy as f64);
            }
        } else {
            let mut r = 0.0;
            while r <= 1.0 + step * 0.5 {
                push(r * sx + ox, oy);
                push(r * sx + ox, oy + sy);
                push(ox, r * sy + oy);
                push(sx + ox, r * sy + oy);
                r += step;
            }
        }
        let ok = self.t.transform(true, &mut xs, &mut ys);
        let mut failed = 0;
        for i in 0..xs.len() {
            if !ok[i] || xs[i].is_nan() || ys[i].is_nan() {
                failed += 1;
                continue;
            }
            b[0] = b[0].min(xs[i]);
            b[1] = b[1].min(ys[i]);
            b[2] = b[2].max(xs[i]);
            b[3] = b[3].max(ys[i]);
        }
        (xs.len(), failed)
    }

    /// GDAL's `ComputeSourceWindowStartingFromSource`.
    fn probe_source(&self, off: [i32; 2], size: [i32; 2], b: &mut [f64; 4]) {
        let grid = self.source_grid.get_or_init(|| {
            let [w, h] = self.src.map(|n| n as f64);
            let step = 1.0 / (STEPS - 1) as f64;
            let ratio = |i: usize, n: f64| {
                if i == 0 {
                    0.5 / n
                } else if i <= STEPS {
                    (i - 1) as f64 * step
                } else {
                    1.0 - 0.5 / n
                }
            };
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for iy in 0..STEPS + 2 {
                for ix in 0..STEPS + 2 {
                    xs.push(ratio(ix, w) * w);
                    ys.push(ratio(iy, h) * h);
                }
            }
            let (sx, sy) = (xs.clone(), ys.clone());
            let mut ok = self.t.transform(false, &mut xs, &mut ys);
            if self.check_round_trip {
                self.round_trip(&sx, &sy, &xs, &ys, &mut ok);
            }
            (0..xs.len())
                .map(|i| (sx[i], sy[i], xs[i], ys[i], ok[i]))
                .collect()
        });
        let (x0, y0) = (off[0] as f64, off[1] as f64);
        let (x1, y1) = (x0 + size[0] as f64, y0 + size[1] as f64);
        for &(sx, sy, dx, dy, ok) in grid {
            if ok && dx >= x0 && dx <= x1 && dy >= y0 && dy <= y1 {
                b[0] = b[0].min(sx);
                b[1] = b[1].min(sy);
                b[2] = b[2].max(sx);
                b[3] = b[3].max(sy);
            }
        }
    }

    /// Fail forward-transformed points that do not transform back to where
    /// they started (to within a source cell). PROJ fails a point it
    /// cannot project, such as the antipode of a polar stereographic
    /// projection; proj4rs can return its centre instead, which would put
    /// the opposite pole of the source inside the target. On the pole rows
    /// of a global lon/lat source any longitude goes back to the same
    /// point, so only the row is compared there.
    fn round_trip(&self, sx: &[f64], sy: &[f64], dx: &[f64], dy: &[f64], ok: &mut [bool]) {
        let (mut bx, mut by) = (dx.to_vec(), dy.to_vec());
        let back = self.t.transform(true, &mut bx, &mut by);
        for i in 0..ok.len() {
            if !ok[i] {
                continue;
            }
            let mut ex = (bx[i] - sx[i]).abs();
            if let Some(w) = self.globe_width {
                ex = ex.min(w - ex);
            }
            let pole = self.pole_rows.is_some_and(|h| sy[i] <= 0.0 || sy[i] >= h);
            ok[i] = back[i] && (by[i] - sy[i]).abs() <= 1.0 && (pole || ex <= 1.0);
        }
    }

    /// GDAL's `ComputeSourceWindow`. A window that cannot be computed
    /// (too few points transform) comes back empty, as GDAL then skips the
    /// chunk.
    fn source_window(&self, off: [i32; 2], size: [i32; 2]) -> SourceWindow {
        let mut grid = false;
        let mut all = false;
        {
            let (ox, oy) = (off[0] as f64, off[1] as f64);
            let (ex, ey) = (ox + size[0] as f64, oy + size[1] as f64);
            let mut cx = [ox, ex, ox, ex];
            let mut cy = [oy, oy, ey, ey];
            if self
                .t
                .transform(true, &mut cx, &mut cy)
                .iter()
                .any(|ok| !ok)
            {
                all = true;
            }
        }
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let (mut n, mut failed) = self.sample(off, size, grid, all, &mut b);
        let [dw, dh] = self.dst.map(|v| v as f64);
        if self
            .special
            .iter()
            .any(|&(x, y)| 0.0 <= x && dw >= x && 0.0 <= y && dh >= y)
        {
            grid = true;
            all = false;
            (n, failed) = self.sample(off, size, grid, all, &mut b);
        }
        if !grid && failed > 0 {
            grid = true;
            all = false;
            (n, failed) = self.sample(off, size, grid, all, &mut b);
        }
        if failed as i64 > n as i64 - 5 {
            return EMPTY;
        }
        if grid {
            self.probe_source(off, size, &mut b);
        }
        let [w, h] = self.src;
        let (wf, hf) = (w as f64, h as f64);
        if b[0] > wf || b[2] < 0.0 || b[1] > hf || b[3] < 0.0 {
            return EMPTY;
        }
        let snap = |v: f64| {
            let r = v.round();
            if (r - v).abs() < 1e-6 {
                r
            } else {
                v
            }
        };
        let [min_x, min_y, max_x, max_y] = b.map(snap);

        let r = self.radius;
        let mut xscale = (size[0] as f64 / (max_x - min_x)).max(1e-3);
        let mut yscale = (size[1] as f64 / (max_y - min_y)).max(1e-3);
        // GDAL pads by the estimated scale even under XSCALE/YSCALE, which
        // clips a kernel stretched further than that at the window's edge
        // (so tiles warped on their own differ there); pad for the given
        // scale too.
        if let Some([fx, fy]) = self.kernel_scale {
            xscale = xscale.min(fx.max(1e-3));
            yscale = yscale.min(fy.max(1e-3));
        }
        let mut xr = if xscale < 0.95 {
            (r as f64 / xscale).ceil() as i32
        } else {
            r
        };
        let mut yr = if yscale < 0.95 {
            (r as f64 / yscale).ceil() as i32
        } else {
            r
        };
        if failed > 0 {
            xr += 10;
            yr += 10;
        }

        let min_xc = min_x.max(0.0) as i32;
        let min_yc = min_y.max(0.0) as i32;
        let max_xc = max_x.ceil().min(wf) as i32;
        let max_yc = max_y.ceil().min(hf) as i32;
        let raw_x = ((w - min_xc) as f64).min(max_x - min_x).max(0.0);
        let raw_y = ((h - min_yc) as f64).min(max_y - min_y).max(0.0);

        let axis = |min_c: i32, max_c: i32, n: i32, radius: i32| {
            if (max_c - min_c) as f64 > 0.9 * n as f64 {
                (0, n)
            } else {
                let o = (min_c - radius).min(n).max(0);
                (o, (max_c - o + radius).min(n - o).max(0))
            }
        };
        let (xo, xs) = axis(min_xc, max_xc, w, xr);
        let (yo, ys) = axis(min_yc, max_yc, h, yr);
        let fill = xs as f64 * ys as f64
            / ((max_x - min_x + 2.0 * xr as f64) * (max_y - min_y + 2.0 * yr as f64)).max(1.0);
        SourceWindow {
            off: [xo, yo],
            size: [xs, ys],
            extra: [xs as f64 - raw_x, ys as f64 - raw_y],
            fill,
        }
    }
}

/// GDAL 3.12's kernel scale for a chunk (`PerformWarp`), including its
/// correction of an x scale far below the y scale from sampled local
/// scales, or the scale given for every chunk.
fn chunk_scale(t: &Approx, c: &Chunk, fixed: Option<[f64; 2]>) -> [f64; 2] {
    if let Some(s) = fixed {
        return s;
    }
    let [dw, dh] = c.dst_size;
    let [mut xs, ys] = Scale::ratios(
        [dw as usize, dh as usize],
        [c.src_size[0] as usize, c.src_size[1] as usize],
        c.extra,
    );
    if ys / xs > 100.0 {
        let (px, py) = (dw.min(10), dh.min(10));
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for iy in 0..py {
            for ix in 0..px {
                // As GDAL: positions within the chunk, without its offset.
                let fx = if px == 1 {
                    0.0
                } else {
                    ix as f64 * dw as f64 / (px - 1) as f64
                };
                let fy = if py == 1 {
                    0.0
                } else {
                    iy as f64 * dh as f64 / (py - 1) as f64
                };
                x.push(fx);
                y.push(fy);
                x.push(if ix == px - 1 { fx - 1.0 } else { fx + 1.0 });
                y.push(fy);
                x.push(fx);
                y.push(if iy == py - 1 { fy - 1.0 } else { fy + 1.0 });
            }
        }
        let ok = t.transform(true, &mut x, &mut y);
        let mut scales: Vec<f64> = (0..x.len())
            .step_by(3)
            .filter(|&i| ok[i] && ok[i + 1] && ok[i + 2])
            .map(|i| 1.0 / (x[i + 1] - x[i]).abs().max((x[i + 2] - x[i]).abs()))
            .collect();
        scales.sort_by(|a, b| a.total_cmp(b));
        if let Some(&max) = scales.last() {
            let kept: Vec<f64> = scales.into_iter().filter(|&s| s > max / 10.0).collect();
            if !kept.is_empty() {
                xs = kept.iter().sum::<f64>() / kept.len() as f64;
            }
        }
    }
    [xs, ys]
}

/// Types GDAL's warper resamples without masks when no nodata is set
/// (its specialised `NoMasksOrDstDensityOnly` kernels); every other type
/// goes through its general real-valued path.
fn has_plain_path<T: Sample>() -> bool {
    use std::any::TypeId;
    let t = TypeId::of::<T>();
    t == TypeId::of::<u8>()
        || t == TypeId::of::<i16>()
        || t == TypeId::of::<u16>()
        || t == TypeId::of::<f32>()
}

/// One band of the chunked warp. `src` is the source region
/// `region_off`/`region_size` (covering every chunk's window); `out` is
/// the whole target, already filled with nodata (or zero).
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_band<T: Sample>(
    t: &Approx,
    chunks: &[Chunk],
    method: Method,
    src: &[T],
    region_off: [i32; 2],
    region_size: [i32; 2],
    out: &mut [T],
    dst_width: usize,
    nodata: Option<f64>,
) {
    let masked = nodata.is_some() || !has_plain_path::<T>();
    // GDAL's target validity mask: the target starts as nodata, so every
    // cell is invalid until written.
    let mut dst_valid: Option<Vec<bool>> = nodata.map(|_| vec![false; out.len()]);
    let nd_src = nodata.and_then(T::from_nodata);

    for c in chunks.iter().filter(|c| !c.is_empty()) {
        let [sw, sh] = c.src_size.map(|v| v as usize);
        let (rx, ry) = (
            (c.src_off[0] - region_off[0]) as usize,
            (c.src_off[1] - region_off[1]) as usize,
        );
        let mut data = Vec::with_capacity(sw * sh);
        for row in 0..sh {
            let start = (ry + row) * region_size[0] as usize + rx;
            data.extend_from_slice(&src[start..start + sw]);
        }
        let valid: Option<Vec<bool>> = nd_src.and_then(|nd| {
            let v: Vec<bool> = data.iter().map(|&x| !x.is_nodata(nd)).collect();
            (!v.iter().all(|&b| b)).then_some(v)
        });
        let win = Window {
            data: &data,
            xsize: sw,
            ysize: sh,
            valid: valid.as_deref(),
        };
        let [sx, sy] = c.scale;
        let mut kernel = Kernel::new(method, Scale::new(method, sx, sy));
        let corner_fails = std::cell::OnceCell::new();

        let [dx0, dy0] = c.dst_off;
        let [cw, ch] = c.dst_size;
        let (sx0, sy0) = (c.src_off[0] as f64, c.src_off[1] as f64);
        for iy in 0..ch {
            let fy = iy as f64 + 0.5 + dy0 as f64;
            let mut xs: Vec<f64> = (0..cw).map(|ix| ix as f64 + 0.5 + dx0 as f64).collect();
            let mut ys = vec![fy; cw as usize];
            let mut ok = t.transform(true, &mut xs, &mut ys);
            for ix in 0..cw as usize {
                let Some(mut off) = src_offset(t, c, ix, fy, &mut xs, &mut ys, &mut ok) else {
                    continue;
                };
                let (x, y) = (xs[ix] - sx0, ys[ix] - sy0);
                let dst = (dy0 as usize + iy as usize) * dst_width + dx0 as usize + ix;
                if !masked {
                    out[dst] = kernel.value(&win, off, x, y);
                    continue;
                }
                if !win.is_valid(off) {
                    if !*corner_fails.get_or_init(|| one_corner_fails(t, c)) {
                        continue;
                    }
                    match adjust_on_edge(t, c, &win, off) {
                        Some(o) => off = o,
                        None => continue,
                    }
                }
                let (density, real) = kernel.density(&win, off, x, y);
                if density < BAND_DENSITY_THRESHOLD {
                    continue;
                }
                write_density(out, dst_valid.as_deref(), dst, density, real, nodata);
                if let Some(v) = dst_valid.as_mut() {
                    v[dst] = true;
                }
            }
        }
    }
}

/// One band of the chunked warp through rwarp: each chunk's source window
/// and target cells go to one `warp` call, at the chunk's kernel scale.
/// Arguments as [`warp_band`]'s; `out` is already filled with `nodata`
/// (or zero), which is what rwarp writes where a cell gets no value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_band_rwarp<T: rwarp::warp::Sample>(
    t: &Approx,
    chunks: &[Chunk],
    opts: &WarpOptions,
    src: &[T],
    region_off: [i32; 2],
    region_size: [i32; 2],
    out: &mut [T],
    dst_width: usize,
    nodata: Option<T>,
) {
    for c in chunks.iter().filter(|c| !c.is_empty()) {
        let [sw, sh] = c.src_size.map(|v| v as usize);
        let (rx, ry) = (
            (c.src_off[0] - region_off[0]) as usize,
            (c.src_off[1] - region_off[1]) as usize,
        );
        let mut data = Vec::with_capacity(sw * sh);
        for row in 0..sh {
            let start = (ry + row) * region_size[0] as usize + rx;
            data.extend_from_slice(&src[start..start + sw]);
        }
        let buf = SourceBuffer {
            data: &data,
            ncol: sw,
            nrow: sh,
            nbands: 1,
            col_off: c.src_off[0] as usize,
            row_off: c.src_off[1] as usize,
        };
        let [x, y] = c.scale;
        let [dx0, dy0] = c.dst_off.map(|v| v as usize);
        let [cw, ch] = c.dst_size.map(|v| v as usize);
        let mut o = opts.rwarp(KernelScaling::Fixed { x, y }, nodata);
        o.dst_col_off = dx0;
        o.dst_row_off = dy0;
        let tile = rwarp::warp::warp(t, &buf, cw, ch, &o);
        for (iy, row) in tile.chunks_exact(cw).enumerate() {
            let at = (dy0 + iy) * dst_width + dx0;
            out[at..at + cw].copy_from_slice(row);
        }
    }
}

/// GDAL's `GWKCheckAndComputeSrcOffsets`: the source cell (offset in the
/// chunk window) for target column `ix`, retrying a position just outside
/// the window with the exact transformer.
fn src_offset(
    t: &Approx,
    c: &Chunk,
    ix: usize,
    fy: f64,
    xs: &mut [f64],
    ys: &mut [f64],
    ok: &mut [bool],
) -> Option<usize> {
    let (ox, oy) = (c.src_off[0] as f64, c.src_off[1] as f64);
    let (w, h) = (c.src_size[0] as f64, c.src_size[1] as f64);
    for attempt in 0..2 {
        if attempt == 1 {
            let mut px = [ix as f64 + 0.5 + c.dst_off[0] as f64];
            let mut py = [fy];
            ok[ix] = t.transform(true, &mut px, &mut py)[0];
            xs[ix] = px[0];
            ys[ix] = py[0];
        }
        if !ok[ix] || xs[ix].is_nan() || ys[ix].is_nan() {
            return None;
        }
        let (x, y) = (xs[ix], ys[ix]);
        let retry = attempt == 0;
        if x < ox {
            if retry && x > ox - 1.0 {
                continue;
            }
            return None;
        }
        if y < oy {
            if retry && y > oy - 1.0 {
                continue;
            }
            return None;
        }
        if x + 1e-10 > w + ox {
            if retry && x < w + ox + 1.0 {
                continue;
            }
            return None;
        }
        if y + 1e-10 > h + oy {
            if retry && y < h + oy + 1.0 {
                continue;
            }
            return None;
        }
        break;
    }
    let (sw, sh) = (c.src_size[0], c.src_size[1]);
    let mut cx = (xs[ix] + 1e-10) as i32 - c.src_off[0];
    let mut cy = (ys[ix] + 1e-10) as i32 - c.src_off[1];
    if cx == sw {
        cx -= 1;
    }
    if cy == sh {
        cy -= 1;
    }
    Some(cx as usize + cy as usize * sw as usize)
}

/// GDAL's `GWKOneSourceCornerFailsToReproject`.
fn one_corner_fails(t: &Approx, c: &Chunk) -> bool {
    for iy in 0..=1 {
        for ix in 0..=1 {
            let mut x = [(c.src_off[0] + ix * c.src_size[0]) as f64];
            let mut y = [(c.src_off[1] + iy * c.src_size[1]) as f64];
            if !t.transform(false, &mut x, &mut y)[0] {
                return true;
            }
        }
    }
    false
}

/// GDAL's `GWKAdjustSrcOffsetOnEdge`: at the edge of the projection's
/// domain, use a valid neighbour of an invalid source cell.
fn adjust_on_edge<T: Sample>(t: &Approx, c: &Chunk, win: &Window<T>, off: usize) -> Option<usize> {
    let w = win.xsize;
    let (cx, cy) = (off % w, off / w);
    let fwd = |dx: usize, dy: usize| {
        let mut x = [(c.src_off[0] as usize + cx + dx) as f64];
        let mut y = [(c.src_off[1] as usize + cy + dy) as f64];
        t.transform(false, &mut x, &mut y)[0]
    };
    if fwd(0, 0) && fwd(0, 1) && fwd(1, 0) {
        return None;
    }
    if cx + 1 < w && win.is_valid(off + 1) {
        Some(off + 1)
    } else if cy + 1 < win.ysize && win.is_valid(off + w) {
        Some(off + w)
    } else if cx > 0 && win.is_valid(off - 1) {
        Some(off - 1)
    } else if cy > 0 && win.is_valid(off - w) {
        Some(off - w)
    } else {
        None
    }
}
