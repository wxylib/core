//! The chunked warp: GDAL's warp operation, chunk by chunk, with rwarp's
//! kernels (design doc 03, S5). Each chunk's source window and target
//! cells go to one `rwarp::warp::warp` call, at the kernel scale GDAL
//! gives the chunk.
//!
//! GDAL splits the target into chunks (`CollectChunkList`), computes each
//! chunk's source window (`ComputeSourceWindow`) and, from the window's
//! size, how much to stretch the kernel (`PerformWarp`). The chunk
//! boundaries matter to the output: they set the kernel scale, the window
//! edges where kernels are clipped, and the scanline extents the
//! approximate transformer interpolates over. This module ports the
//! planning steps so the output matches GDAL's (3.14, the oracles'
//! version); the kernels, including GDAL's source offset checks, are
//! rwarp's.
//!
//! Not ported: GDAL's retry with `CHECK_WITH_INVERT_PROJ` when a window
//! comes back with wild values (a PROJ behaviour the proj4rs backend has no
//! switch for), destination alpha, cutlines and source masks other than
//! nodata.

use rwarp::warp::{KernelScaling, SourceBuffer};
use rwarp::Transformer;

use crate::{Approx, ResampleAlg, WarpOptions};

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
    fill: f64,
}

const EMPTY: SourceWindow = SourceWindow {
    off: [0, 0],
    size: [0, 0],
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
            fill,
        }
    }
}

/// GDAL's kernel scale for a chunk (`GDALWarpKernel::PerformWarp`, GDAL
/// 3.13 and later), or the scale given for every chunk. As GDAL does, the
/// unit squares are sampled over `0..size` of the chunk without its
/// offset, so every chunk is estimated at the target's top-left (rwarp's
/// `estimate_scale` with no offset); the oracles need GDAL's answer.
fn chunk_scale(t: &Approx, c: &Chunk, fixed: Option<[f64; 2]>) -> [f64; 2] {
    if let Some(s) = fixed {
        return s;
    }
    let [w, h] = c.dst_size.map(|v| v as usize);
    let (x, y) = rwarp::warp::estimate_scale(t, w, h, 0, 0);
    [x, y]
}

/// GDAL's retry in `GWKCheckAndComputeSrcOffsets`: a cell centre that the
/// approximate transformer puts less than one source cell outside the
/// chunk's window is transformed again on its own, which is exact. rwarp
/// uses the transformer it is given, so the retry wraps it here, where the
/// window is known.
struct Retry<'a> {
    t: &'a Approx,
    off: [f64; 2],
    size: [f64; 2],
}

impl Transformer for Retry<'_> {
    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (x0, y0) = (x.to_vec(), y.to_vec());
        let mut ok = self.t.transform(dst_to_src, x, y);
        if !dst_to_src {
            return ok;
        }
        // just outside [off, off + size), as GDAL tests each side
        let near = |v: f64, o: f64, n: f64| {
            (v < o && v > o - 1.0) || (v + 1e-10 > o + n && v < o + n + 1.0)
        };
        for i in 0..ok.len() {
            if ok[i]
                && (near(x[i], self.off[0], self.size[0]) || near(y[i], self.off[1], self.size[1]))
            {
                let (mut a, mut b) = ([x0[i]], [y0[i]]);
                ok[i] = self.t.transform(true, &mut a, &mut b)[0];
                x[i] = a[0];
                y[i] = b[0];
            }
        }
        ok
    }
}

/// One band of the chunked warp. `src` is the source region
/// `region_off`/`region_size` (covering every chunk's window); `out` is
/// the whole target, already filled with `nodata` (or zero), which is
/// what rwarp writes where a cell gets no value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_band<T: rwarp::warp::Sample>(
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
        let tile = match opts.resampling {
            // footprints from cell corners, which GDAL does not retry
            ResampleAlg::Average | ResampleAlg::Mode => rwarp::warp::warp(t, &buf, cw, ch, &o),
            _ => {
                let retry = Retry {
                    t,
                    off: c.src_off.map(f64::from),
                    size: c.src_size.map(f64::from),
                };
                rwarp::warp::warp(&retry, &buf, cw, ch, &o)
            }
        };
        for (iy, row) in tile.chunks_exact(cw).enumerate() {
            let at = (dy0 + iy) * dst_width + dx0;
            out[at..at + cw].copy_from_slice(row);
        }
    }
}
