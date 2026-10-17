//! # wxy-resample
//!
//! Resampling kernels for warping: given a window of source cells and a
//! fractional position in it, the value there. The kernels are ports of
//! GDAL's warp kernel (`alg/gdalwarpkernel.cpp`, GDAL 3.12), kept to its
//! arithmetic order so that wxylib warps match GDAL's cell for cell,
//! including its kernel scaling when the target is coarser than the source
//! (design doc 03, S5).
//!
//! No transforms and no I/O: a warp driver (wxy-warp) picks the source
//! window and the positions, and calls one kernel per target cell. Two
//! families mirror GDAL's two paths:
//!
//! - [`Kernel::value`]: no masks anywhere (GDAL's
//!   `GWKResampleNoMasksOrDstDensityOnlyThread`), the result already
//!   rounded to the cell type;
//! - [`Kernel::density`]: with a validity mask or a target nodata (GDAL's
//!   `GWKRealCase`), giving a density and a value for
//!   [`write_density`] to apply.
//!
//! Positions are in the window's pixel space: `(0.5, 0.5)` is the centre
//! of its first cell.

mod sample;
pub use sample::Sample;

/// Resampling methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Nearest,
    Bilinear,
    Cubic,
}

impl Method {
    /// The kernel's radius in source cells at 1:1 (GDAL's
    /// `anGWKFilterRadius`).
    pub fn radius(self) -> i32 {
        match self {
            Method::Nearest => 0,
            Method::Bilinear => 1,
            Method::Cubic => 2,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Method::Nearest => 0.0,
            Method::Bilinear => bilinear(x),
            Method::Cubic => cubic(x),
        }
    }

    fn weight4(self, v: &mut [f64]) -> f64 {
        for e in v.iter_mut().take(4) {
            *e = self.weight(*e);
        }
        v[0] + v[1] + v[2] + v[3]
    }
}

fn bilinear(x: f64) -> f64 {
    let a = x.abs();
    if a <= 1.0 {
        1.0 - a
    } else {
        0.0
    }
}

/// Catmull-Rom (cubic convolution with a = -0.5).
fn cubic(x: f64) -> f64 {
    let a = x.abs();
    if a <= 1.0 {
        let x2 = x * x;
        x2 * (1.5 * a - 2.5) + 1.0
    } else if a <= 2.0 {
        let x2 = x * x;
        x2 * (-0.5 * a + 2.5) - 4.0 * a + 2.0
    } else {
        0.0
    }
}

/// How the kernel is stretched for one warp chunk, from the ratio of
/// target to source size (GDAL's `PerformWarp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale {
    pub x: f64,
    pub y: f64,
    pub x_radius: i32,
    pub y_radius: i32,
    /// First kernel tap relative to the cell left of (above) the position.
    pub x_first: i32,
    pub y_first: i32,
}

impl Scale {
    /// The raw scales for a chunk of `dst` target cells reading `src`
    /// source cells, `extra` of which are kernel margin (GDAL's
    /// `dfSrcXExtraSize`), snapped to 1 and to whole reciprocals as GDAL
    /// does. GDAL's correction for a suspiciously small x scale (y / x over
    /// 100) needs the transformer, so it is left to the caller.
    pub fn ratios(dst: [usize; 2], src: [usize; 2], extra: [f64; 2]) -> [f64; 2] {
        let one = |d: usize, s: usize, e: f64| {
            let (d, sf) = (d as f64, s as f64);
            let mut k = d / (sf - e);
            if s >= d as usize && sf <= d + e {
                k = 1.0;
            }
            if k < 1.0 {
                let r = 1.0 / k;
                let n = (r + 0.5) as i32;
                if (r - n as f64).abs() < 0.05 {
                    k = 1.0 / n as f64;
                }
            }
            k
        };
        [one(dst[0], src[0], extra[0]), one(dst[1], src[1], extra[1])]
    }

    pub fn new(method: Method, x: f64, y: f64) -> Self {
        let r = method.radius();
        let radius = |k: f64| {
            if k < 1.0 {
                (r as f64 / k).ceil() as i32
            } else {
                r
            }
        };
        let (x_radius, y_radius) = (radius(x), radius(y));
        let parity = (r + 1) % 2;
        Self {
            x,
            y,
            x_radius,
            y_radius,
            x_first: parity - x_radius,
            y_first: parity - y_radius,
        }
    }

    /// Whether GDAL's 4-sample (unscaled 2x2 or 4x4) formulas apply.
    pub fn four_samples(&self) -> bool {
        self.x >= 0.95 && self.y >= 0.95
    }
}

/// A window of source cells, row-major, with optional validity (GDAL's
/// unified source validity mask; `None` when every cell is valid).
#[derive(Debug, Clone, Copy)]
pub struct Window<'a, T> {
    pub data: &'a [T],
    pub xsize: usize,
    pub ysize: usize,
    pub valid: Option<&'a [bool]>,
}

/// GDAL's density thresholds.
const SRC_DENSITY_THRESHOLD: f64 = 0.000000001;
pub const BAND_DENSITY_THRESHOLD: f64 = 0.0000000001;

impl<T: Sample> Window<'_, T> {
    fn at(&self, off: isize) -> f64 {
        self.data[off as usize].to_f64()
    }

    /// GDAL's `GWKGetPixelRow`: `2 * half` cells from `off`, with
    /// densities when asked. Cells past the end of the window (GDAL reads
    /// its arrays' spare elements there) count as invalid zeros. False
    /// when no cell in the run is valid.
    fn row(&self, off: isize, half: usize, density: Option<&mut [f64]>, real: &mut [f64]) -> bool {
        let n = 2 * half;
        let len = self.data.len() as isize;
        for (i, r) in real.iter_mut().take(n).enumerate() {
            let o = off + i as isize;
            *r = if o >= 0 && o < len { self.at(o) } else { 0.0 };
        }
        let Some(density) = density else {
            return true;
        };
        let mut any = false;
        for (i, d) in density.iter_mut().take(n).enumerate() {
            let o = off + i as isize;
            let ok = o >= 0 && o < len && self.valid.is_none_or(|v| v[o as usize]);
            *d = if ok { 1.0 } else { 0.0 };
            any |= ok;
        }
        any
    }

    /// GDAL's `GWKGetPixelValueReal`: density and value of one cell.
    pub fn cell(&self, off: usize) -> (f64, f64) {
        if self.valid.is_some_and(|v| !v[off]) {
            return (0.0, 0.0);
        }
        (1.0, self.data[off].to_f64())
    }

    pub fn is_valid(&self, off: usize) -> bool {
        self.valid.is_none_or(|v| v[off])
    }
}

/// A kernel for one chunk: method, scale and scratch weights.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub method: Method,
    pub scale: Scale,
    wx: Vec<f64>,
    wy: Vec<f64>,
    calc_x: Vec<bool>,
    dens: Vec<f64>,
    real: Vec<f64>,
}

impl Kernel {
    pub fn new(method: Method, scale: Scale) -> Self {
        let nx = 2 * (scale.x_radius as usize + 1) + 4;
        let ny = 2 * (scale.y_radius as usize + 1) + 4;
        Self {
            method,
            scale,
            wx: vec![0.0; nx],
            wy: vec![0.0; ny],
            calc_x: vec![false; nx],
            dens: vec![0.0; nx],
            real: vec![0.0; nx],
        }
    }

    /// The value at `(x, y)` with no masks in play, rounded to the cell
    /// type. `cell` is the offset of the cell `(x, y)` falls in (GDAL's
    /// `iSrcOffset`, which nearest neighbour reads).
    pub fn value<T: Sample>(&mut self, src: &Window<T>, cell: usize, x: f64, y: f64) -> T {
        match self.method {
            Method::Nearest => src.data[cell],
            Method::Bilinear if self.scale.four_samples() => bilinear4_plain(src, x, y),
            Method::Cubic if self.scale.four_samples() => cubic4_plain(src, x, y),
            _ => self.scaled_plain(src, x, y),
        }
    }

    /// GDAL's `GWKComputeWeights`.
    fn weights(&mut self, i_min: i32, i_max: i32, dx: f64, j_min: i32, j_max: i32, dy: f64) -> f64 {
        let (sx, sy) = (self.scale.x.min(1.0), self.scale.y.min(1.0));
        let m = self.method;
        let fill = |w: &mut [f64], lo: i32, hi: i32, d: f64, s: f64| {
            let mut acc = f64::MIN_POSITIVE;
            let (mut i, mut c) = (lo, 0usize);
            while i + 2 < hi {
                w[c] = (i as f64 - d) * s;
                w[c + 1] = w[c] + s;
                w[c + 2] = w[c + 1] + s;
                w[c + 3] = w[c + 2] + s;
                acc += m.weight4(&mut w[c..c + 4]);
                i += 4;
                c += 4;
            }
            while i <= hi {
                let v = m.weight((i as f64 - d) * s);
                w[c] = v;
                acc += v;
                i += 1;
                c += 1;
            }
            acc
        };
        let h = fill(&mut self.wx, i_min, i_max, dx, sx);
        let v = fill(&mut self.wy, j_min, j_max, dy, sy);
        1.0 / (h * v)
    }

    /// GDAL's `GWKResampleNoMasks_SSE2_T`, in the lane order of its
    /// SSE2 build (no AVX): four rows at a time, four taps per lane group,
    /// lanes summed as `(0 + 2) + (1 + 3)`.
    fn scaled_plain<T: Sample>(&mut self, src: &Window<T>, x: f64, y: f64) -> T {
        let (xs, ys) = (src.xsize as i32, src.ysize as i32);
        let ix = (x - 0.5).floor() as i32;
        let iy = (y - 0.5).floor() as i32;
        let (rx, ry) = (self.scale.x_radius, self.scale.y_radius);
        if ix >= xs || iy >= ys || rx > xs || ry > ys {
            return bilinear4_plain(src, x, y);
        }
        let dx = x - 0.5 - ix as f64;
        let dy = y - 0.5 - iy as f64;
        let mut i_min = 1 - rx;
        if ix + i_min < 0 {
            i_min = -ix;
        }
        let mut i_max = rx;
        if ix + i_max >= xs - 1 {
            i_max = xs - 1 - ix;
        }
        let mut j_min = 1 - ry;
        if iy + j_min < 0 {
            j_min = -iy;
        }
        let mut j_max = ry;
        if iy + j_max >= ys - 1 {
            j_max = ys - 1 - iy;
        }
        let inv = self.weights(i_min, i_max, dx, j_min, j_max, dy);
        let (wx, wy) = (&self.wx, &self.wy);
        let stride = src.xsize as isize;
        let origin = ix as isize + iy as isize * stride;
        let p = |o: isize| src.at(o);

        let mut acc = 0.0;
        let mut j = j_min;
        let mut jc = 0usize;
        let mut row = origin + j_min as isize * stride;
        while j + 2 < j_max {
            let mut lanes = [[0.0f64; 4]; 4];
            let mut i = i_min;
            let mut ic = 0usize;
            while i + 2 < i_max {
                for (r, lane) in lanes.iter_mut().enumerate() {
                    let o = row + r as isize * stride + i as isize;
                    for (k, l) in lane.iter_mut().enumerate() {
                        *l += p(o + k as isize) * wx[ic + k];
                    }
                }
                i += 4;
                ic += 4;
            }
            if i < i_max {
                for (r, lane) in lanes.iter_mut().enumerate() {
                    let o = row + r as isize * stride + i as isize;
                    lane[0] += p(o) * wx[ic];
                    lane[1] += p(o + 1) * wx[ic + 1];
                }
                i += 2;
                ic += 2;
            }
            let mut local = lanes.map(|l| (l[0] + l[2]) + (l[1] + l[3]));
            if i == i_max {
                for (r, s) in local.iter_mut().enumerate() {
                    *s += p(row + r as isize * stride + i as isize) * wx[ic];
                }
            }
            for (r, s) in local.iter().enumerate() {
                acc += wy[jc + r] * s;
            }
            j += 4;
            jc += 4;
            row += 4 * stride;
        }
        while j <= j_max {
            let mut lane = [0.0f64; 4];
            let mut i = i_min;
            let mut ic = 0usize;
            while i + 2 < i_max {
                let o = row + i as isize;
                for (k, l) in lane.iter_mut().enumerate() {
                    *l += p(o + k as isize) * wx[ic + k];
                }
                i += 4;
                ic += 4;
            }
            let mut local = (lane[0] + lane[2]) + (lane[1] + lane[3]);
            if i < i_max {
                local += p(row + i as isize) * wx[ic];
                local += p(row + i as isize + 1) * wx[ic + 1];
                i += 2;
                ic += 2;
            }
            if i == i_max {
                local += p(row + i as isize) * wx[ic];
            }
            acc += wy[jc] * local;
            j += 1;
            jc += 1;
            row += stride;
        }
        T::clamp_from(acc * inv)
    }

    /// Density and value at `(x, y)` when masks are in play (GDAL's
    /// `GWKRealCase` kernels). A density below
    /// [`BAND_DENSITY_THRESHOLD`] means nothing valid was found.
    ///
    /// Nearest neighbour, and any method on a window one cell wide or
    /// high, reads `cell` alone, as GDAL does.
    pub fn density<T: Sample>(
        &mut self,
        src: &Window<T>,
        cell: usize,
        x: f64,
        y: f64,
    ) -> (f64, f64) {
        if self.method == Method::Nearest || src.xsize == 1 || src.ysize == 1 {
            return src.cell(cell);
        }
        match self.method {
            Method::Bilinear if self.scale.four_samples() => bilinear4(src, x, y),
            Method::Cubic if self.scale.four_samples() => cubic4(src, x, y),
            _ => self.general(src, x, y),
        }
    }

    /// GDAL's `GWKResample`: every tap weighted, invalid taps skipped, the
    /// result normalised by the weight found.
    fn general<T: Sample>(&mut self, src: &Window<T>, x: f64, y: f64) -> (f64, f64) {
        let (xs, ys) = (src.xsize as i32, src.ysize as i32);
        let stride = src.xsize as isize;
        let ix = (x - 0.5).floor() as i32;
        let iy = (y - 0.5).floor() as i32;
        let origin = ix as isize + iy as isize * stride;
        let dx = x - 0.5 - ix as f64;
        let dy = y - 0.5 - iy as f64;
        let (sx, sy) = (self.scale.x, self.scale.y);
        let m = self.method;
        let masked = src.valid.is_some();
        self.calc_x.iter_mut().for_each(|c| *c = false);

        let mut j = self.scale.y_first;
        let mut j_max = self.scale.y_radius;
        if iy + j < 0 {
            j = -iy;
        }
        if iy + j_max >= ys {
            j_max = ys - iy - 1;
        }
        let mut i_min = self.scale.x_first;
        let mut i_max = self.scale.x_radius;
        if ix + i_min < 0 {
            i_min = -ix;
        }
        if ix + i_max >= xs {
            i_max = xs - ix - 1;
        }
        let n = (i_max - i_min + 2).max(0) as usize / 2;
        if 2 * n > self.real.len() {
            self.real.resize(2 * n, 0.0);
            self.dens.resize(2 * n, 0.0);
            self.wx.resize(2 * n, 0.0);
            self.calc_x.resize(2 * n, false);
        }

        let (mut a_real, mut a_dens, mut a_weight) = (0.0, 0.0, 0.0);
        let mut row = origin + (j as isize - 1) * stride + i_min as isize;
        while j <= j_max {
            row += stride;
            let dens = if masked {
                Some(&mut self.dens[..])
            } else {
                None
            };
            if !src.row(row, n, dens, &mut self.real) {
                j += 1;
                continue;
            }
            let w1 = if sy < 1.0 {
                m.weight((j as f64 - dy) * sy)
            } else {
                m.weight(j as f64 - dy)
            };
            let (mut l_real, mut l_dens, mut l_weight) = (0.0, 0.0, 0.0);
            for i in i_min..=i_max {
                let k = (i - i_min) as usize;
                if masked && self.dens[k] < SRC_DENSITY_THRESHOLD {
                    continue;
                }
                let w2 = if self.calc_x[k] {
                    self.wx[k]
                } else {
                    let w = if sx < 1.0 {
                        m.weight((i as f64 - dx) * sx)
                    } else {
                        m.weight(i as f64 - dx)
                    };
                    self.wx[k] = w;
                    self.calc_x[k] = true;
                    w
                };
                l_real += self.real[k] * w2;
                if masked {
                    l_dens += self.dens[k] * w2;
                }
                l_weight += w2;
            }
            a_real += l_real * w1;
            a_dens += l_dens * w1;
            a_weight += l_weight * w1;
            j += 1;
        }
        if a_weight < 0.000001 || (masked && a_dens < 0.000001) {
            return (0.0, 0.0);
        }
        if !(0.99999..=1.00001).contains(&a_weight) {
            let d = if masked { a_dens / a_weight } else { 1.0 };
            (d, a_real / a_weight)
        } else {
            (if masked { a_dens } else { 1.0 }, a_real)
        }
    }
}

/// GDAL's `GWKBilinearResampleNoMasks4SampleT`.
fn bilinear4_plain<T: Sample>(src: &Window<T>, x: f64, y: f64) -> T {
    let (xs, ys) = (src.xsize as i32, src.ysize as i32);
    let stride = src.xsize as isize;
    let ix = (x - 0.5).floor() as i32;
    let iy = (y - 0.5).floor() as i32;
    let off = ix as isize + iy as isize * stride;
    let rx = 1.5 - (x - ix as f64);
    let ry = 1.5 - (y - iy as f64);
    let p = |o: isize| src.at(o);
    if ix >= 0 && ix + 1 < xs && iy >= 0 && iy + 1 < ys {
        let acc = (p(off) * rx + p(off + 1) * (1.0 - rx)) * ry
            + (p(off + stride) * rx + p(off + 1 + stride) * (1.0 - rx)) * (1.0 - ry);
        return T::round_from(acc);
    }
    let (mut div, mut acc) = (0.0, 0.0);
    let inside = |cx: i32, cy: i32| cx >= 0 && cx < xs && cy >= 0 && cy < ys;
    if inside(ix, iy) {
        let m = rx * ry;
        div += m;
        acc += p(off) * m;
    }
    if inside(ix + 1, iy) {
        let m = (1.0 - rx) * ry;
        div += m;
        acc += p(off + 1) * m;
    }
    if inside(ix + 1, iy + 1) {
        let m = (1.0 - rx) * (1.0 - ry);
        div += m;
        acc += p(off + 1 + stride) * m;
    }
    if inside(ix, iy + 1) {
        let m = rx * (1.0 - ry);
        div += m;
        acc += p(off + stride) * m;
    }
    if div < 0.00001 {
        return T::default();
    }
    T::round_from(if div == 1.0 { acc } else { acc / div })
}

/// GDAL's `GWKCubicComputeWeights`.
fn cubic_coeffs(x: f64) -> [f64; 4] {
    let half_x = 0.5 * x;
    let three_x = 3.0 * x;
    let half_x2 = half_x * x;
    [
        half_x * (-1.0 + x * (2.0 - x)),
        1.0 + half_x2 * (-5.0 + three_x),
        half_x * (1.0 + x * (4.0 - three_x)),
        half_x2 * (-1.0 + x),
    ]
}

fn convol4(c: &[f64; 4], v: [f64; 4]) -> f64 {
    c[0] * v[0] + c[1] * v[1] + c[2] * v[2] + c[3] * v[3]
}

/// GDAL's `GWKCubicResampleNoMasks4SampleT`.
fn cubic4_plain<T: Sample>(src: &Window<T>, x: f64, y: f64) -> T {
    let (xs, ys) = (src.xsize as i32, src.ysize as i32);
    let stride = src.xsize as isize;
    let ix = (x - 0.5) as i32;
    let iy = (y - 0.5) as i32;
    let off = ix as isize + iy as isize * stride;
    let dx = x - 0.5 - ix as f64;
    let dy = y - 0.5 - iy as f64;
    let dy2 = dy * dy;
    let dy3 = dy2 * dy;
    if ix - 1 < 0 || ix + 2 >= xs || iy - 1 < 0 || iy + 2 >= ys {
        return bilinear4_plain(src, x, y);
    }
    let c = cubic_coeffs(dx);
    let mut v = [0.0; 4];
    for (r, e) in v.iter_mut().enumerate() {
        let o = off + (r as isize - 1) * stride - 1;
        *e = convol4(&c, [src.at(o), src.at(o + 1), src.at(o + 2), src.at(o + 3)]);
    }
    let [f0, f1, f2, f3] = v;
    let value = f1
        + 0.5
            * (dy * (f2 - f0)
                + dy2 * (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3)
                + dy3 * (3.0 * (f1 - f2) + f3 - f0));
    T::clamp_from(value)
}

/// GDAL's `GWKBilinearResample4Sample`: density and value.
fn bilinear4<T: Sample>(src: &Window<T>, x: f64, y: f64) -> (f64, f64) {
    let (xs, ys) = (src.xsize as i32, src.ysize as i32);
    let stride = src.xsize as isize;
    let mut ix = (x - 0.5).floor() as i32;
    let mut iy = (y - 0.5).floor() as i32;
    let mut rx = 1.5 - (x - ix as f64);
    let mut ry = 1.5 - (y - iy as f64);
    if ix == -1 {
        ix = 0;
        rx = 1.0;
    }
    if iy == -1 {
        iy = 0;
        ry = 1.0;
    }
    let mut off = ix as isize + iy as isize * stride;
    let n = src.data.len() as isize;
    let mut shifted = false;
    if n == off + 1 || n == off + stride + 1 {
        shifted = true;
        off -= 1;
    }
    let (mut a_real, mut a_dens, mut div) = (0.0, 0.0, 0.0);
    let mut dens = [0.0; 2];
    let mut real = [0.0; 2];
    let mut pair = |o: isize,
                    row_ok: bool,
                    m1: f64,
                    m2: f64,
                    a_real: &mut f64,
                    a_dens: &mut f64,
                    div: &mut f64| {
        if row_ok && o >= 0 && o < n && src.row(o, 1, Some(&mut dens), &mut real) {
            if shifted {
                real[0] = real[1];
                dens[0] = dens[1];
            }
            if ix >= 0 && ix < xs && dens[0] > SRC_DENSITY_THRESHOLD {
                *div += m1;
                *a_real += real[0] * m1;
                *a_dens += dens[0] * m1;
            }
            if ix + 1 >= 0 && ix + 1 < xs && dens[1] > SRC_DENSITY_THRESHOLD {
                *div += m2;
                *a_real += real[1] * m2;
                *a_dens += dens[1] * m2;
            }
        }
    };
    pair(
        off,
        iy >= 0 && iy < ys,
        rx * ry,
        (1.0 - rx) * ry,
        &mut a_real,
        &mut a_dens,
        &mut div,
    );
    pair(
        off + stride,
        iy + 1 >= 0 && iy + 1 < ys,
        rx * (1.0 - ry),
        (1.0 - rx) * (1.0 - ry),
        &mut a_real,
        &mut a_dens,
        &mut div,
    );
    if div == 1.0 {
        (a_dens, a_real)
    } else if div < 0.00001 {
        (0.0, 0.0)
    } else {
        (a_dens / div, a_real / div)
    }
}

/// GDAL's `GWKCubicResample4Sample`: falls back to bilinear at window
/// edges or when any of the 16 taps is invalid.
fn cubic4<T: Sample>(src: &Window<T>, x: f64, y: f64) -> (f64, f64) {
    let (xs, ys) = (src.xsize as i32, src.ysize as i32);
    let stride = src.xsize as isize;
    let ix = (x - 0.5) as i32;
    let iy = (y - 0.5) as i32;
    let off = ix as isize + iy as isize * stride;
    let dx = x - 0.5 - ix as f64;
    let dy = y - 0.5 - iy as f64;
    if ix - 1 < 0 || ix + 2 >= xs || iy - 1 < 0 || iy + 2 >= ys {
        return bilinear4(src, x, y);
    }
    let cx = cubic_coeffs(dx);
    let mut v_dens = [0.0; 4];
    let mut v_real = [0.0; 4];
    let mut dens = [0.0; 4];
    let mut real = [0.0; 4];
    for r in 0..4 {
        let o = off + (r as isize - 1) * stride - 1;
        if !src.row(o, 2, Some(&mut dens), &mut real)
            || dens.iter().any(|&d| d < SRC_DENSITY_THRESHOLD)
        {
            return bilinear4(src, x, y);
        }
        v_dens[r] = convol4(&cx, dens);
        v_real[r] = convol4(&cx, real);
    }
    let cy = cubic_coeffs(dy);
    (convol4(&cy, v_dens), convol4(&cy, v_real))
}

/// GDAL's `GWKSetPixelValueReal`: apply a kernel's density and value to
/// target cell `dst`, mixing with what is there when the density is
/// partial, then rounding, clamping and stepping off `nodata`.
/// `dst_valid` is GDAL's target validity mask (present when the target
/// has a nodata value).
pub fn write_density<T: Sample>(
    out: &mut [T],
    dst_valid: Option<&[bool]>,
    dst: usize,
    density: f64,
    mut real: f64,
    nodata: Option<f64>,
) {
    if density < 0.9999 {
        if density < 0.0001 {
            return;
        }
        let dst_density = if dst_valid.is_some_and(|v| !v[dst]) {
            0.0
        } else {
            1.0
        };
        let dst_real = out[dst].to_f64();
        let influence = (1.0 - density) * dst_density;
        real = (real * density + dst_real * influence) / (density + influence);
    }
    let mut v = T::clamp_from(real);
    if nodata.is_some_and(|nd| nd == v.to_f64()) {
        v = v.nudge();
    }
    out[dst] = v;
}
