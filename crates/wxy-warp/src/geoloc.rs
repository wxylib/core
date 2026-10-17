//! GDAL's geolocation-array transformer (`alg/gdalgeoloc.cpp`, GDAL 3.12),
//! for sources whose cells are placed by 2D coordinate arrays (stage S5b,
//! curvilinear grids): forward by bilinear interpolation of the arrays,
//! inverse through a backmap (a regular grid in the arrays' CRS holding
//! source pixel positions, filled as GDAL fills it) refined to the exact
//! inverse bilinear position in the source cell that holds the point.
//!
//! Ported in GDAL's arithmetic order, with its float32 backmap, so that a
//! warp from a curvilinear source matches GDAL's cell for cell. Only the
//! in-memory (C array) backmap and the default options are ported:
//! oversampling 1.3, maximum accuracy, no quadtree, no axis swap.

/// GDAL's marker for a backmap cell with no source position.
const INVALID_BMXY: f32 = -10.0;
/// GDAL's block size for walking the arrays and the backmap.
const TILE_SIZE: usize = 256;
/// GDAL's `GEOLOC_BACKMAP_OVERSAMPLE_FACTOR` default.
const OVERSAMPLE: f64 = 1.3;

/// A geolocation array: positions of a source grid's cells, as GDAL's
/// `GDALGeoLocTransformInfo`.
#[derive(Debug, Clone)]
pub(crate) struct GeoLoc {
    nx: i32,
    ny: i32,
    x: Vec<f64>,
    y: Vec<f64>,
    nodata: Option<f64>,
    /// `GEOREFERENCING_CONVENTION=TOP_LEFT_CORNER`: array values sit at
    /// whole pixel coordinates; otherwise (`PIXEL_CENTER`) at half pixels.
    top_left: bool,
    /// A geographic CRS whose longitudes are all in [-180, 180]: longitudes
    /// are interpolated across the antimeridian.
    lon180: bool,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    bm_gt: [f64; 6],
    bm_w: i32,
    bm_h: i32,
    bm_x: Vec<f32>,
    bm_y: Vec<f32>,
}

/// GDAL's `OGRLinearRing` tests on the ring `(x0, y0), (x2, y2), (x3, y3),
/// (x1, y1), (x0, y0)`, each with the envelope test first.
struct Ring([(f64, f64); 5]);

impl Ring {
    fn new(q: &[f64; 8]) -> Self {
        let [x0, y0, x1, y1, x2, y2, x3, y3] = *q;
        Ring([(x0, y0), (x2, y2), (x3, y3), (x1, y1), (x0, y0)])
    }

    fn in_envelope(&self, tx: f64, ty: f64) -> bool {
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &(x, y) in &self.0 {
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        tx >= x0 && tx <= x1 && ty >= y0 && ty <= y1
    }

    fn contains(&self, tx: f64, ty: f64) -> bool {
        if !self.in_envelope(tx, ty) {
            return false;
        }
        let p = &self.0;
        let mut crossings = 0;
        let (mut px, mut py) = (p[0].0 - tx, p[0].1 - ty);
        for &(x, y) in &p[1..] {
            let (x1, y1) = (x - tx, y - ty);
            let (x2, y2) = (px, py);
            if (y1 > 0.0 && y2 <= 0.0) || (y2 > 0.0 && y1 <= 0.0) {
                let cut = (x1 * y2 - x2 * y1) / (y2 - y1);
                if 0.0 < cut {
                    crossings += 1;
                }
            }
            px = x1;
            py = y1;
        }
        crossings % 2 == 1
    }

    fn on_boundary(&self, tx: f64, ty: f64) -> bool {
        if !self.in_envelope(tx, ty) {
            return false;
        }
        let p = &self.0;
        let (mut px, mut py) = (tx - p[0].0, ty - p[0].1);
        for i in 1..p.len() {
            let (dx1, dy1) = (tx - p[i].0, ty - p[i].1);
            let (dx2, dy2) = (px, py);
            if dx1 * dy2 - dx2 * dy1 == 0.0 && !(dx1 == dx2 && dy1 == dy2) {
                let sx = p[i].0 - p[i - 1].0;
                let sy = p[i].1 - p[i - 1].1;
                let cross = dx2 * sx + dy2 * sy;
                if cross >= 0.0 && cross <= sx * sx + sy * sy {
                    return true;
                }
            }
            px = dx1;
            py = dy1;
        }
        false
    }

    fn holds(&self, tx: f64, ty: f64) -> bool {
        self.contains(tx, ty) || self.on_boundary(tx, ty)
    }
}

/// GDAL's `GDALInverseBilinearInterpolation`: adds to `(i, j)` the offset
/// within the quadrilateral of `(x, y)`.
#[allow(clippy::too_many_arguments)]
fn inverse_bilinear(x: f64, y: f64, q: &[f64; 8], i: &mut f64, j: &mut f64) {
    let [x0, y0, x1, y1, x2, y2, x3, y3] = *q;
    let a = (x0 - x) * (y0 - y2) - (y0 - y) * (x0 - x2);
    let b = (((x0 - x) * (y1 - y3) - (y0 - y) * (x1 - x3))
        + ((x1 - x) * (y0 - y2) - (y1 - y) * (x0 - x2)))
        / 2.0;
    let c = (x1 - x) * (y1 - y3) - (y1 - y) * (x1 - x3);
    let denom = a - 2.0 * b + c;
    let magnitude = a.abs() + b.abs() + c.abs();
    let s = if denom.abs() <= 1e-12 * magnitude {
        a / (a - c)
    } else {
        let root = (b * b - a * c).sqrt();
        let s1 = ((a - b) + root) / denom;
        let s2 = ((a - b) - root) / denom;
        if !(0.0..=1.0).contains(&s1) {
            s2
        } else {
            s1
        }
    };
    let tx = (1.0 - s) * (x0 - x2) + s * (x1 - x3);
    if tx.abs() > 1e-12 * magnitude {
        *i += ((1.0 - s) * (x0 - x) + s * (x1 - x)) / tx;
    } else {
        let ty = (1.0 - s) * (y0 - y2) + s * (y1 - y3);
        if ty.abs() > 1e-12 * magnitude {
            *i += ((1.0 - s) * (y0 - y) + s * (y1 - y)) / ty;
        }
    }
    *j += s;
}

impl GeoLoc {
    /// The transformer for arrays `x` and `y` (`ny` rows of `nx`, C order),
    /// with GDAL's backmap built. `geographic`: the arrays' CRS is
    /// geographic (in degrees).
    pub(crate) fn new(
        nx: usize,
        ny: usize,
        x: Vec<f64>,
        y: Vec<f64>,
        nodata: Option<f64>,
        top_left: bool,
        geographic: bool,
    ) -> Result<Self, String> {
        if nx == 0 || ny == 0 || x.len() != nx * ny || y.len() != nx * ny {
            return Err("geolocation arrays must both be ny by nx".into());
        }
        let mut g = GeoLoc {
            nx: nx as i32,
            ny: ny as i32,
            x,
            y,
            nodata,
            top_left,
            lon180: false,
            min_x: f64::MAX,
            max_x: -f64::MAX,
            min_y: f64::MAX,
            max_y: -f64::MAX,
            bm_gt: [0.0; 6],
            bm_w: 0,
            bm_h: 0,
            bm_x: Vec::new(),
            bm_y: Vec::new(),
        };
        g.load_finish(geographic);
        g.generate_backmap()?;
        Ok(g)
    }

    fn gx(&self, ix: i32, iy: i32) -> f64 {
        self.x[(iy * self.nx + ix) as usize]
    }

    fn gy(&self, ix: i32, iy: i32) -> f64 {
        self.y[(iy * self.nx + ix) as usize]
    }

    fn is_nodata(&self, v: f64) -> bool {
        self.nodata == Some(v)
    }

    fn offset(&self) -> f64 {
        if self.top_left {
            0.0
        } else {
            0.5
        }
    }

    fn unshift(&self, x: f64) -> f64 {
        if !self.lon180 {
            return x;
        }
        if !(-180.0..=180.0).contains(&x) {
            let v = (x + 180.0) % 360.0;
            return if v < 0.0 { v + 180.0 } else { v - 180.0 };
        }
        x
    }

    fn shift(&self, xref: f64, x: f64) -> f64 {
        if !self.lon180 {
            return x;
        }
        if xref < -170.0 && x > 170.0 {
            return x - 360.0;
        }
        if xref > 170.0 && x < -170.0 {
            return x + 360.0;
        }
        x
    }

    fn update_min_max(&mut self, x: f64, y: f64) {
        if x < self.min_x {
            self.min_x = x;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if y > self.max_y {
            self.max_y = y;
        }
    }

    /// GDAL's `LoadGeolocFinish`: the arrays' extent, widened to the
    /// grid's outer edges.
    fn load_finish(&mut self, geographic: bool) {
        for (by, bx) in blocks(self.ny as usize, self.nx as usize) {
            for iy in by.clone() {
                for ix in bx.clone() {
                    let x = self.gx(ix as i32, iy as i32);
                    if !self.is_nodata(x) {
                        let x = self.unshift(x);
                        let y = self.gy(ix as i32, iy as i32);
                        self.update_min_max(x, y);
                    }
                }
            }
        }
        if geographic && self.min_x >= -180.0 && self.max_x <= 180.0 {
            self.lon180 = true;
        }
        let (nx, ny) = (self.nx as f64, self.ny as f64);
        let edge = |g: &mut GeoLoc, px: f64, py: f64| {
            if let Some((mut x, y)) = g.pixel_line_to_xy(px, py) {
                if g.lon180 {
                    x = x.clamp(-180.0, 180.0);
                }
                g.update_min_max(x, y);
            }
        };
        if self.top_left {
            for ix in 0..=self.nx {
                edge(self, ix as f64, ny);
            }
            for iy in 0..=self.ny {
                edge(self, nx, iy as f64);
            }
        } else {
            for ix in 0..=self.nx {
                edge(self, ix as f64, -0.5);
            }
            for ix in 0..=self.nx {
                edge(self, ix as f64, ny - 1.0 + 0.5);
            }
            for iy in 0..=self.ny {
                edge(self, -0.5, iy as f64);
            }
            for iy in 0..=self.ny {
                edge(self, nx - 1.0 + 0.5, iy as f64);
            }
        }
    }

    /// GDAL's `PixelLineToXY` (floating point): bilinear interpolation of
    /// the arrays, extended beyond their edges by the nearest square.
    fn pixel_line_to_xy(&self, px: f64, py: f64) -> Option<(f64, f64)> {
        let (nx, ny) = (self.nx, self.ny);
        let mut ix = px.max(0.0).min((nx - 1) as f64) as i32;
        let mut iy = py.max(0.0).min((ny - 1) as f64) as i32;
        let (mut ox, mut oy) = (0.0, 0.0);
        for _ in 0..2 {
            let x00 = self.gx(ix, iy);
            let y00 = self.gy(ix, iy);
            if self.is_nodata(x00) {
                return None;
            }
            if ix + 1 < nx && iy + 1 < ny {
                let x10 = self.gx(ix + 1, iy);
                let y10 = self.gy(ix + 1, iy);
                let x01 = self.gx(ix, iy + 1);
                let y01 = self.gy(ix, iy + 1);
                let x11 = self.gx(ix + 1, iy + 1);
                let y11 = self.gy(ix + 1, iy + 1);
                if self.nodata.is_none()
                    || (!self.is_nodata(x10) && !self.is_nodata(x01) && !self.is_nodata(x11))
                {
                    let x10a = self.shift(x00, x10);
                    let x01a = self.shift(x00, x01);
                    let x11a = self.shift(x00, x11);
                    let (fx, fy) = (px - ix as f64, py - iy as f64);
                    ox = (1.0 - fy) * (x00 + fx * (x10a - x00)) + fy * (x01a + fx * (x11a - x01a));
                    ox = self.unshift(ox);
                    oy = (1.0 - fy) * (y00 + fx * (y10 - y00)) + fy * (y01 + fx * (y11 - y01));
                    break;
                }
            }
            if ix == nx - 1 && ix >= 1 && iy + 1 < ny {
                ix -= 1;
                continue;
            } else if iy == ny - 1 && iy >= 1 && ix + 1 < nx {
                iy -= 1;
                continue;
            } else if ix == nx - 1 && iy == ny - 1 && ix >= 1 && iy >= 1 {
                ix -= 1;
                iy -= 1;
                continue;
            } else if ix + 1 < nx && (self.nodata.is_none() || !self.is_nodata(self.gx(ix + 1, iy)))
            {
                let x10 = self.gx(ix + 1, iy);
                let y10 = self.gy(ix + 1, iy);
                ox = x00 + (px - ix as f64) * (self.shift(x00, x10) - x00);
                ox = self.unshift(ox);
                oy = y00 + (px - ix as f64) * (y10 - y00);
            } else if iy + 1 < ny && (self.nodata.is_none() || !self.is_nodata(self.gx(ix, iy + 1)))
            {
                let x01 = self.gx(ix, iy + 1);
                let y01 = self.gy(ix, iy + 1);
                ox = x00 + (py - iy as f64) * (self.shift(x00, x01) - x00);
                ox = self.unshift(ox);
                oy = y00 + (py - iy as f64) * (y01 - y00);
            } else {
                ox = self.unshift(x00);
                oy = y00;
            }
            break;
        }
        Some((ox, oy))
    }

    /// GDAL's `PixelLineToXY` (whole pixels): the array value, or the
    /// interpolation above off the arrays.
    fn pixel_line_to_xy_i(&self, px: i32, py: i32) -> Option<(f64, f64)> {
        if px >= 0 && px < self.nx && py >= 0 && py < self.ny {
            let x = self.gx(px, py);
            if self.is_nodata(x) {
                return None;
            }
            return Some((self.unshift(x), self.gy(px, py)));
        }
        self.pixel_line_to_xy(px as f64, py as f64)
    }

    /// The square from `(px, py)` to `(px + 1, py + 1)` as
    /// `[x0, y0, x1, y1, x2, y2, x3, y3]`: 0 its corner, 2 one pixel right,
    /// 1 one line down, 3 both.
    fn square(&self, px: i32, py: i32) -> Option<[f64; 8]> {
        let (x0, y0) = self.pixel_line_to_xy_i(px, py)?;
        let (x2, y2) = self.pixel_line_to_xy_i(px + 1, py)?;
        let (x1, y1) = self.pixel_line_to_xy_i(px, py + 1)?;
        let (x3, y3) = self.pixel_line_to_xy_i(px + 1, py + 1)?;
        Some([x0, y0, x1, y1, x2, y2, x3, y3])
    }

    /// Whether a square straddles the antimeridian (GDAL tests this
    /// before trying it at both -180 and +180); if so, its longitudes are
    /// moved below -180.
    fn straddles(&self, q: &mut [f64; 8]) -> bool {
        let [x0, _, x1, _, x2, _, x3, _] = *q;
        if self.lon180
            && x0.abs() > 170.0
            && x1.abs() > 170.0
            && x2.abs() > 170.0
            && x3.abs() > 170.0
            && ((x1 - x0).abs() > 180.0 || (x2 - x0).abs() > 180.0 || (x3 - x0).abs() > 180.0)
        {
            for k in [0, 2, 4, 6] {
                if q[k] > 0.0 {
                    q[k] -= 360.0;
                }
            }
            return true;
        }
        false
    }

    /// GDAL's `GenerateBackMap`, with its two hole-filling passes.
    fn generate_backmap(&mut self) -> Result<(), String> {
        let (nx, ny) = (self.nx, self.ny);
        let target = nx as f64 * ny as f64 * OVERSAMPLE;
        let size = ((self.max_x - self.min_x) * (self.max_y - self.min_y) / target).sqrt();
        if size == 0.0 || !size.is_finite() {
            return Err("invalid pixel size for the geolocation backmap".into());
        }
        let min_x = self.min_x - size / 2.0;
        let max_x = self.max_x + size / 2.0;
        let max_y = self.max_y + size / 2.0;
        let min_y = self.min_y - size / 2.0;
        let bw = ((max_x - min_x) / size).ceil();
        let bh = ((max_y - min_y) / size).ceil();
        if !(bw > 0.0 && bw + 2.0 < i32::MAX as f64) || !(bh > 0.0 && bh + 2.0 < i32::MAX as f64) {
            return Err(format!("backmap overflow: {bw} x {bh}"));
        }
        let (mut bw, mut bh) = (bw as i32, bh as i32);
        let psx = (max_x - min_x) / bw as f64;
        let psy = (max_y - min_y) / bh as f64;
        bw += 1;
        bh += 1;
        self.bm_w = bw;
        self.bm_h = bh;
        self.bm_gt = [min_x, psx, 0.0, max_y, 0.0, -psy];
        let n = (bw as usize) * (bh as usize);
        let mut bx = vec![0f32; n];
        let mut by = vec![0f32; n];
        let mut wt = vec![0f32; n];
        let at = |ix: i32, iy: i32| (iy * bw + ix) as usize;
        let off = self.offset();

        let update = |g: &GeoLoc,
                      bx: &mut [f32],
                      by: &mut [f32],
                      wt: &mut [f32],
                      ibx: i32,
                      iby: i32,
                      dx: f64,
                      dy: f64,
                      w: f64| {
            let k = at(ibx, iby);
            let ux = bx[k] + (w * (dx + off)) as f32;
            let uy = by[k] + (w * (dy + off)) as f32;
            let uw = wt[k] + w as f32;
            if uw > 0.0 {
                let fx = ux / uw;
                let fy = uy / uw;
                let gpx = f64::from(fx) - off;
                let gpy = f64::from(fy) - off;
                let ixa = (gpx.max(0.0) as i32).min(nx - 1);
                let iya = (gpy.max(0.0) as i32).min(ny - 1);
                let glx = g.gx(ixa, iya);
                let gly = g.gy(ixa, iya);
                let ix = dx as u32;
                let iy = dy as u32;
                if !g.is_nodata(glx)
                    && ((ix >= (nx - 1) as u32 || iy >= (ny - 1) as u32)
                        || ((glx - g.gx(ix as i32, iy as i32)).abs() <= 2.0 * psx
                            && (gly - g.gy(ix as i32, iy as i32)).abs() <= 2.0 * psy))
                {
                    bx[k] = ux;
                    by[k] = uy;
                    wt[k] = uw;
                }
            }
        };

        let step = 1.0 / OVERSAMPLE;
        let ranges = |n: i32| -> Vec<(f64, f64)> {
            let nblocks = (n as usize).div_ceil(TILE_SIZE);
            let mut se = vec![(0.0, 0.0); nblocks + 1];
            let mut last: i64 = -1;
            let mut d = -step;
            while d <= n as f64 + 2.0 * step {
                let b = (d / TILE_SIZE as f64) as i64;
                if b != last {
                    if last >= 0 {
                        se[last as usize].1 = d + step / 10.0;
                    }
                    se[b as usize].0 = d;
                    last = b;
                }
                d += step;
            }
            let b = (d / TILE_SIZE as f64) as usize;
            se[b].1 = d + step / 10.0;
            se
        };
        let ys = ranges(ny);
        let xs = ranges(nx);
        let (nyb, nxb) = (
            (ny as usize).div_ceil(TILE_SIZE),
            (nx as usize).div_ceil(TILE_SIZE),
        );

        for yb in ys.iter().take(nyb) {
            for xb in xs.iter().take(nxb) {
                let mut dy = yb.0;
                while dy < yb.1 {
                    let mut dx = xb.0;
                    while dx < xb.1 {
                        self.backmap_point(
                            dx, dy, min_x, max_y, psx, psy, &mut bx, &mut by, &mut wt, &update,
                        );
                        dx += step;
                    }
                    dy += step;
                }
            }
        }

        for k in 0..n {
            let w = wt[k];
            if w > 0.0 {
                bx[k] /= w;
                by[k] /= w;
            } else {
                bx[k] = INVALID_BMXY;
                by[k] = INVALID_BMXY;
            }
        }
        drop(wt);

        let (w, h) = (bw as usize, bh as usize);
        fill_nodata(&mut bx, w, h, 3.0, 1);
        fill_nodata(&mut by, w, h, 3.0, 1);

        // GDAL's last pass: along each row, bridge a gap between two valid
        // cells whose positions differ by at most 2 pixels.
        for iy in 0..h {
            let mut last: i64 = -1;
            let mut last_x = 0f32;
            for ix in 0..w {
                let v = bx[iy * w + ix];
                if v == INVALID_BMXY {
                    continue;
                }
                if last != -1 && ix as i64 > last + 1 && (v - last_x).abs() <= 2.0 {
                    let vy = by[iy * w + ix];
                    let last_y = by[iy * w + last as usize];
                    if (vy - last_y).abs() <= 2.0 {
                        for inner in (last + 1) as usize..ix {
                            let alpha = (inner as i64 - last) as f32 / (ix as i64 - last) as f32;
                            bx[iy * w + inner] = (1.0 - alpha) * last_x + alpha * v;
                            by[iy * w + inner] = (1.0 - alpha) * last_y + alpha * vy;
                        }
                    }
                }
                last = ix as i64;
                last_x = v;
            }
        }
        self.bm_x = bx;
        self.bm_y = by;
        Ok(())
    }

    /// One forward-projected sample of [`GeoLoc::generate_backmap`].
    #[allow(clippy::too_many_arguments)]
    fn backmap_point<F>(
        &self,
        dx: f64,
        dy: f64,
        min_x: f64,
        max_y: f64,
        psx: f64,
        psy: f64,
        bx: &mut [f32],
        by: &mut [f32],
        wt: &mut [f32],
        update: &F,
    ) where
        F: Fn(&GeoLoc, &mut [f32], &mut [f32], &mut [f32], i32, i32, f64, f64, f64),
    {
        let Some((glx, gly)) = self.pixel_line_to_xy(dx, dy) else {
            return;
        };
        let (bw, bh) = (self.bm_w, self.bm_h);
        let at = |ix: i32, iy: i32| (iy * bw + ix) as usize;
        let off = self.offset();
        let dbx = (glx - min_x) / psx;
        let dby = (max_y - gly) / psy;
        let ibx = dbx.floor() as i32;
        let iby = dby.floor() as i32;

        if ibx >= 0 && ibx < bw && iby >= 0 && iby < bh {
            let mut gx = min_x + ibx as f64 * psx;
            let gy = max_y - iby as f64 * psy;
            let mut found = false;
            let outer = if self.lon180 && gx.abs() >= 180.0 {
                2
            } else {
                1
            };
            for it in 0..outer {
                if it == 1 && gx >= 180.0 {
                    gx -= 360.0;
                } else if it == 1 && gx <= -180.0 {
                    gx += 360.0;
                }
                let nx0 = dx.floor() as i32;
                let ny0 = dy.floor() as i32;
                let mut sx = -1;
                while !found && sx <= 0 {
                    let mut sy = -1;
                    while !found && sy <= 0 {
                        let (pixel, line) = (nx0 + sx, ny0 + sy);
                        let Some(mut q) = self.square(pixel, line) else {
                            break;
                        };
                        let iters = if self.straddles(&mut q) { 2 } else { 1 };
                        for k in 0..iters {
                            if k == 1 {
                                for c in [0, 2, 4, 6] {
                                    q[c] += 360.0;
                                }
                            }
                            if Ring::new(&q).holds(gx, gy) {
                                found = true;
                                let (mut vx, mut vy) = (pixel as f64, line as f64);
                                inverse_bilinear(gx, gy, &q, &mut vx, &mut vy);
                                let k = at(ibx, iby);
                                bx[k] = (vx + off) as f32;
                                by[k] = (vy + off) as f32;
                                wt[k] = 1.0;
                            }
                        }
                        sy += 1;
                    }
                    sx += 1;
                }
            }
            if found {
                return;
            }
        }

        if ibx < -1 || iby < -1 || ibx > bw || iby > bh {
            return;
        }
        let fx = dbx - ibx as f64;
        let fy = dby - iby as f64;
        if ibx >= 0 && iby >= 0 && ibx < bw && iby < bh && wt[at(ibx, iby)] != 1.0 {
            update(self, bx, by, wt, ibx, iby, dx, dy, (1.0 - fx) * (1.0 - fy));
        }
        if iby >= 0 && ibx + 1 < bw && iby < bh && wt[at(ibx + 1, iby)] != 1.0 {
            update(self, bx, by, wt, ibx + 1, iby, dx, dy, fx * (1.0 - fy));
        }
        if ibx + 1 < bw && iby + 1 < bh && wt[at(ibx + 1, iby + 1)] != 1.0 {
            update(self, bx, by, wt, ibx + 1, iby + 1, dx, dy, fx * fy);
        }
        if ibx >= 0 && ibx < bw && iby + 1 < bh && wt[at(ibx, iby + 1)] != 1.0 {
            update(self, bx, by, wt, ibx, iby + 1, dx, dy, (1.0 - fx) * fy);
        }
    }

    /// Source pixel positions to positions in the arrays' CRS.
    pub(crate) fn forward(&self, x: &mut [f64], y: &mut [f64], ok: &mut [bool]) {
        let off = self.offset();
        for i in 0..x.len() {
            if x[i] == f64::INFINITY || y[i] == f64::INFINITY {
                ok[i] = false;
                continue;
            }
            match self.pixel_line_to_xy(x[i] - off, y[i] - off) {
                Some((gx, gy)) => {
                    x[i] = gx;
                    y[i] = gy;
                    ok[i] = true;
                }
                None => {
                    ok[i] = false;
                    x[i] = f64::INFINITY;
                    y[i] = f64::INFINITY;
                }
            }
        }
    }

    /// Positions in the arrays' CRS to source pixel positions: GDAL's
    /// backmap estimate, then the exact inverse in the source cell found
    /// near it.
    pub(crate) fn inverse(&self, x: &mut [f64], y: &mut [f64], ok: &mut [bool]) {
        let off = self.offset();
        let fail = |x: &mut f64, y: &mut f64, ok: &mut bool| {
            *ok = false;
            *x = f64::INFINITY;
            *y = f64::INFINITY;
        };
        let (bw, bh) = (self.bm_w, self.bm_h);
        let bm = |ix: i32, iy: i32| {
            let k = (iy * bw + ix) as usize;
            (self.bm_x[k], self.bm_y[k])
        };
        for i in 0..x.len() {
            if x[i] == f64::INFINITY || y[i] == f64::INFINITY {
                ok[i] = false;
                continue;
            }
            let (geo_x, geo_y) = (x[i], y[i]);
            let dbx = (x[i] - self.bm_gt[0]) / self.bm_gt[1];
            let dby = (y[i] - self.bm_gt[3]) / self.bm_gt[5];
            if !(dbx >= 0.0 && dby >= 0.0 && dbx + 1.0 < bw as f64 && dby + 1.0 < bh as f64) {
                fail(&mut x[i], &mut y[i], &mut ok[i]);
                continue;
            }
            let (ibx, iby) = (dbx as i32, dby as i32);
            let (x00, y00) = bm(ibx, iby);
            if x00 == INVALID_BMXY {
                fail(&mut x[i], &mut y[i], &mut ok[i]);
                continue;
            }
            let (x10, y10) = bm(ibx + 1, iby);
            let (x01, y01) = bm(ibx, iby + 1);
            let (x11, y11) = bm(ibx + 1, iby + 1);
            let (fx, fy) = (dbx - ibx as f64, dby - iby as f64);
            let d = f64::from;
            if x10 != INVALID_BMXY && x01 != INVALID_BMXY && x11 != INVALID_BMXY {
                x[i] =
                    (1.0 - fy) * (d(x00) + fx * d(x10 - x00)) + fy * (d(x01) + fx * d(x11 - x01));
                y[i] =
                    (1.0 - fy) * (d(y00) + fx * d(y10 - y00)) + fy * (d(y01) + fx * d(y11 - y01));
            } else if x10 != INVALID_BMXY {
                x[i] = d(x00) + fx * d(x10 - x00);
                y[i] = d(y00) + fx * d(y10 - y00);
            } else if x01 != INVALID_BMXY {
                x[i] = d(x00) + fy * d(x01 - x00);
                y[i] = d(y00) + fy * d(y01 - y00);
            } else {
                x[i] = d(x00);
                y[i] = d(y00);
            }
            let gpx = x[i] - off;
            let gpy = y[i] - off;

            let radius = if self.lon180 && geo_y.abs() >= 85.0 {
                5
            } else {
                3
            };
            let (npx, npy) = (gpx.floor() as i32, gpy.floor() as i32);
            let mut done = false;
            let mut r = 0;
            while !done && r <= radius {
                let count = if r == 0 { 1 } else { 8 * r };
                let mut it = 0;
                while !done && it < count {
                    let sx = if r == 0 {
                        0
                    } else if it < 2 * r {
                        -r + it
                    } else if it < 4 * r {
                        r
                    } else if it < 6 * r {
                        r - (it - 4 * r)
                    } else {
                        -r
                    };
                    let sy = if r == 0 {
                        0
                    } else if it < 2 * r {
                        r
                    } else if it < 4 * r {
                        r - (it - 2 * r)
                    } else if it < 6 * r {
                        -r
                    } else {
                        -r + (it - 6 * r)
                    };
                    it += 1;
                    if npx >= self.nx - sx || npy >= self.ny - sy {
                        continue;
                    }
                    let (ix, iy) = (npx + sx, npy + sy);
                    if ix >= -1 || iy >= -1 {
                        let Some(mut q) = self.square(ix, iy) else {
                            continue;
                        };
                        let iters = if self.straddles(&mut q) { 2 } else { 1 };
                        let mut k = 0;
                        while !done && k < iters {
                            if k == 1 {
                                for c in [0, 2, 4, 6] {
                                    q[c] += 360.0;
                                }
                            }
                            if Ring::new(&q).holds(geo_x, geo_y) {
                                let (mut vx, mut vy) = (ix as f64, iy as f64);
                                inverse_bilinear(geo_x, geo_y, &q, &mut vx, &mut vy);
                                x[i] = vx + off;
                                y[i] = vy + off;
                                done = true;
                            }
                            k += 1;
                        }
                    }
                }
                r += 1;
            }
            if !done {
                fail(&mut x[i], &mut y[i], &mut ok[i]);
                continue;
            }
            ok[i] = true;
        }
    }
}

/// GDAL's `START_ITER_PER_BLOCK`: row and column ranges of 256 by 256
/// blocks, rows of blocks first.
fn blocks(
    ny: usize,
    nx: usize,
) -> impl Iterator<Item = (std::ops::Range<usize>, std::ops::Range<usize>)> {
    let (nyb, nxb) = (ny.div_ceil(TILE_SIZE), nx.div_ceil(TILE_SIZE));
    (0..nyb).flat_map(move |by| {
        (0..nxb).map(move |bx| {
            let y0 = by * TILE_SIZE;
            let y1 = if by == nyb - 1 { ny } else { y0 + TILE_SIZE };
            let x0 = bx * TILE_SIZE;
            let x1 = if bx == nxb - 1 { nx } else { x0 + TILE_SIZE };
            (y0..y1, x0..x1)
        })
    })
}

/// GDAL's `GDALFillNodata` on a float32 band whose nodata is
/// [`INVALID_BMXY`], with inverse distance weighting and no `NODATA`
/// option, then its smoothing passes (`GDALMultiFilter`).
fn fill_nodata(band: &mut [f32], w: usize, h: usize, max_dist: f64, smoothing: usize) {
    let max_steps = max_dist.floor() as i32;
    let none: u32 = if w > 65533 || h > 65533 {
        4_000_002
    } else {
        65535
    };
    let mut mask: Vec<bool> = band.iter().map(|&v| v != INVALID_BMXY).collect();

    let mut y_work = vec![0u32; w * h];
    let mut v_work = vec![0f32; w * h];
    let mut last_y = vec![none; w];
    let mut this_y = vec![0u32; w];
    let mut last_v = vec![0f32; w];
    let mut this_v = vec![0f32; w];
    for iy in 0..h {
        for ix in 0..w {
            if mask[iy * w + ix] {
                this_v[ix] = band[iy * w + ix];
                this_y[ix] = iy as u32;
            } else if iy as f64 <= max_dist + last_y[ix] as f64 {
                this_v[ix] = last_v[ix];
                this_y[ix] = last_y[ix];
            } else {
                this_y[ix] = none;
            }
        }
        y_work[iy * w..(iy + 1) * w].copy_from_slice(&this_y);
        v_work[iy * w..(iy + 1) * w].copy_from_slice(&this_v);
        std::mem::swap(&mut this_v, &mut last_v);
        std::mem::swap(&mut this_y, &mut last_y);
    }

    last_y.iter_mut().for_each(|v| *v = none);
    let mut filt = vec![false; w * h];
    for iy in (0..h).rev() {
        let row = iy * w;
        let mut scan: Vec<f32> = band[row..row + w].to_vec();
        for ix in 0..w {
            if mask[row + ix] {
                this_v[ix] = scan[ix];
                this_y[ix] = iy as u32;
            } else if last_y[ix].wrapping_sub(iy as u32) as f64 <= max_dist {
                this_v[ix] = last_v[ix];
                this_y[ix] = last_y[ix];
            } else {
                this_y[ix] = none;
            }
        }
        let top_y = &y_work[row..row + w];
        let top_v = &v_work[row..row + w];
        for ix in 0..w {
            if mask[row + ix] {
                continue;
            }
            let mut steps = max_steps;
            let mut dist = [max_dist + 1.0; 4];
            let mut val = [0f32; 4];
            let check = |d: &mut f64, v: &mut f32, tx: usize, ty: u32, tv: f32| {
                if ty != none {
                    let dx = tx as f64 - ix as f64;
                    let dy = ty as f64 - iy as f64;
                    let sq = dx * dx + dy * dy;
                    if sq < *d * *d {
                        *d = sq.sqrt();
                        *v = tv;
                    }
                }
            };
            let mut step = 0;
            while step <= steps {
                let left = (ix as i32 - step).max(0) as usize;
                let right = (ix as i32 + step).min(w as i32 - 1) as usize;
                check(&mut dist[0], &mut val[0], left, top_y[left], top_v[left]);
                check(&mut dist[1], &mut val[1], left, last_y[left], last_v[left]);
                if step != 0 {
                    check(&mut dist[2], &mut val[2], right, top_y[right], top_v[right]);
                    check(
                        &mut dist[3],
                        &mut val[3],
                        right,
                        last_y[right],
                        last_v[right],
                    );
                    if step & 0x3 == 0 {
                        steps = dist[0].max(dist[1]).max(dist[2].max(dist[3])).floor() as i32;
                    }
                }
                step += 1;
            }
            let (mut wsum, mut vsum) = (0.0f64, 0.0f64);
            let mut any = false;
            for q in 0..4 {
                if dist[q] <= max_dist {
                    any = true;
                    let wq = 1.0 / dist[q];
                    wsum += wq;
                    vsum += f64::from(val[q]) * wq;
                }
            }
            if any {
                filt[row + ix] = true;
                if wsum > 0.0 {
                    mask[row + ix] = true;
                    scan[ix] = (vsum / wsum) as f32;
                } else {
                    scan[ix] = 0.0;
                }
            }
        }
        band[row..row + w].copy_from_slice(&scan);
        std::mem::swap(&mut this_v, &mut last_v);
        std::mem::swap(&mut this_y, &mut last_y);
    }

    if smoothing > 0 {
        multi_filter(band, w, h, &filt, smoothing);
    }
}

/// GDAL's `GDALMultiFilter`: `iterations` 3 by 3 averaging passes over
/// the filled cells (`filt`), in its rolling buffer of `iterations + 2`
/// lines and three passes, with valid cells (not nodata) as contributors.
fn multi_filter(band: &mut [f32], w: usize, h: usize, filt: &[bool], iterations: usize) {
    let lines = iterations + 2;
    let mut tmask = vec![false; w * lines];
    let mut fmask = vec![false; w * lines];
    let mut pass = vec![0f32; 3 * w * lines];
    let mut counter = 0usize;
    let plane = w * lines;
    for new_line in 0..h + iterations {
        counter = (counter + 1) % 3;
        let slast = (counter % 3) * plane;
        let last = ((counter + 1) % 3) * plane;
        let this = ((counter + 2) % 3) * plane;
        let off = new_line % lines;
        if new_line < h {
            for ix in 0..w {
                let v = band[new_line * w + ix];
                tmask[off * w + ix] = v != INVALID_BMXY;
                fmask[off * w + ix] = filt[new_line * w + ix];
                pass[this + off * w + ix] = v;
            }
        }
        let lo = new_line as i64 - iterations as i64;
        let mut f = new_line as i64 - 1;
        while f >= lo {
            let fl = f;
            f -= 1;
            let this_off = (fl.rem_euclid(lines as i64)) as usize;
            if fl >= 0 {
                let (src, dst) = (last + this_off * w, this + this_off * w);
                pass.copy_within(src..src + w, dst);
            }
            if fl < 1 || fl >= h as i64 - 1 {
                continue;
            }
            let last_off = ((fl - 1) % lines as i64) as usize;
            let next_off = ((fl + 1) % lines as i64) as usize;
            for ix in 0..w {
                let out = this + this_off * w + ix;
                if !fmask[this_off * w + ix] {
                    pass[out] = pass[last + this_off * w + ix];
                    continue;
                }
                let mut sum = 0f32;
                let mut wsum = 0f32;
                let mut add = |line: usize, m: usize, k: usize| {
                    if tmask[m * w + k] {
                        sum += pass[line + k];
                        wsum += 1.0;
                    }
                };
                let rows = [
                    (slast + last_off * w, last_off),
                    (last + this_off * w, this_off),
                    (this + next_off * w, next_off),
                ];
                for (line, m) in rows {
                    if ix > 0 {
                        add(line, m, ix - 1);
                    }
                    add(line, m, ix);
                    if ix < w - 1 {
                        add(line, m, ix + 1);
                    }
                }
                pass[out] = sum / wsum;
            }
        }
        if new_line >= iterations {
            let save = new_line - iterations;
            let o = this + (save % lines) * w;
            band[save * w..(save + 1) * w].copy_from_slice(&pass[o..o + w]);
        }
    }
}
