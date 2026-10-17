//! GDAL's polygon rasterizer, for finding the cells a polygon covers:
//! `GDALdllImageFilledPolygon` (a cell is in when its centre is) and
//! `GDALdllImageLineAllTouched` (with `ALL_TOUCHED`, every cell a ring
//! passes through), ported from GDAL 3.12's `llrasterize.cpp`, with the
//! ring orientation `gdalrasterize.cpp` imposes (OGR's `isClockwise`).

/// Cells as `(row, col)`, on a grid of `w` by `h`.
pub(crate) type Cells = Vec<(u32, u32)>;

/// OGR's `OGRLineString::isClockwise` on a closed ring.
pub(crate) fn is_clockwise(r: &[[f64; 2]]) -> bool {
    let n = r.len();
    if n < 2 {
        return true;
    }
    let mut fallback = false;
    // the lowest rightmost vertex
    let mut v = 0;
    for i in 1..n - 1 {
        if r[i][1] < r[v][1] || (r[i][1] == r[v][1] && r[i][0] > r[v][0]) {
            v = i;
            fallback = false;
        } else if r[i][1] == r[v][1] && r[i][0] == r[v][0] {
            fallback = true;
        }
    }
    const EPSILON: f64 = 1.0e-5;
    let near =
        |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < EPSILON && (a[1] - b[1]).abs() < EPSILON;
    let prev = if v == 0 { n - 2 } else { v - 1 };
    if near(r[prev], r[v]) {
        fallback = true;
    }
    let (dx0, dy0) = (r[prev][0] - r[v][0], r[prev][1] - r[v][1]);
    let next = if v + 1 >= n - 1 { 0 } else { v + 1 };
    if near(r[next], r[v]) {
        fallback = true;
    }
    let (dx1, dy1) = (r[next][0] - r[v][0], r[next][1] - r[v][1]);
    let cross = dx1 * dy0 - dx0 * dy1;
    if !fallback {
        if cross > 0.0 {
            return false;
        } else if cross < 0.0 {
            return true;
        }
    }
    // Green's formula, for degenerate rings
    let mut sum = r[0][0] * (r[1][1] - r[n - 1][1]);
    for i in 1..n - 1 {
        sum += r[i][0] * (r[i + 1][1] - r[i - 1][1]);
    }
    sum += r[n - 1][0] * (r[0][1] - r[n - 2][1]);
    sum < 0.0
}

/// C's `static_cast<int>` of a double already floored and in range.
fn int(v: f64) -> i32 {
    v as i32
}

/// `GDALdllImageFilledPolygon` in replace mode: every ring of one polygon
/// (`parts` point counts over `x`, `y`, in pixel and line), its spans
/// pushed onto `out` clipped to the grid.
pub(crate) fn filled_polygon(
    w: i32,
    h: i32,
    parts: &[usize],
    x: &[f64],
    y: &[f64],
    out: &mut Cells,
) {
    if parts.is_empty() {
        return;
    }
    let n: usize = parts.iter().sum();
    let mut ints: Vec<i32> = vec![0; n];
    let (mut dminy, mut dmaxy) = (y[0], y[0]);
    for &v in &y[1..n] {
        if v < dminy {
            dminy = v;
        } else if v > dmaxy {
            dmaxy = v;
        }
    }
    let miny = int(dminy.max(0.0));
    let maxy = int(dmaxy.min((h - 1) as f64));
    let (minx, maxx) = (0, w - 1);
    let mut burn = |row: i32, x1: i32, x2: i32| {
        // gvBurnScanline's clipping
        if x1 > x2 {
            return;
        }
        let (a, b) = (x1.max(0), x2.min(w - 1));
        for c in a..=b {
            out.push((row as u32, c as u32));
        }
    };
    for yy in miny..=maxy {
        let dy = yy as f64 + 0.5;
        let (mut part, mut partoffset, mut count) = (0, 0, 0);
        for i in 0..n {
            if i == partoffset + parts[part] {
                partoffset += parts[part];
                part += 1;
            }
            let (ind1, ind2) = if i == partoffset {
                (partoffset + parts[part] - 1, partoffset)
            } else {
                (i - 1, i)
            };
            let (mut dy1, mut dy2) = (y[ind1], y[ind2]);
            if (dy1 < dy && dy2 < dy) || (dy1 > dy && dy2 > dy) {
                continue;
            }
            let (dx1, dx2);
            if dy1 < dy2 {
                dx1 = x[ind1];
                dx2 = x[ind2];
            } else if dy1 > dy2 {
                std::mem::swap(&mut dy1, &mut dy2);
                dx2 = x[ind1];
                dx1 = x[ind2];
            } else {
                // bottom horizontal segments are filled on their own; top
                // ones by the regular loop
                if x[ind1] > x[ind2] {
                    let hx1 = (x[ind2] + 0.5).floor();
                    let hx2 = (x[ind1] + 0.5).floor();
                    if hx1 > maxx as f64 || hx2 <= 0.0 {
                        continue;
                    }
                    let h1 = int(hx1.max(0.0));
                    let h2 = int(hx2.min(w as f64));
                    burn(yy, h1, h2 - 1);
                }
                continue;
            }
            if dy < dy2 && dy >= dy1 {
                let intersect = ((dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1)
                    .clamp(i32::MIN as f64, i32::MAX as f64);
                ints[count] = int((intersect + 0.5).floor());
                count += 1;
            }
        }
        ints[..count].sort_unstable();
        let mut i = 0;
        while i + 1 < count {
            if ints[i] <= maxx && ints[i + 1] > minx {
                burn(yy, ints[i], ints[i + 1] - 1);
            }
            i += 2;
        }
    }
}

/// `GDALdllImageLineAllTouched` for polygon rings (`bIntersectOnly`, no
/// burn-once bookkeeping): every cell each ring passes through.
pub(crate) fn line_all_touched(
    w: i32,
    h: i32,
    parts: &[usize],
    px: &[f64],
    py: &[f64],
    out: &mut Cells,
) {
    const EPS: f64 = 1e-4;
    let (wf, hf) = (w as f64, h as f64);
    let mut point = |row: i32, col: i32| out.push((row as u32, col as u32));
    let mut n = 0;
    for &size in parts {
        for j in 1..size {
            let (mut x0, mut y0) = (px[n + j - 1], py[n + j - 1]);
            let (mut x1, mut y1) = (px[n + j], py[n + j]);
            if (y0 < 0.0 && y1 < 0.0)
                || (y0 > hf && y1 > hf)
                || (x0 < 0.0 && x1 < 0.0)
                || (x0 > wf && x1 > wf)
            {
                continue;
            }
            let lim = |v: f64| v >= i32::MIN as f64 && v <= i32::MAX as f64;
            if !(lim(x0) && lim(y0) && lim(x1) && lim(y1)) {
                continue;
            }
            if x0 > x1 {
                std::mem::swap(&mut x0, &mut x1);
                std::mem::swap(&mut y0, &mut y1);
            }
            // vertical
            if (x0 - x1).abs() < 0.01 {
                if (x0 - x0.round()).abs() < EPS && (x1 - x1.round()).abs() < EPS {
                    continue;
                }
                if y1 < y0 {
                    std::mem::swap(&mut y0, &mut y1);
                }
                let ix = int(x1.floor());
                let mut iy = int(y0.floor());
                let mut iy_end = int((y1 - EPS).floor());
                if ix < 0 || ix >= w {
                    continue;
                }
                if iy < 0 {
                    iy = 0;
                }
                if iy_end >= h {
                    iy_end = h - 1;
                }
                while iy <= iy_end {
                    point(iy, ix);
                    iy += 1;
                }
                continue;
            }
            // horizontal
            if (y0 - y1).abs() < 0.01 {
                if (y0 - y0.round()).abs() < EPS && (y1 - y1.round()).abs() < EPS {
                    continue;
                }
                if x1 < x0 {
                    std::mem::swap(&mut x0, &mut x1);
                }
                let mut ix = int(x0.floor());
                let iy = int(y0.floor());
                let mut ix_end = int((x1 - EPS).floor());
                if iy < 0 || iy >= h {
                    continue;
                }
                if ix < 0 {
                    ix = 0;
                }
                if ix_end >= w {
                    ix_end = w - 1;
                }
                while ix <= ix_end {
                    point(iy, ix);
                    ix += 1;
                }
                continue;
            }
            // sloped, left to right
            let slope = (y1 - y0) / (x1 - x0);
            if x1 > wf {
                y1 -= (x1 - wf) * slope;
                x1 = wf;
            }
            if x0 < 0.0 {
                y0 += (0.0 - x0) * slope;
                x0 = 0.0;
            }
            if y1 > y0 {
                if y0 < 0.0 {
                    let dx = (0.0 - y0) / slope;
                    x0 += dx;
                    y0 = 0.0;
                }
                if y1 >= hf {
                    x1 += (y1 - hf) / slope;
                    if x1 > wf {
                        x1 = wf;
                    }
                }
            } else {
                if y0 >= hf {
                    let dx = (hf - y0) / slope;
                    x0 += dx;
                    y0 = hf;
                }
                if y1 < 0.0 {
                    x1 -= (y1 - 0.0) / slope;
                }
            }
            while x0 >= 0.0 && x0 < x1 {
                let ix = int(x0.floor());
                let iy = int(y0.floor());
                if iy >= 0 && iy < h {
                    point(iy, ix);
                }
                let mut sx = (x0 + 1.0).floor() - x0;
                let mut sy = sx * slope;
                if int((y0 + sy).floor()) == iy {
                    x0 += sx;
                    y0 += sy;
                } else if slope < 0.0 {
                    sy = iy as f64 - y0;
                    if sy > -0.000000001 {
                        sy = -0.000000001;
                    }
                    sx = sy / slope;
                    x0 += sx;
                    y0 += sy;
                } else {
                    sy = (iy + 1) as f64 - y0;
                    if sy < 0.000000001 {
                        sy = 0.000000001;
                    }
                    sx = sy / slope;
                    x0 += sx;
                    y0 += sy;
                }
            }
        }
        n += size;
    }
}
