//! The planner: for a read of a classic VRT band, which sources are touched,
//! which source window each contributes, where it lands in the output buffer,
//! and which per-source operations apply. No bytes are read here.
//!
//! The window arithmetic follows the reference library's
//! `VRTSimpleSource::GetSrcDstWindow` (GDAL `frmts/vrt/vrtsources.cpp`, MIT,
//! Frank Warmerdam and Even Rouault), and [`nearest_indices`] follows the
//! nearest-neighbour loop of `GDALRasterBand::IRasterIO`
//! (`gcore/rasterio.cpp`), so a reader that samples as the plan says gives
//! the reference library's values.

use crate::classic::{Dataset, Rect, Source};
use crate::dsn::Dsn;
use crate::Error;

/// A read request on the virtual raster: a window (possibly fractional) and
/// the output buffer size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request {
    pub x_off: f64,
    pub y_off: f64,
    pub x_size: f64,
    pub y_size: f64,
    pub buf_x: usize,
    pub buf_y: usize,
}

impl Request {
    /// A window at full resolution.
    pub fn window(x_off: usize, y_off: usize, x_size: usize, y_size: usize) -> Request {
        Request::resampled(x_off, y_off, x_size, y_size, x_size, y_size)
    }

    /// A window read into a buffer of another size (nearest).
    pub fn resampled(
        x_off: usize,
        y_off: usize,
        x_size: usize,
        y_size: usize,
        buf_x: usize,
        buf_y: usize,
    ) -> Request {
        Request {
            x_off: x_off as f64,
            y_off: y_off as f64,
            x_size: x_size as f64,
            y_size: y_size as f64,
            buf_x,
            buf_y,
        }
    }
}

/// What one source contributes to one request.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceWindow {
    /// The source window, fractional (what nearest sampling steps through).
    pub req: [f64; 4],
    /// The same window in whole pixels, as the source is asked for it.
    pub req_px: [usize; 4],
    /// Offset and size in the output buffer: x_off, y_off, x_size, y_size.
    pub out: [usize; 4],
}

/// A per-source operation, in the order it is applied.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Source values equal to this are skipped (the buffer keeps what it had).
    SkipNodata(f64),
    /// value * ratio + offset.
    Linear { ratio: f64, offset: f64 },
    /// Piecewise-linear lookup (input, output), clamped at both ends.
    Lut(Vec<(f64, f64)>),
}

/// One row of a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanRow {
    /// Index of the source in the band's source list.
    pub source: usize,
    /// The source name, resolved against the VRT's location when relative.
    pub dsn: Dsn,
    /// Source band number, and whether it is that band's mask.
    pub band: (usize, bool),
    pub window: SourceWindow,
    pub ops: Vec<Op>,
    /// True when the source window and the output region differ in size:
    /// an implicit resampling (design m11, "quiet operations").
    pub resampled: bool,
}

/// Source facts the planner needs that a document may not carry: the
/// source's size (when `<SourceProperties>` is absent), and the parsed
/// document of a nested VRT. A reader implements this over its own I/O.
pub trait Resolver {
    fn size(&self, dsn: &Dsn) -> Option<(usize, usize)>;
    fn open_vrt(&self, _dsn: &Dsn) -> Option<Dataset> {
        None
    }
}

/// A plan row at the bottom of any nesting: `dsn` is not a VRT.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafRow {
    pub dsn: Dsn,
    pub band: (usize, bool),
    pub window: SourceWindow,
    /// Operations from the outermost source inwards; apply innermost first.
    pub ops: Vec<Vec<Op>>,
}

/// GDAL's rounding: snap to the nearest integer when within 1e-3.
fn round_if_close(v: f64) -> f64 {
    let c = (v + 0.5).floor();
    if (v - c).abs() < 1e-3 {
        c
    } else {
        v
    }
}

/// The source and destination windows for one source, or `None` when the
/// request misses it. `nearest` selects the reference library's rounding for
/// nearest-neighbour reads with fractional source or destination offsets.
pub fn src_dst_window(
    src: &Rect,
    dst: &Rect,
    source_size: (usize, usize),
    r: &Request,
    nearest: bool,
) -> Option<SourceWindow> {
    if src.x_size == 0.0 || src.y_size == 0.0 || dst.x_size == 0.0 || dst.y_size == 0.0 {
        return None;
    }
    let (sx, sy) = (source_size.0 as f64, source_size.1 as f64);
    if r.x_off >= dst.x_off + dst.x_size
        || r.y_off >= dst.y_off + dst.y_size
        || r.x_off + r.x_size <= dst.x_off
        || r.y_off + r.y_size <= dst.y_off
    {
        return None;
    }
    let (mut out_x, mut out_y, mut out_w, mut out_h) = (0i64, 0i64, r.buf_x as i64, r.buf_y as i64);
    let (mut mod_x, mut mod_y) = (false, false);
    let (mut rx, mut ry, mut rw, mut rh) = (r.x_off, r.y_off, r.x_size, r.y_size);
    if rx < dst.x_off {
        rw = rw + rx - dst.x_off;
        rx = dst.x_off;
        mod_x = true;
    }
    if ry < dst.y_off {
        rh = rh + ry - dst.y_off;
        ry = dst.y_off;
        mod_y = true;
    }
    if rx + rw > dst.x_off + dst.x_size {
        rw = dst.x_off + dst.x_size - rx;
        mod_x = true;
    }
    if ry + rh > dst.y_off + dst.y_size {
        rh = dst.y_off + dst.y_size - ry;
        mod_y = true;
    }
    let scale_x = src.x_size / dst.x_size;
    let scale_y = src.y_size / dst.y_size;
    let mut dx = (rx - dst.x_off) * scale_x + src.x_off;
    let mut dy = (ry - dst.y_off) * scale_y + src.y_off;
    let mut dw = rw * scale_x;
    let mut dh = rh * scale_y;
    if !dx.is_finite()
        || !dy.is_finite()
        || !dw.is_finite()
        || !dh.is_finite()
        || dw < 0.0
        || dh < 0.0
    {
        return None;
    }
    if dx < 0.0 {
        dw += dx;
        dx = 0.0;
        mod_x = true;
    }
    if dy < 0.0 {
        dh += dy;
        dy = 0.0;
        mod_y = true;
    }
    const EPSILON: f64 = 1e-10;
    let frac = |v: f64| (v - v.round()).abs() > EPSILON;
    let (mut nx, mut ny) =
        if nearest && (frac(src.x_off) || frac(src.y_off) || frac(dst.x_off) || frac(dst.y_off)) {
            ((dx + 0.5 + EPSILON) as i64, (dy + 0.5 + EPSILON) as i64)
        } else {
            (dx as i64, dy as i64)
        };
    const EPS: f64 = 1e-3;
    if dx - nx as f64 > 1.0 - EPS {
        nx += 1;
        dx = nx as f64;
    }
    if dy - ny as f64 > 1.0 - EPS {
        ny += 1;
        dy = ny as f64;
    }
    let mut nw = ((dw + 0.5).floor() as i64).max(1);
    let mut nh = ((dh + 0.5).floor() as i64).max(1);
    if nx + nw > sx as i64 {
        nw = sx as i64 - nx;
        mod_x = true;
    }
    if dx + dw > sx {
        dw = sx - dx;
        mod_x = true;
    }
    if ny + nh > sy as i64 {
        nh = sy as i64 - ny;
        mod_y = true;
    }
    if dy + dh > sy {
        dh = sy - dy;
        mod_y = true;
    }
    if nx >= sx as i64 || ny >= sy as i64 || nw <= 0 || nh <= 0 {
        return None;
    }
    if mod_x || mod_y {
        let to_dst_x = |x: f64| ((x - src.x_off) / src.x_size) * dst.x_size + dst.x_off;
        let to_dst_y = |y: f64| ((y - src.y_off) / src.y_size) * dst.y_size + dst.y_off;
        let (ulx, uly) = (to_dst_x(dx), to_dst_y(dy));
        let (lrx, lry) = (to_dst_x(dx + dw), to_dst_y(dy + dh));
        if mod_x {
            let s = r.buf_x as f64 / r.x_size;
            let o = (ulx - r.x_off) * s;
            out_x = if o <= 0.0 { 0 } else { (o + EPS) as i64 };
            let d = (o - out_x as f64) / s / dst.x_size * src.x_size;
            dx -= d;
            dw += d;
            let right = (lrx - r.x_off) * s;
            if right < o {
                return None;
            }
            let nright = (right - EPS).ceil() as i64;
            out_w = nright - out_x;
            if out_x + out_w > r.buf_x as i64 {
                out_w = r.buf_x as i64 - out_x;
            }
            dw += (nright as f64 - right) / s / dst.x_size * src.x_size;
        }
        if mod_y {
            let s = r.buf_y as f64 / r.y_size;
            let o = (uly - r.y_off) * s;
            out_y = if o <= 0.0 { 0 } else { (o + EPS) as i64 };
            let d = (o - out_y as f64) / s / dst.y_size * src.y_size;
            dy -= d;
            dh += d;
            let bottom = (lry - r.y_off) * s;
            if bottom < o {
                return None;
            }
            let nbottom = (bottom - EPS).ceil() as i64;
            out_h = nbottom - out_y;
            if out_y + out_h > r.buf_y as i64 {
                out_h = r.buf_y as i64 - out_y;
            }
            dh += (nbottom as f64 - bottom) / s / dst.y_size * src.y_size;
        }
        if out_w < 1 || out_h < 1 {
            return None;
        }
    }
    Some(SourceWindow {
        req: [
            round_if_close(dx),
            round_if_close(dy),
            round_if_close(dw),
            round_if_close(dh),
        ],
        req_px: [nx as usize, ny as usize, nw as usize, nh as usize],
        out: [
            out_x as usize,
            out_y as usize,
            out_w as usize,
            out_h as usize,
        ],
    })
}

/// For each output column and row of a source window, the source pixel a
/// nearest-neighbour read takes (the reference library's sampling loop).
pub fn nearest_indices(w: &SourceWindow, source_size: (usize, usize)) -> (Vec<usize>, Vec<usize>) {
    const EPS: f64 = 1e-10;
    let [dx, dy, dw, dh] = w.req;
    let (ow, oh) = (w.out[2], w.out[3]);
    let inc_x = dw / ow as f64;
    let inc_y = dh / oh as f64;
    let max_x = (source_size.0 - 1) as f64;
    let max_y = (source_size.1 - 1) as f64;
    let mut xs = Vec::with_capacity(ow);
    let mut x = 0.5 * inc_x + dx + EPS;
    for _ in 0..ow {
        xs.push(x.max(0.0).min(max_x) as usize);
        x += inc_x;
    }
    let ys = (0..oh)
        .map(|i| ((i as f64 + 0.5) * inc_y + dy + EPS).max(0.0).min(max_y) as usize)
        .collect();
    (xs, ys)
}

fn ops(s: &Source) -> Vec<Op> {
    let mut v = vec![];
    if s.kind != "ComplexSource" {
        return v;
    }
    if let Some(n) = s.nodata_value() {
        v.push(Op::SkipNodata(n));
    }
    if s.scale_offset.is_some() || s.scale_ratio.is_some() {
        v.push(Op::Linear {
            ratio: s.scale_ratio.unwrap_or(1.0),
            offset: s.scale_offset.unwrap_or(0.0),
        });
    }
    if let Some(l) = &s.lut {
        v.push(Op::Lut(l.clone()));
    }
    v
}

/// Plan a read of band `band` (1-based) of `ds`. `base` is the directory or
/// URL prefix of the VRT, for relative source names. Sources not yet modelled
/// for planning (averaged, kernel-filtered, warped or derived bands) are an
/// error rather than a silent omission.
pub fn plan_band(
    ds: &Dataset,
    band: usize,
    r: &Request,
    base: &str,
    resolver: &dyn Resolver,
) -> Result<Vec<PlanRow>, Error> {
    if let Some(s) = &ds.sub_class {
        return Err(Error::Unsupported(format!("subClass {s}")));
    }
    let b = ds
        .bands
        .get(band.wrapping_sub(1))
        .ok_or_else(|| Error::Invalid(format!("no band {band}")))?;
    if let Some(s) = &b.sub_class {
        return Err(Error::Unsupported(format!("band subClass {s}")));
    }
    let mut rows = vec![];
    for (i, s) in b.sources.iter().enumerate() {
        if s.kind != "SimpleSource" && s.kind != "ComplexSource" {
            return Err(Error::Unsupported(s.kind.clone()));
        }
        let dsn = if s.relative {
            s.filename.relative_to(base)
        } else {
            s.filename.clone()
        };
        let size = s
            .declared_size()
            .or_else(|| resolver.size(&dsn))
            .ok_or_else(|| Error::Invalid(format!("size of source {dsn} unknown")))?;
        let full = Rect {
            x_off: 0.0,
            y_off: 0.0,
            x_size: size.0 as f64,
            y_size: size.1 as f64,
        };
        let src = s.src_rect.unwrap_or(full);
        let dst = s.dst_rect.unwrap_or(full);
        let nearest = s
            .resampling
            .as_deref()
            .is_none_or(|m| m.eq_ignore_ascii_case("nearest"));
        if !nearest {
            return Err(Error::Unsupported(format!(
                "resampling {}",
                s.resampling.as_deref().unwrap_or("")
            )));
        }
        if let Some(w) = src_dst_window(&src, &dst, size, r, true) {
            let resampled = (w.req[2] - w.out[2] as f64).abs() > 1e-9
                || (w.req[3] - w.out[3] as f64).abs() > 1e-9;
            rows.push(PlanRow {
                source: i,
                dsn,
                band: s.band_number()?,
                window: w,
                ops: ops(s),
                resampled,
            });
        }
    }
    Ok(rows)
}

/// Plan through nested VRTs down to sources that are not VRTs. Each nested
/// VRT is planned for the fractional source window its parent asks of it,
/// into the parent's output region, as the reference library reads it.
pub fn plan_band_leaves(
    ds: &Dataset,
    band: usize,
    r: &Request,
    base: &str,
    resolver: &dyn Resolver,
) -> Result<Vec<LeafRow>, Error> {
    let mut out = vec![];
    for row in plan_band(ds, band, r, base, resolver)? {
        if row.dsn.is_virtual() {
            if let Some(child) = resolver.open_vrt(&row.dsn) {
                let child_base = parent_of(&row.dsn.to_string());
                let w = &row.window;
                let cr = Request {
                    x_off: w.req[0],
                    y_off: w.req[1],
                    x_size: w.req[2],
                    y_size: w.req[3],
                    buf_x: w.out[2],
                    buf_y: w.out[3],
                };
                for mut leaf in plan_band_leaves(&child, row.band.0, &cr, &child_base, resolver)? {
                    leaf.window.out[0] += w.out[0];
                    leaf.window.out[1] += w.out[1];
                    leaf.ops.insert(0, row.ops.clone());
                    out.push(leaf);
                }
                continue;
            }
        }
        out.push(LeafRow {
            dsn: row.dsn,
            band: row.band,
            window: row.window,
            ops: vec![row.ops],
        });
    }
    Ok(out)
}

/// Directory part of a path or URL, without the trailing separator.
pub fn parent_of(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[..i].to_string(),
        None => String::new(),
    }
}
