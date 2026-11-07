//! Source footprints: where each source of a classic VRT band lands, in the
//! VRT's pixel grid and in its coordinate reference system, and which sources
//! meet a box.
//!
//! This is the catalogue view of a VRT: the question a map client asks
//! ("which files cover this viewport, and what do I fetch") without a read
//! plan. It holds for any source kind and any resampling, because it says
//! where a source is placed, not how its pixels are sampled. A read still
//! goes through [`crate::plan`].
//!
//! Each footprint is the destination rectangle the reference library would
//! fill: `DstRect` (or the source's own size at the origin when absent),
//! shrunk where `SrcRect` reaches past the source's edges (the same
//! proportional clip `VRTSimpleSource::GetSrcDstWindow` makes) and clipped
//! to the VRT's own size. Boxes meet when their interiors overlap, so a
//! source that only touches the query along an edge is not returned.

use crate::classic::{Dataset, Rect};
use crate::dsn::Dsn;
use crate::plan::Resolver;
use crate::Error;

/// An axis-aligned box in the VRT's coordinate reference system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl Bounds {
    pub fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Bounds {
        Bounds {
            xmin,
            xmax,
            ymin,
            ymax,
        }
    }

    /// True when the interiors overlap: interval overlap on each axis. This
    /// also holds when two boxes cross without either holding a corner of the
    /// other, which a corner-in-box test misses (design watch/vrtmate.md).
    pub fn overlaps(&self, other: &Bounds) -> bool {
        self.xmin < other.xmax
            && other.xmin < self.xmax
            && self.ymin < other.ymax
            && other.ymin < self.ymax
    }
}

/// Where one source of a band is placed.
#[derive(Debug, Clone, PartialEq)]
pub struct Footprint {
    /// Index of the source in the band's source list.
    pub source: usize,
    /// Element name: `SimpleSource`, `ComplexSource`, ...
    pub kind: String,
    /// The source name, resolved against the VRT's location when relative.
    pub dsn: Dsn,
    /// Source band number, and whether it is that band's mask.
    pub band: (usize, bool),
    /// The destination rectangle in VRT pixels, after the clips above.
    pub dst: Rect,
    /// The same rectangle in the VRT's coordinates. With a rotated
    /// geotransform this is the box around the four transformed corners.
    pub bounds: Bounds,
    /// The `resampling` attribute, as written.
    pub resampling: Option<String>,
    /// True when the source's size was known, so `SrcRect` beyond its edges
    /// could be clipped. Without `<SourceProperties>` and a resolver that
    /// knows the size, `dst` is `DstRect` clipped to the VRT only.
    pub size_known: bool,
}

/// The reference library's geotransform for a VRT without one.
const DEFAULT_GT: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Box around a pixel rectangle under a geotransform.
pub fn rect_bounds(gt: &[f64; 6], r: &Rect) -> Bounds {
    let corners = [
        (r.x_off, r.y_off),
        (r.x_off + r.x_size, r.y_off),
        (r.x_off, r.y_off + r.y_size),
        (r.x_off + r.x_size, r.y_off + r.y_size),
    ];
    let mut b = Bounds::new(
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (c, l) in corners {
        let x = gt[0] + c * gt[1] + l * gt[2];
        let y = gt[3] + c * gt[4] + l * gt[5];
        b.xmin = b.xmin.min(x);
        b.xmax = b.xmax.max(x);
        b.ymin = b.ymin.min(y);
        b.ymax = b.ymax.max(y);
    }
    b
}

/// Shrink `dst` where `src` reaches outside `[0, size)`, in proportion, as
/// the reference library does before any read.
fn clip_to_source(src: &Rect, dst: &Rect, size: (usize, usize)) -> Rect {
    let mut d = *dst;
    if src.x_size > 0.0 {
        let sx = dst.x_size / src.x_size;
        let x0 = src.x_off.max(0.0);
        let x1 = (src.x_off + src.x_size).min(size.0 as f64);
        d.x_off = dst.x_off + (x0 - src.x_off) * sx;
        d.x_size = ((x1 - x0) * sx).max(0.0);
    }
    if src.y_size > 0.0 {
        let sy = dst.y_size / src.y_size;
        let y0 = src.y_off.max(0.0);
        let y1 = (src.y_off + src.y_size).min(size.1 as f64);
        d.y_off = dst.y_off + (y0 - src.y_off) * sy;
        d.y_size = ((y1 - y0) * sy).max(0.0);
    }
    d
}

fn clip_to_raster(d: &Rect, ds: &Dataset) -> Rect {
    let x0 = d.x_off.max(0.0);
    let y0 = d.y_off.max(0.0);
    let x1 = (d.x_off + d.x_size).min(ds.x_size as f64);
    let y1 = (d.y_off + d.y_size).min(ds.y_size as f64);
    Rect {
        x_off: x0,
        y_off: y0,
        x_size: (x1 - x0).max(0.0),
        y_size: (y1 - y0).max(0.0),
    }
}

/// Footprints of every source of band `band` (1-based) that lands on the
/// VRT. `base` is the directory or URL prefix of the VRT, for relative source
/// names. Warped VRTs have no source placements and are an error.
pub fn footprints(
    ds: &Dataset,
    band: usize,
    base: &str,
    resolver: &dyn Resolver,
) -> Result<Vec<Footprint>, Error> {
    if let Some(s) = &ds.sub_class {
        return Err(Error::Unsupported(format!("subClass {s}")));
    }
    let b = ds
        .bands
        .get(band.wrapping_sub(1))
        .ok_or_else(|| Error::Invalid(format!("no band {band}")))?;
    let gt = ds.geotransform.unwrap_or(DEFAULT_GT);
    let mut out = vec![];
    for (i, s) in b.sources.iter().enumerate() {
        let dsn = if s.relative {
            s.filename.relative_to(base)
        } else {
            s.filename.clone()
        };
        let size = s.declared_size().or_else(|| resolver.size(&dsn));
        let dst = match (s.dst_rect, size) {
            (Some(d), _) => d,
            (None, Some(sz)) => Rect {
                x_off: 0.0,
                y_off: 0.0,
                x_size: sz.0 as f64,
                y_size: sz.1 as f64,
            },
            (None, None) => {
                return Err(Error::Invalid(format!(
                    "source {dsn} has no DstRect and its size is unknown"
                )))
            }
        };
        let dst = match (s.src_rect, size) {
            (Some(src), Some(sz)) => clip_to_source(&src, &dst, sz),
            _ => dst,
        };
        let dst = clip_to_raster(&dst, ds);
        if dst.x_size <= 0.0 || dst.y_size <= 0.0 {
            continue;
        }
        out.push(Footprint {
            source: i,
            kind: s.kind.clone(),
            dsn,
            band: s.band_number()?,
            bounds: rect_bounds(&gt, &dst),
            dst,
            resampling: s.resampling.clone(),
            size_known: size.is_some(),
        });
    }
    Ok(out)
}

/// The footprints of band `band` whose box overlaps `query`, in source order.
pub fn sources_in_bounds(
    ds: &Dataset,
    band: usize,
    query: &Bounds,
    base: &str,
    resolver: &dyn Resolver,
) -> Result<Vec<Footprint>, Error> {
    Ok(footprints(ds, band, base, resolver)?
        .into_iter()
        .filter(|f| f.bounds.overlaps(query))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoIo;
    impl Resolver for NoIo {
        fn size(&self, _: &Dsn) -> Option<(usize, usize)> {
            None
        }
    }

    /// Two 1-degree tiles side by side at 0.1 degree, with fractional
    /// offsets and bilinear resampling, as the USGS 3DEP VRTs are written.
    const TILES: &str = r#"<VRTDataset rasterXSize="20" rasterYSize="10">
  <GeoTransform>-121.0, 0.1, 0.0, 45.0, 0.0, -0.1</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <ComplexSource resampling="bilinear">
      <SourceFilename relativeToVRT="0">/vsicurl/https://h/n45w121.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="10" RasterYSize="10" DataType="Float32" />
      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10" />
      <DstRect xOff="0.0002" yOff="0" xSize="10" ySize="10" />
    </ComplexSource>
    <ComplexSource resampling="bilinear">
      <SourceFilename relativeToVRT="1">n45w120.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="10" RasterYSize="10" DataType="Float32" />
      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10" />
      <DstRect xOff="10.0002" yOff="0" xSize="10" ySize="10" />
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"#;

    #[test]
    fn boxes_from_dst_rect_and_geotransform() {
        let ds = Dataset::from_xml(TILES).unwrap();
        let f = footprints(&ds, 1, "https://h", &NoIo).unwrap();
        assert_eq!(f.len(), 2);
        let b = f[0].bounds;
        assert!((b.xmin - -120.99998).abs() < 1e-9);
        assert!((b.ymax - 45.0).abs() < 1e-12 && (b.ymin - 44.0).abs() < 1e-12);
        // The second tile is clipped to the VRT's width.
        assert!((f[1].dst.x_size - 9.9998).abs() < 1e-9);
        assert_eq!(f[1].dsn.to_string(), "https://h/n45w120.tif");
        assert_eq!(f[1].resampling.as_deref(), Some("bilinear"));
    }

    #[test]
    fn crossing_query_is_found() {
        // A tall narrow strip through the first tile: no corner of either box
        // lies inside the other.
        let ds = Dataset::from_xml(TILES).unwrap();
        let q = Bounds::new(-120.6, -120.5, 43.0, 46.0);
        let hit = sources_in_bounds(&ds, 1, &q, "", &NoIo).unwrap();
        assert_eq!(hit.iter().map(|f| f.source).collect::<Vec<_>>(), vec![0]);
        // Touching along an edge is not overlap.
        let q = Bounds::new(-122.0, -120.99998, 44.0, 45.0);
        assert!(sources_in_bounds(&ds, 1, &q, "", &NoIo).unwrap().is_empty());
    }

    #[test]
    fn src_rect_past_the_edge_shrinks_the_footprint() {
        let xml = r#"<VRTDataset rasterXSize="40" rasterYSize="40">
  <VRTRasterBand dataType="Byte" band="1">
    <SimpleSource>
      <SourceFilename>a.tif</SourceFilename>
      <SourceProperties RasterXSize="10" RasterYSize="10" DataType="Byte" />
      <SrcRect xOff="-5" yOff="0" xSize="20" ySize="10" />
      <DstRect xOff="0" yOff="0" xSize="40" ySize="20" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"#;
        let ds = Dataset::from_xml(xml).unwrap();
        let f = &footprints(&ds, 1, "", &NoIo).unwrap()[0];
        // Source columns 0..10 of the requested -5..15, at 2 px per column.
        assert_eq!(
            f.dst,
            Rect {
                x_off: 10.0,
                y_off: 0.0,
                x_size: 20.0,
                y_size: 20.0
            }
        );
        // No geotransform: pixel coordinates, y down, as the reference
        // library's default.
        assert_eq!(f.bounds, Bounds::new(10.0, 30.0, 0.0, 20.0));
    }

    #[test]
    fn rotated_geotransform_gives_the_enclosing_box() {
        let gt = [0.0, 1.0, 1.0, 0.0, 1.0, -1.0];
        let r = Rect {
            x_off: 0.0,
            y_off: 0.0,
            x_size: 2.0,
            y_size: 2.0,
        };
        assert_eq!(rect_bounds(&gt, &r), Bounds::new(0.0, 4.0, -2.0, 2.0));
    }

    #[test]
    fn source_off_the_vrt_is_dropped() {
        let xml = r#"<VRTDataset rasterXSize="10" rasterYSize="10">
  <VRTRasterBand dataType="Byte" band="1">
    <SimpleSource>
      <SourceFilename>a.tif</SourceFilename>
      <DstRect xOff="10" yOff="0" xSize="5" ySize="5" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"#;
        let ds = Dataset::from_xml(xml).unwrap();
        assert!(footprints(&ds, 1, "", &NoIo).unwrap().is_empty());
    }
}
