//! Zonal statistics by lazy rasterization (stage S4c; doc 05, E1):
//! polygons become row spans of cells, never a raster; the spans are
//! grouped by the chunks they cross; each chunk is fetched and decoded
//! once and summarised for every polygon over it. Memory is the spans
//! (about the polygons' perimeter in cells, plus one span per row), one
//! batch of chunks, and a chunk-sized weight buffer.
//!
//! Cells come from one of three rules:
//!
//! - [`ZoneCells::Centre`]: a cell whose centre is inside, as GDAL's
//!   rasterizer burns it (`GDALdllImageFilledPolygon`, ported);
//! - [`ZoneCells::AllTouched`]: any cell a polygon touches (GDAL's
//!   `ALL_TOUCHED`);
//! - [`ZoneCells::Coverage`]: every cell a polygon covers, weighted by the
//!   fraction it covers (controlledburn, exactextract's algorithm).

use std::collections::BTreeMap;

use wxy_domain::Domain;
use wxy_source::{ChunkKey, ChunkSource, Error, Result};

use crate::extract::{get, leading_slices, FETCH_BATCH};
use crate::rasterize::{filled_polygon, is_clockwise, line_all_touched, Spans};
use crate::{transformer, Approx, GridSpec, ResampleAlg, SourceGrid, WarpOptions, IDENTITY};

/// A polygon: its rings, exterior first, each closed, as `[x, y]`.
pub type Rings = Vec<Vec<[f64; 2]>>;

/// A cell `(row, col)` and the fraction of it a polygon covers.
pub type CellCover = ([u64; 2], f64);

/// Which cells a polygon counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZoneCells {
    /// Cells whose centre is inside (GDAL's rasterizer).
    #[default]
    Centre,
    /// Every cell touched (GDAL's `ALL_TOUCHED`).
    AllTouched,
    /// Every cell covered, weighted by its covered fraction.
    Coverage,
}

/// A span of cells on one row, `c0..c1` (exclusive), with a weight: 1 for
/// whole cells, the covered fraction for a coverage edge cell, negative
/// for a hole's.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Span {
    row: u32,
    c0: u32,
    c1: u32,
    w: f32,
}

/// Polygon vertices to source index space.
struct Burner {
    t: Approx,
    h: u64,
    w: u64,
}

impl Burner {
    fn new(domain: &Domain, crs: &str) -> Result<Self> {
        let src = SourceGrid::from_domain(domain)?;
        let points = GridSpec {
            crs: crs.into(),
            geotransform: IDENTITY,
            size: [1, 1],
        };
        let opts = WarpOptions {
            resampling: ResampleAlg::NearestNeighbour,
            error_threshold: 0.0,
            wrap_globe: false,
            ..Default::default()
        };
        Ok(Self {
            t: transformer(&src, &points, &opts)?,
            h: src.shape[src.y_dim],
            w: src.shape[src.x_dim],
        })
    }

    /// GDAL's cells for one polygon: rings wound clockwise in their own
    /// CRS (as gdalrasterize.cpp collects them), vertices moved one by one
    /// (no densifying), then the scanline fill.
    fn gdal(&self, rings: &Rings, all_touched: bool) -> Spans {
        let (mut x, mut y, mut parts) = (Vec::new(), Vec::new(), Vec::new());
        for r in rings.iter().filter(|r| !r.is_empty()) {
            let mut push = |p: &[f64; 2]| {
                x.push(p[0]);
                y.push(p[1]);
            };
            if is_clockwise(r) {
                r.iter().for_each(&mut push);
            } else {
                r.iter().rev().for_each(&mut push);
            }
            parts.push(r.len());
        }
        let mut spans = Vec::new();
        if !parts.is_empty() {
            rwarp::Transformer::transform(&self.t, true, &mut x, &mut y);
            let (w, h) = (self.w as i32, self.h as i32);
            if all_touched {
                line_all_touched(w, h, &parts, &x, &y, &mut spans);
            }
            filled_polygon(w, h, &parts, &x, &y, &mut spans);
        }
        spans
    }

    /// Exact cover of every feature (each a set of polygons), from
    /// controlledburn in index space. Each ring is burned alone and holes
    /// subtracted: controlledburn 0.2.1 drops a hole's boundary cells where
    /// the exterior covers them (a patch is with hypertidy). The holes of a
    /// valid polygon lie inside its exterior and apart, so the signed sum
    /// is exact.
    fn coverage(&self, features: &[Vec<Rings>]) -> Result<Vec<Vec<Span>>> {
        use controlledburn::{burn, BurnOptions, Coord, Geometry, GridSpec as BurnGrid, Polygon};
        // index space with y up: (pixel, -line) on a grid of unit cells
        let grid = BurnGrid::new(
            0.0,
            -(self.h as f64),
            self.w as f64,
            0.0,
            self.w as u32,
            self.h as u32,
        );
        let (mut geoms, mut owner) = (Vec::new(), Vec::new());
        for (k, polys) in features.iter().enumerate() {
            for rings in polys {
                for (i, r) in rings.iter().enumerate() {
                    let (mut x, mut y): (Vec<f64>, Vec<f64>) =
                        r.iter().map(|p| (p[0], p[1])).unzip();
                    rwarp::Transformer::transform(&self.t, true, &mut x, &mut y);
                    let ring = x.iter().zip(&y).map(|(&x, &y)| Coord::new(x, -y)).collect();
                    geoms.push(Geometry::Polygon(Polygon::new(vec![ring])));
                    owner.push((k, if i == 0 { 1.0f32 } else { -1.0 }));
                }
            }
        }
        let r = burn(&geoms, &grid, BurnOptions::coverage())
            .map_err(|e| Error::Unsupported(format!("coverage: {e}")))?;
        let mut out = vec![Vec::new(); features.len()];
        for run in &r.runs {
            let (k, sign) = owner[run.id as usize];
            out[k].push(Span {
                row: run.row as u32,
                c0: run.col_start as u32,
                c1: run.col_end as u32,
                w: sign,
            });
        }
        for e in &r.edges {
            let (k, sign) = owner[e.id as usize];
            out[k].push(Span {
                row: e.row as u32,
                c0: e.col as u32,
                c1: e.col as u32 + 1,
                w: sign * e.fraction,
            });
        }
        Ok(out)
    }
}

/// The cells of a source each polygon covers, as GDAL's rasterizer burns
/// them for that polygon alone: a cell whose centre is inside, or with
/// `all_touched` any cell the polygon touches. Vertices in `crs` go to
/// the source one by one, as gdal_rasterize reprojects them. Cells are
/// `(row, col)`, sorted.
pub fn polygon_cells(
    domain: &Domain,
    crs: &str,
    polygons: &[Rings],
    all_touched: bool,
) -> Result<Vec<Vec<[u64; 2]>>> {
    let b = Burner::new(domain, crs)?;
    Ok(polygons
        .iter()
        .map(|rings| {
            let mut cells: Vec<[u64; 2]> = b
                .gdal(rings, all_touched)
                .into_iter()
                .flat_map(|(r, c0, c1)| (c0..=c1).map(move |c| [r as u64, c as u64]))
                .collect();
            cells.sort_unstable();
            cells.dedup();
            cells
        })
        .collect())
}

/// The cells of a source each polygon covers and the fraction of each it
/// covers (in `(0, 1]`), from controlledburn's exact coverage. The
/// fractions are taken in the source's index space: exact areas for an
/// affine source, index-space areas for a curvilinear one.
pub fn polygon_coverage(
    domain: &Domain,
    crs: &str,
    polygons: &[Rings],
) -> Result<Vec<Vec<CellCover>>> {
    let b = Burner::new(domain, crs)?;
    let features: Vec<Vec<Rings>> = polygons.iter().map(|p| vec![p.clone()]).collect();
    Ok(b.coverage(&features)?
        .into_iter()
        .map(|spans| {
            let mut acc: BTreeMap<[u64; 2], f64> = BTreeMap::new();
            for s in spans {
                for c in s.c0..s.c1 {
                    *acc.entry([s.row as u64, c as u64]).or_insert(0.0) += s.w as f64;
                }
            }
            acc.into_iter()
                .filter(|&(_, f)| f > 1e-6)
                .map(|(c, f)| (c, f.min(1.0)))
                .collect()
        })
        .collect())
}

/// Summary of one zone over one leading slice. Counts and sums are
/// weighted by cover (1 for whole cells); nodata and NaN are left out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoneStats {
    /// Cells in the zone (weighted by cover), nodata or not.
    pub cells: f64,
    /// Cells with a value (weighted by cover).
    pub count: f64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

impl ZoneStats {
    fn new() -> Self {
        Self {
            cells: 0.0,
            count: 0.0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn mean(&self) -> Option<f64> {
        (self.count > 0.0).then(|| self.sum / self.count)
    }
}

/// Zones (features, each one or more polygons) as row spans grouped by
/// the source chunks they cross.
pub struct ZonePlan {
    shape: Vec<u64>,
    chunk: Vec<u64>,
    cells: ZoneCells,
    zones: usize,
    /// Per spatial chunk `[row, col]`: (zone, span clipped to the chunk).
    by_chunk: BTreeMap<[u64; 2], Vec<(u32, Span)>>,
}

/// Plan zonal statistics of `features` (each a set of polygons in `crs`)
/// over a source with `domain`, chunked as `chunk`.
pub fn plan_zones(
    domain: &Domain,
    chunk: &[u64],
    crs: &str,
    features: &[Vec<Rings>],
    cells: ZoneCells,
) -> Result<ZonePlan> {
    let b = Burner::new(domain, crs)?;
    let shape = domain.shape();
    if chunk.len() != shape.len() || shape.len() < 2 {
        return Err(Error::Rank {
            got: chunk.len(),
            want: shape.len(),
        });
    }
    let n = shape.len();
    let (ch, cw) = (chunk[n - 2], chunk[n - 1]);
    let spans: Vec<Vec<Span>> = match cells {
        ZoneCells::Coverage => b.coverage(features)?,
        _ => features
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .flat_map(|rings| b.gdal(rings, cells == ZoneCells::AllTouched))
                    .map(|(row, c0, c1)| Span {
                        row,
                        c0,
                        c1: c1 + 1,
                        w: 1.0,
                    })
                    .collect()
            })
            .collect(),
    };
    let mut by_chunk: BTreeMap<[u64; 2], Vec<(u32, Span)>> = BTreeMap::new();
    for (k, zone) in spans.iter().enumerate() {
        for s in zone {
            let cy = s.row as u64 / ch;
            let mut c = s.c0 as u64;
            while c < s.c1 as u64 {
                let cx = c / cw;
                let end = ((cx + 1) * cw).min(s.c1 as u64);
                by_chunk.entry([cy, cx]).or_default().push((
                    k as u32,
                    Span {
                        c0: c as u32,
                        c1: end as u32,
                        ..*s
                    },
                ));
                c = end;
            }
        }
    }
    Ok(ZonePlan {
        shape,
        chunk: chunk.to_vec(),
        cells,
        zones: features.len(),
        by_chunk,
    })
}

impl ZonePlan {
    /// The leading dims' sizes: statistics come per zone per leading slice.
    pub fn leading(&self) -> &[u64] {
        &self.shape[..self.shape.len() - 2]
    }

    /// Every chunk the zones need, each once.
    pub fn keys(&self) -> Vec<ChunkKey> {
        let lead = self.shape.len() - 2;
        let counts: Vec<u64> = (0..lead)
            .map(|d| self.shape[d].div_ceil(self.chunk[d]))
            .collect();
        let n_lead: u64 = counts.iter().product();
        let mut keys = Vec::new();
        for g in self.by_chunk.keys() {
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

    /// Spans held in the plan (its memory, in units of 20 bytes).
    pub fn spans(&self) -> usize {
        self.by_chunk.values().map(Vec::len).sum()
    }
}

/// Zonal statistics: `zones x leading` in C order (zone major), each chunk
/// fetched once.
pub async fn zonal_stats<S: ChunkSource + ?Sized>(
    source: &S,
    plan: &ZonePlan,
    nodata: Option<f64>,
) -> Result<Vec<ZoneStats>> {
    if source.grid().shape != plan.shape || source.grid().chunk != plan.chunk {
        return Err(Error::Unsupported(
            "source grid differs from the planned one".into(),
        ));
    }
    let lead = plan.shape.len() - 2;
    let leading = plan.leading().to_vec();
    let n_lead = leading.iter().product::<u64>() as usize;
    let mut stats = vec![ZoneStats::new(); plan.zones * n_lead];
    let (ch, cw) = (plan.chunk[lead] as usize, plan.chunk[lead + 1] as usize);
    // one chunk's weights for one zone, and the cells it set
    let mut weight = vec![0f32; ch * cw];
    let mut seen = vec![false; ch * cw];
    let mut touched: Vec<usize> = Vec::new();
    let is_nodata = |v: f64| v.is_nan() || nodata == Some(v);
    let keys = plan.keys();
    for batch in keys.chunks(FETCH_BATCH) {
        for e in source.fetch(batch).await? {
            let chunk = source.decode(e)?;
            let key = &chunk.key.0;
            let lw = &chunk.window;
            let (r0, c0) = (lw.offset[lead], lw.offset[lead + 1]);
            let stride = chunk.stored[lead + 1] as usize;
            let slices = leading_slices(&chunk, lead, &leading);
            let mut spans = plan.by_chunk[&[key[lead], key[lead + 1]]].clone();
            spans.sort_by_key(|&(z, _)| z);
            for zone in spans.chunk_by(|a, b| a.0 == b.0) {
                let z = zone[0].0 as usize;
                for &(_, s) in zone {
                    let r = s.row as usize - r0 as usize;
                    for c in s.c0 as usize..s.c1 as usize {
                        let i = r * cw + (c - c0 as usize);
                        if !seen[i] {
                            seen[i] = true;
                            touched.push(i);
                        }
                        if plan.cells == ZoneCells::Coverage {
                            weight[i] += s.w;
                        } else {
                            weight[i] = 1.0;
                        }
                    }
                }
                for &(lead_src, l) in &slices {
                    let st = &mut stats[z * n_lead + l];
                    for &i in &touched {
                        let f = (weight[i] as f64).min(1.0);
                        if f <= 1e-6 {
                            continue;
                        }
                        let (r, c) = (i / cw, i % cw);
                        let v = get(&chunk.values, lead_src as usize + r * stride + c);
                        st.cells += f;
                        if is_nodata(v) {
                            continue;
                        }
                        st.count += f;
                        st.sum += v * f;
                        st.min = st.min.min(v);
                        st.max = st.max.max(v);
                    }
                }
                for &i in &touched {
                    weight[i] = 0.0;
                    seen[i] = false;
                }
                touched.clear();
            }
        }
    }
    Ok(stats)
}
