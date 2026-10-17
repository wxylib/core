//! Plans as values: a plan's `explain` lists exactly the chunks its run
//! then fetches, in order, and round-trips through JSON; options do too.

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use wxy_domain::{DataType, Positions};
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result};
use wxy_warp::{
    extract_points, plan_points, plan_zones, zonal_stats, Detail, Explain, GridSpec, PointMethod,
    PointOptions, ResampleAlg, WarpOptions, WarpPlan, ZoneCells,
};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

/// A source that records every key it is asked to fetch.
struct Counting<'a, S>(&'a S, Mutex<Vec<ChunkKey>>);

#[async_trait]
impl<S: ChunkSource> ChunkSource for Counting<'_, S> {
    fn grid(&self) -> &ChunkGrid {
        self.0.grid()
    }
    fn dtype(&self) -> DataType {
        self.0.dtype()
    }
    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
        self.1.lock().unwrap().extend_from_slice(keys);
        self.0.fetch(keys).await
    }
    fn decode(&self, e: Encoded) -> Result<Chunk> {
        self.0.decode(e)
    }
    fn stored_bytes(&self, key: &ChunkKey) -> Option<u64> {
        self.0.stored_bytes(key)
    }
}

impl<S> Counting<'_, S> {
    fn take(&self) -> Vec<Vec<u64>> {
        std::mem::take(&mut *self.1.lock().unwrap())
            .into_iter()
            .map(|k| k.0)
            .collect()
    }
}

fn round_trip(e: &Explain) {
    let json = serde_json::to_string(e).unwrap();
    let back: Explain = serde_json::from_str(&json).unwrap();
    assert_eq!(&back, e);
}

#[tokio::test]
async fn explain_lists_what_the_run_fetches() {
    let path = fixtures().join("cog/planar-u8-deflate-2band.tif");
    if !path.exists() {
        eprintln!("skipping: no fixture at {}", path.display());
        return;
    }
    let ds = wxy_tiff::open_path(&path).await.unwrap();
    let (level, domain) = (&ds.levels[0], &ds.collection.domains[0]);
    let chunk = &level.grid().chunk;
    let shape = domain.shape();
    let (h, w) = (shape[shape.len() - 2] as f64, shape[shape.len() - 1] as f64);
    let m = domain.primary_for(shape.len() - 1).unwrap();
    let gt = match &m.positions {
        Positions::Affine(a) => a.0,
        _ => unreachable!(),
    };
    // positions in the source's own CRS
    let wxy_domain::Frame::Crs(crs) = &m.frame else {
        unreachable!()
    };
    let crs = crs.as_str();
    let at = |c: f64, r: f64| [gt[0] + c * gt[1], gt[3] + r * gt[5]];
    let counting = Counting(level, Mutex::new(Vec::new()));

    // points: a diagonal across the source, some off it
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for i in 0..50 {
        let f = i as f64 / 40.0 - 0.1;
        let p = at(f * w, (1.0 - f) * h);
        x.push(p[0]);
        y.push(p[1]);
    }
    let options = PointOptions {
        method: PointMethod::Bilinear,
        wrap_globe: false,
    };
    let plan = plan_points(domain, chunk, crs, &x, &y, &options).unwrap();
    let e = plan.explain(&counting, &options);
    extract_points(&counting, &plan, None).await.unwrap();
    assert_eq!(e.fetch.keys, counting.take(), "points");
    assert!(e.fetch.stored_bytes.unwrap() > 0);
    let Detail::Points {
        points, on_source, ..
    } = e.detail
    else {
        panic!("not a points plan")
    };
    assert_eq!(points, 50);
    assert!(on_source > 0 && on_source < 50);
    round_trip(&e);

    // zonal: two boxes, one crossing the source's edge
    let square = |c0: f64, r0: f64, c1: f64, r1: f64| {
        vec![vec![
            at(c0, r0),
            at(c1, r0),
            at(c1, r1),
            at(c0, r1),
            at(c0, r0),
        ]]
    };
    let features = vec![
        vec![square(0.1 * w, 0.2 * h, 0.4 * w, 0.5 * h)],
        vec![square(0.7 * w, 0.6 * h, 1.3 * w, 0.9 * h)],
    ];
    for cells in [
        ZoneCells::Centre,
        ZoneCells::AllTouched,
        ZoneCells::Coverage,
    ] {
        let plan = plan_zones(domain, chunk, crs, &features, cells).unwrap();
        let e = plan.explain(&counting);
        zonal_stats(&counting, &plan, None).await.unwrap();
        assert_eq!(e.fetch.keys, counting.take(), "zonal {cells:?}");
        assert!(e.fetch.chunks < e.fetch.of as usize);
        round_trip(&e);
    }

    // warp: a target over part of the source, each kernel and route, with
    // batches small enough to split the two bands
    let dst = GridSpec {
        crs: crs.into(),
        geotransform: [
            at(0.2 * w, 0.0)[0],
            gt[1] * 1.7,
            0.0,
            at(0.0, 0.3 * h)[1],
            0.0,
            gt[5] * 1.7,
        ],
        size: [(0.3 * w) as usize, (0.3 * h) as usize],
    };
    for (resampling, chunked) in [
        (ResampleAlg::NearestNeighbour, true),
        (ResampleAlg::Bilinear, true),
        (ResampleAlg::Cubic, false),
    ] {
        let opts = WarpOptions {
            resampling,
            chunked,
            ..Default::default()
        };
        let plan = WarpPlan::new(domain, level.dtype(), &dst, &opts, None).unwrap();
        for max_bytes in [1, wxy_warp::BATCH_BYTES] {
            let e = plan.explain(&counting, max_bytes);
            let Detail::Warp { batches, .. } = &e.detail else {
                panic!("not a warp plan")
            };
            let windows = plan.batches(chunk, max_bytes);
            assert_eq!(batches.len(), windows.len());
            for b in &windows {
                plan.warp_slices(&counting, b).await.unwrap();
            }
            assert_eq!(
                e.fetch.keys,
                counting.take(),
                "warp {resampling:?} {chunked}"
            );
            assert!(!e.fetch.keys.is_empty());
            round_trip(&e);
        }
    }
}

#[test]
fn options_round_trip_and_default() {
    let o = WarpOptions {
        resampling: ResampleAlg::Cubic,
        overview: wxy_warp::Overview::AutoFiner(1),
        ..Default::default()
    };
    let json = serde_json::to_string(&o).unwrap();
    assert_eq!(serde_json::from_str::<WarpOptions>(&json).unwrap(), o);
    // missing fields take their defaults
    let partial: WarpOptions = serde_json::from_str(r#"{"resampling": "bilinear"}"#).unwrap();
    assert_eq!(
        partial,
        WarpOptions {
            resampling: ResampleAlg::Bilinear,
            ..Default::default()
        }
    );
    assert!(serde_json::from_str::<WarpOptions>(r#"{"resampling": "nearest"}"#).is_err());
    let p: PointOptions = serde_json::from_str(r#"{"method": "cubicspline"}"#).unwrap();
    assert_eq!(p.method, PointMethod::CubicSpline);
    assert!(p.wrap_globe);
    let z: ZoneCells = serde_json::from_str(r#""all_touched""#).unwrap();
    assert_eq!(z, ZoneCells::AllTouched);
}
