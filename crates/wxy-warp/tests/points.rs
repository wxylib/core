//! Stage S4c: point reads planned by chunk match GDAL's point reads on
//! the COG fixtures (wxylib/fixtures, points/): gdallocationinfo's cell
//! (reproject, inverse geotransform, floor) for nearest, and
//! InterpolateAtPoint at GDAL's own positions for bilinear, cubic and
//! cubic spline, with our positions bounded against GDAL's. Each chunk is
//! fetched once.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use wxy_domain::DataType;
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result};
use wxy_warp::{extract_points, plan_pixels, plan_points, PointMethod, PointOptions};

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
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[tokio::test]
async fn point_reads_match_gdal() {
    let dir = fixtures().join("points");
    let Ok(file) = std::fs::File::open(dir.join("manifest.json")) else {
        eprintln!("skipping: no point fixtures at {}", dir.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let mut bad = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let read = |s: &str| std::fs::read(dir.join(format!("{name}-{s}.bin"))).unwrap();
        let (x, y) = (f64s(&read("x")), f64s(&read("y")));
        let want = f64s(&read("values"));
        let found: Vec<bool> = read("found").iter().map(|&b| b == 1).collect();

        let ds = wxy_tiff::open_path(fixtures().join(case["source"].as_str().unwrap()))
            .await
            .unwrap();
        let source = &ds.levels[0];
        let domain = &ds.collection.domains[0];
        let nodata = ds.collection.fields[0].values.nodata;
        let chunk = &source.grid().chunk;
        let crs = case["crs"].as_str().unwrap();
        let opts = |method| PointOptions {
            method,
            wrap_globe: false,
        };
        let plan = plan_points(domain, chunk, crs, &x, &y, &opts(PointMethod::Nearest)).unwrap();
        let counting = Counting(source, Mutex::new(Vec::new()));
        let got = extract_points(&counting, &plan, nodata).await.unwrap();

        assert_eq!(got.found, found, "{name}: which points are on the file");
        let got_v = got.values.to_f64();
        let n = x.len();
        let bands = case["bands"].as_u64().unwrap() as usize;
        assert_eq!(got_v.len(), bands * n, "{name}");
        let differ = (0..bands * n)
            .filter(|&i| found[i % n] && got_v[i] != want[i])
            .count();
        assert_eq!(differ, 0, "{name}: values differ from GDAL's");

        let fetched = counting.1.into_inner().unwrap();
        let unique: HashSet<_> = fetched.iter().cloned().collect();
        assert_eq!(unique.len(), fetched.len(), "{name}: a chunk fetched twice");
        assert_eq!(fetched.len(), plan.keys().len(), "{name}");

        // GDAL's InterpolateAtPoint, NaN where GDAL has no value. At
        // GDAL's own (pixel, line) every value is identical; from the
        // points, positions differ from GDAL's by what separates proj4rs
        // from PROJ, bounded here.
        let (gp, gl) = (f64s(&read("pixel")), f64s(&read("line")));
        let plan = plan_points(domain, chunk, crs, &x, &y, &opts(PointMethod::Bilinear)).unwrap();
        let mut shift = 0.0f64;
        for (i, p) in plan.positions.iter().enumerate() {
            if let Some([px, py]) = p {
                shift = shift.max((px - gp[i]).abs()).max((py - gl[i]).abs());
            }
        }
        assert!(shift < 1e-8, "{name}: positions {shift} px from GDAL's");
        for (m, method) in [
            ("bilinear", PointMethod::Bilinear),
            ("cubic", PointMethod::Cubic),
            ("cubicspline", PointMethod::CubicSpline),
        ] {
            let want = f64s(&read(m));
            let plan = plan_pixels(domain, chunk, &gp, &gl, &opts(method)).unwrap();
            let counting = Counting(source, Mutex::new(Vec::new()));
            let got = extract_points(&counting, &plan, nodata).await.unwrap();
            let got = got.values.to_f64();
            let differ = (0..bands * n)
                .filter(|&i| got[i] != want[i] && !(got[i].is_nan() && want[i].is_nan()))
                .count();
            eprintln!("{name} {m}: {differ} of {} differ", bands * n);
            bad += differ;
            let fetched = counting.1.into_inner().unwrap();
            let unique: HashSet<_> = fetched.iter().cloned().collect();
            assert_eq!(
                unique.len(),
                fetched.len(),
                "{name} {m}: a chunk fetched twice"
            );
        }
        eprintln!("{name}: positions within {shift:e} px of GDAL's");
        eprintln!(
            "{name}: {} of {n} points on the file, {} chunks fetched",
            found.iter().filter(|&&f| f).count(),
            plan.keys().len()
        );
    }
    assert_eq!(bad, 0, "interpolated reads differ from GDAL's");
}
