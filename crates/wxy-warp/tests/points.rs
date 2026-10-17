//! Stage S4c: point reads planned by chunk match GDAL's point reads
//! (gdallocationinfo's cell: reproject, inverse geotransform, floor) on
//! the COG fixtures (wxylib/fixtures, points/), and each chunk is fetched
//! once.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use wxy_domain::DataType;
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result};
use wxy_warp::{extract_points, plan_points};

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
        let plan = plan_points(
            domain,
            &source.grid().chunk,
            case["crs"].as_str().unwrap(),
            &x,
            &y,
            false,
        )
        .unwrap();
        let counting = Counting(source, Mutex::new(Vec::new()));
        let got = extract_points(&counting, &plan).await.unwrap();

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
        eprintln!(
            "{name}: {} of {n} points on the file, {} chunks fetched",
            found.iter().filter(|&&f| f).count(),
            fetched.len()
        );
    }
}
