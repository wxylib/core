//! Stage S5b: warps from a curvilinear source (2D longitude and latitude
//! arrays at cell centres, or at cell corners) match GDAL's warper with its geolocation-array
//! transformer, cell for cell (wxylib/fixtures, curvilinear/).

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;
use wxy_domain::{CellConvention, CoordArray, DataType, Dim, Domain, Frame, Mapping, Positions};
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result, Values};
use wxy_warp::{warp, GridSpec, ResampleAlg, WarpOptions};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

/// One 2D float32 array as a single-chunk source.
struct Memory(ChunkGrid, Vec<f32>);

#[async_trait]
impl ChunkSource for Memory {
    fn grid(&self) -> &ChunkGrid {
        &self.0
    }
    fn dtype(&self) -> DataType {
        DataType::F32
    }
    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
        Ok(keys
            .iter()
            .map(|k| Encoded {
                key: k.clone(),
                payload: Box::new(()),
            })
            .collect())
    }
    fn decode(&self, e: Encoded) -> Result<Chunk> {
        Ok(Chunk {
            window: self.0.window(&e.key),
            key: e.key,
            stored: self.0.chunk.clone(),
            values: Values::F32(self.1.clone()),
        })
    }
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[tokio::test]
async fn curvilinear_warps_match_gdal() {
    let dir = fixtures().join("curvilinear");
    let Ok(file) = std::fs::File::open(dir.join("manifest.json")) else {
        eprintln!("skipping: no curvilinear fixtures at {}", dir.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let mut bad = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let base = case["source"].as_str().unwrap();
        let shape: Vec<usize> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let (ny, nx) = (shape[0], shape[1]);
        let (cell, e) = match case["cell"].as_str() {
            Some("edge") => (CellConvention::Edge, 1),
            _ => (CellConvention::Centre, 0),
        };
        let read = |s: &str| std::fs::read(dir.join(format!("{base}-{s}.bin"))).unwrap();
        let lon = CoordArray::new(vec![ny + e, nx + e], f64s(&read("lon")));
        let lat = CoordArray::new(vec![ny + e, nx + e], f64s(&read("lat")));
        let values = f32s(&read("values"));
        let mut domain = Domain::new(vec![Dim::new("y", ny as u64), Dim::new("x", nx as u64)]);
        let k = domain.add_mapping(
            Mapping::new(
                vec![0, 1],
                Positions::Curvilinear(vec![lon, lat]),
                Frame::crs(case["source_crs"].as_str().unwrap()),
            )
            .with_cell(cell),
        );
        domain.set_primary(k);
        let src = Memory(
            ChunkGrid::new(vec![ny as u64, nx as u64], vec![ny as u64, nx as u64]),
            values,
        );
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let dst = GridSpec {
            crs: case["crs"].as_str().unwrap().into(),
            geotransform: gt.try_into().unwrap(),
            size: [
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
            ],
        };
        for r in case["results"].as_array().unwrap() {
            let m = r["resampling"].as_str().unwrap();
            let alg = match m {
                "near" => ResampleAlg::NearestNeighbour,
                "bilinear" => ResampleAlg::Bilinear,
                "cubic" => ResampleAlg::Cubic,
                o => panic!("{o}"),
            };
            let opts = WarpOptions {
                resampling: alg,
                ..Default::default()
            };
            let got = warp(&src, &domain, &dst, &opts, None).await.unwrap();
            let Values::F32(got) = got.values else {
                panic!()
            };
            let want = f32s(&std::fs::read(dir.join(r["file"].as_str().unwrap())).unwrap());
            let diff: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
            let max = diff
                .iter()
                .map(|&i| (got[i] - want[i]).abs())
                .fold(0f32, f32::max);
            let filled = |v: &[f32]| v.iter().filter(|&&x| x != 0.0).count();
            eprintln!(
                "{name} {m}: {} of {} cells differ (max {max}); filled {} vs GDAL {}",
                diff.len(),
                want.len(),
                filled(&got),
                filled(&want)
            );
            if !diff.is_empty() {
                bad.push(format!("{name} {m}"));
            }
        }
    }
    assert!(bad.is_empty(), "differ from GDAL: {bad:?}");
}
