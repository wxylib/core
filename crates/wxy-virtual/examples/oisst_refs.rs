//! P3 demo (design doc m02): 30 days of real NOAA OISST v2.1 (September
//! 1981) from an mdim-refs table, remapped to the public NOAA bucket over
//! HTTPS, written to a local Icechunk repository as virtual references,
//! and read back for day 30. Each array's bytes are compared with what
//! GDAL read from the source netCDF (SHA-256 in the table's manifest).
//!
//! The table is made by wxylib/fixtures
//! `scripts/make_mdimrefs_oisst_198109.py`. Needs network access; not a
//! test.
//!
//! ```sh
//! cargo run -p wxy-virtual --example oisst_refs -- ../fixtures/mdimrefs/oisst-198109
//! ```
//!
//! With a second argument, day 30 `sst` is also warped to EPSG:3031 (a
//! 400 x 400 grid of 20 km cells, nearest) and written there as raw
//! little-endian int16, to compare with `gdalwarp`. The 4D array is
//! sliced to 2D in memory here, since wxy-warp takes 2D or 3D domains.

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use wxy_domain::{Affine, CellConvention, DataType, Dim, Domain, Mapping, Positions};
use wxy_source::{
    read_window, Buffer, Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result, Values, Window,
};

/// One 2D buffer as a single-chunk source.
struct Memory(ChunkGrid, Values);

#[async_trait]
impl ChunkSource for Memory {
    fn grid(&self) -> &ChunkGrid {
        &self.0
    }
    fn dtype(&self) -> DataType {
        self.1.dtype()
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
            values: self.1.clone(),
        })
    }
}

/// Warp one (1, 1, y, x) day to EPSG:3031 and write the raw values.
async fn warp_day(buf: Buffer, domain: &Domain, nodata: Option<f64>, out: &std::path::Path) {
    let n = domain.shape().len();
    let m = domain.primary_for(n - 1).unwrap();
    let Positions::Affine(a) = &m.positions else {
        panic!("expected an affine")
    };
    let (ny, nx) = (buf.window.size[2], buf.window.size[3]);
    let mut d2 = Domain::new(vec![Dim::new("lat", ny), Dim::new("lon", nx)]);
    let k = d2.add_mapping(
        Mapping::new(vec![0, 1], Positions::Affine(Affine(a.0)), m.frame.clone())
            .with_cell(CellConvention::Edge),
    );
    d2.set_primary(k);
    let src = Memory(ChunkGrid::new(vec![ny, nx], vec![ny, nx]), buf.values);
    let dst = wxy_warp::GridSpec::from_extent("EPSG:3031", [-4e6, 4e6, -4e6, 4e6], [400, 400]);
    let w = wxy_warp::warp(&src, &d2, &dst, &wxy_warp::WarpOptions::default(), nodata)
        .await
        .unwrap();
    std::fs::write(out, w.values.to_le_bytes()).unwrap();
    println!(
        "sst: day 30 warped to EPSG:3031, 400 x 400, in {}",
        out.display()
    );
}

#[tokio::main]
async fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../fixtures/mdimrefs/oisst-198109"));
    let warp_out = std::env::args().nth(2).map(PathBuf::from);
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(dir.join("manifest.json")).unwrap()).unwrap();
    let target = manifest["remap_target"].as_str();

    let arrays = wxy_virtual::from_refs_table(&dir, target).unwrap();
    let prefixes = wxy_virtual::prefixes(&arrays);
    println!("{} arrays, chunk containers {prefixes:?}", arrays.len());

    let repo = std::env::temp_dir().join(format!("wxy-oisst-refs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    wxy_virtual::write_icechunk(&repo, &arrays).await.unwrap();
    let storage = wxy_virtual::open_icechunk(&repo, &prefixes).await.unwrap();

    let mut ok = true;
    for (name, o) in manifest["oracle"].as_object().unwrap() {
        let v = |k: &str| -> Vec<u64> {
            o[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let ds = wxy_zarr::open(storage.clone(), name).await.unwrap();
        let t0 = std::time::Instant::now();
        let buf = read_window(&ds.source, &Window::new(v("start"), v("count")))
            .await
            .unwrap();
        let got: String = Sha256::digest(buf.values.to_le_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let same = got == o["sha256"].as_str().unwrap();
        ok &= same;
        println!(
            "{name}: day 30 {} GDAL ({:.2} s)",
            if same { "identical to" } else { "DIFFERS from" },
            t0.elapsed().as_secs_f64()
        );
        if let (Some(out), "sst") = (&warp_out, name.as_str()) {
            let d = &ds.collection.domains[0];
            warp_day(buf, d, ds.collection.fields[0].values.nodata, out).await;
        }
    }
    let _ = std::fs::remove_dir_all(&repo);
    if !ok {
        std::process::exit(1);
    }
}
