//! Stage S4b: a time x depth x lat x lon source warps slice by slice under
//! one plan, and every slice equals the 2D warp of that slice alone, for
//! each kernel and route, however the slices are batched.

use async_trait::async_trait;
use wxy_domain::{Affine, DataType, Dim, Domain, Frame, Mapping, Positions};
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Result, Values, Window};
use wxy_warp::{warp, GridSpec, ResampleAlg, WarpOptions, WarpPlan, BATCH_BYTES};

/// An in-memory C-order array, served in chunks of the given shape.
struct Memory {
    grid: ChunkGrid,
    values: Vec<f32>,
}

#[async_trait]
impl ChunkSource for Memory {
    fn grid(&self) -> &ChunkGrid {
        &self.grid
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
        let w = self.grid.window(&e.key);
        let stored = self.grid.chunk.clone();
        let shape = &self.grid.shape;
        let n: u64 = stored.iter().product();
        let mut v = vec![0f32; n as usize];
        for (i, out) in v.iter_mut().enumerate() {
            // index within the stored chunk, then within the array
            let mut rem = i as u64;
            let mut flat = 0u64;
            let mut inside = true;
            for d in 0..stored.len() {
                let stride: u64 = stored[d + 1..].iter().product();
                let c = rem / stride;
                rem %= stride;
                let g = w.offset[d] + c;
                inside &= g < shape[d];
                flat = flat * shape[d] + g.min(shape[d] - 1);
            }
            if inside {
                *out = self.values[flat as usize];
            }
        }
        Ok(Chunk {
            key: e.key,
            window: w,
            stored,
            values: Values::F32(v),
        })
    }
}

fn domain(names: &[&str], shape: &[u64], gt: [f64; 6]) -> Domain {
    let n = shape.len();
    let mut d = Domain::new(
        names
            .iter()
            .zip(shape)
            .map(|(n, &s)| Dim::new(*n, s))
            .collect(),
    );
    let k = d.add_mapping(Mapping::new(
        vec![n - 2, n - 1],
        Positions::Affine(Affine(gt)),
        Frame::crs("EPSG:4326"),
    ));
    d.set_primary(k);
    d
}

/// A 2-degree global field per (time, depth), with a nodata patch.
fn field(nt: u64, nz: u64, ny: u64, nx: u64) -> Vec<f32> {
    let mut v = Vec::new();
    for t in 0..nt {
        for z in 0..nz {
            for r in 0..ny {
                for c in 0..nx {
                    let lon = -179.0 + 2.0 * c as f64;
                    let lat = 89.0 - 2.0 * r as f64;
                    let x =
                        100.0 * (lon.to_radians() * (1 + t) as f64).sin() * lat.to_radians().cos()
                            + 10.0 * z as f64;
                    let hole = (r + 7 * t + 3 * z) % 17 == 0 && c % 5 < 2;
                    v.push(if hole { -9999.0 } else { x as f32 });
                }
            }
        }
    }
    v
}

#[tokio::test]
async fn slices_match_their_2d_warps() {
    let (nt, nz, ny, nx) = (3u64, 2u64, 90u64, 180u64);
    let gt = [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0];
    let values = field(nt, nz, ny, nx);
    let shape = vec![nt, nz, ny, nx];
    let src = Memory {
        grid: ChunkGrid::new(shape.clone(), vec![2, 1, 32, 64]),
        values: values.clone(),
    };
    let d4 = domain(&["time", "depth", "lat", "lon"], &shape, gt);
    let d2 = domain(&["lat", "lon"], &[ny, nx], gt);
    // coarse enough that bilinear and cubic are scaled
    let dst = GridSpec::from_extent("EPSG:3031", [-5e6, 5e6, -5e6, 5e6], [60, 60]);
    let n_dst = 60 * 60;
    for (alg, chunked) in [
        (ResampleAlg::NearestNeighbour, true),
        (ResampleAlg::Bilinear, true),
        (ResampleAlg::Cubic, true),
        (ResampleAlg::Bilinear, false),
    ] {
        let opts = WarpOptions {
            resampling: alg,
            chunked,
            nodata: Some(-9999.0),
            ..Default::default()
        };
        let plan = WarpPlan::new(&d4, DataType::F32, &dst, &opts, None).unwrap();
        assert_eq!(plan.leading(), &[nt, nz]);
        let whole = warp(&src, &d4, &dst, &opts, None).await.unwrap();
        assert_eq!(whole.window.size, vec![nt, nz, 60, 60]);
        let Values::F32(whole) = whole.values else {
            panic!()
        };
        // one slice at a time, and the source's own chunking
        for max in [1, BATCH_BYTES] {
            let batches = plan.batches(&src.grid.chunk, max);
            let want = if max == 1 { 6 } else { 4 };
            assert_eq!(batches.len(), want, "{alg:?} batches under {max}");
            let mut seen = 0;
            for b in &batches {
                let got = plan.warp_slices(&src, b).await.unwrap();
                let Values::F32(got) = got.values else {
                    panic!()
                };
                for (k, slice) in got.chunks(n_dst).enumerate() {
                    let (t, z) = (
                        b.offset[0] + k as u64 / b.size[1],
                        b.offset[1] + k as u64 % b.size[1],
                    );
                    let i = (t * nz + z) as usize;
                    let n2 = (ny * nx) as usize;
                    let one = Memory {
                        grid: ChunkGrid::new(vec![ny, nx], vec![ny, nx]),
                        values: values[i * n2..(i + 1) * n2].to_vec(),
                    };
                    let Values::F32(want) =
                        warp(&one, &d2, &dst, &opts, None).await.unwrap().values
                    else {
                        panic!()
                    };
                    assert_eq!(slice, &want[..], "{alg:?} chunked={chunked} t={t} z={z}");
                    let w = &whole[i * n_dst..(i + 1) * n_dst];
                    let bad: Vec<usize> = (0..n_dst).filter(|&j| slice[j] != w[j]).collect();
                    assert!(
                        bad.is_empty(),
                        "{alg:?} chunked={chunked} t={t} z={z} max={max}: {} differ, first {:?}",
                        bad.len(),
                        bad.iter()
                            .take(3)
                            .map(|&j| (j, slice[j], w[j]))
                            .collect::<Vec<_>>()
                    );
                    seen += 1;
                }
            }
            assert_eq!(seen, nt * nz);
        }
    }
}

#[test]
fn batches_cover_the_leading_dims_once() {
    let d = domain(
        &["t", "z", "y", "x"],
        &[5, 3, 10, 10],
        [0.0, 1.0, 0.0, 10.0, 0.0, -1.0],
    );
    let dst = GridSpec::from_extent("EPSG:4326", [0.0, 10.0, 0.0, 10.0], [10, 10]);
    let plan = WarpPlan::new(&d, DataType::F32, &dst, &WarpOptions::default(), None).unwrap();
    for (chunk, max, n) in [
        (vec![2, 3, 10, 10], BATCH_BYTES, 3),
        (vec![1, 1, 10, 10], BATCH_BYTES, 15),
        (vec![5, 3, 10, 10], BATCH_BYTES, 1),
        (vec![5, 3, 10, 10], 4 * 200 * 4, 6),
    ] {
        let b = plan.batches(&chunk, max);
        assert_eq!(b.len(), n, "{chunk:?} {max}");
        let mut hit = vec![0; 15];
        for w in &b {
            for t in w.offset[0]..w.offset[0] + w.size[0] {
                for z in w.offset[1]..w.offset[1] + w.size[1] {
                    hit[(t * 3 + z) as usize] += 1;
                }
            }
        }
        assert!(hit.iter().all(|&h| h == 1), "{chunk:?} {max}: {hit:?}");
    }
}

#[tokio::test]
async fn a_2d_source_is_one_batch_of_rank_0() {
    let d = domain(&["y", "x"], &[10, 10], [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]);
    let dst = GridSpec::from_extent("EPSG:4326", [0.0, 10.0, 0.0, 10.0], [10, 10]);
    let plan = WarpPlan::new(&d, DataType::F32, &dst, &WarpOptions::default(), None).unwrap();
    let b = plan.batches(&[10, 10], BATCH_BYTES);
    assert_eq!(b, vec![Window::new(vec![], vec![])]);
    let src = Memory {
        grid: ChunkGrid::new(vec![10, 10], vec![4, 4]),
        values: (0..100).map(|i| i as f32).collect(),
    };
    let got = plan.warp_slices(&src, &b[0]).await.unwrap();
    assert_eq!(got.window.size, vec![10, 10]);
    assert_eq!(
        got.values,
        Values::F32((0..100).map(|i| i as f32).collect())
    );
}
