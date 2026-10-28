//! Lanczos, average and mode, which run rwarp's kernels on each chunk of
//! the chunked warp, and fixed kernel scales (gdalwarp's XSCALE/YSCALE).
//! GDAL oracles for these wait on new fixtures; these tests check the
//! properties the methods promise: tiles warped separately agree with the
//! whole, mode keeps source values, and the exact footprint agrees with
//! GDAL's box where the box is exact.

use std::path::PathBuf;

use wxy_warp::{warp, Footprint, GridSpec, ResampleAlg, WarpOptions};

/// Cells that differ: how many, and the first few as (index, a, b).
fn diff(a: &[f64], b: &[f64]) -> (usize, Vec<(usize, f64, f64)>) {
    assert_eq!(a.len(), b.len());
    let d: Vec<(usize, f64, f64)> = (0..a.len())
        .filter(|&i| a[i] != b[i])
        .map(|i| (i, a[i], b[i]))
        .collect();
    (d.len(), d.into_iter().take(8).collect())
}

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

async fn world() -> Option<wxy_tiff::TiffDataset> {
    let root = fixtures();
    let ds = wxy_tiff::open_path(root.join("cog/world-int16-deflate.tif")).await;
    if ds.is_err() {
        eprintln!("skipping: no fixtures at {}", root.display());
    }
    ds.ok()
}

async fn run(ds: &wxy_tiff::TiffDataset, dst: &GridSpec, opts: &WarpOptions) -> Vec<f64> {
    warp(&ds.levels[0], &ds.collection.domains[0], dst, opts, None)
        .await
        .unwrap()
        .values
        .to_f64()
}

/// The world at half a degree to a 200 by 200 polar target, whole and as
/// top and bottom halves warped on their own, the halves stacked.
///
/// The target is shifted a quarter cell so that no cell edge lies on the
/// antimeridian (x = 0, y < 0), and split across it rather than along it:
/// GDAL's box footprint (and rwarp's port) detects a cell straddling the
/// antimeridian only when the chunk has a cell to its right (OSGeo/gdal
/// #6478), so a tile whose last column straddles it averages the whole
/// source row there instead.
async fn whole_and_halves(ds: &wxy_tiff::TiffDataset, opts: &WarpOptions) -> (Vec<f64>, Vec<f64>) {
    let (n, e, s) = (200, 4.0e6, 1.0e4);
    let whole = GridSpec::from_extent("EPSG:3031", [s - e, s + e, -e, e], [n, n]);
    let top = GridSpec::from_extent("EPSG:3031", [s - e, s + e, 0.0, e], [n, n / 2]);
    let bottom = GridSpec::from_extent("EPSG:3031", [s - e, s + e, -e, 0.0], [n, n / 2]);
    let a = run(ds, &whole, opts).await;
    let mut joined = run(ds, &top, opts).await;
    joined.extend(run(ds, &bottom, opts).await);
    (a, joined)
}

/// Average and mode read each target cell's own footprint, so a tile
/// warped on its own gives the cells the whole warp gives.
#[tokio::test]
async fn average_and_mode_tiles_are_seamless() {
    let Some(ds) = world().await else { return };
    for alg in [ResampleAlg::Average, ResampleAlg::Mode] {
        let opts = WarpOptions {
            resampling: alg,
            // exact positions, so the approximate transformer's scanline
            // spans (which differ between whole and halves) do not matter
            error_threshold: 0.0,
            wrap_globe: false,
            ..Default::default()
        };
        let (whole, joined) = whole_and_halves(&ds, &opts).await;
        assert!(whole.iter().any(|&v| v != 0.0), "{alg:?}: empty warp");
        let (n, first) = diff(&whole, &joined);
        assert_eq!(n, 0, "{alg:?}: halves differ from the whole at {first:?}");
    }
}

/// With one kernel scale for every chunk, interpolating kernels give
/// separately warped tiles the values of the whole warp.
#[tokio::test]
async fn a_fixed_kernel_scale_keeps_tiles_seamless() {
    let Some(ds) = world().await else { return };
    for alg in [
        ResampleAlg::Bilinear,
        ResampleAlg::Cubic,
        ResampleAlg::Lanczos,
    ] {
        let opts = WarpOptions {
            resampling: alg,
            kernel_scale: Some([0.5, 0.5]),
            error_threshold: 0.0,
            wrap_globe: false,
            ..Default::default()
        };
        let (whole, joined) = whole_and_halves(&ds, &opts).await;
        let (n, first) = diff(&whole, &joined);
        assert_eq!(n, 0, "{alg:?}: halves differ from the whole at {first:?}");
    }
}

/// Mode picks among the values present; it never blends them.
#[tokio::test]
async fn mode_keeps_source_values() {
    let Some(ds) = world().await else { return };
    let src = wxy_source::read_window(
        &ds.levels[0],
        &wxy_source::Window::new(vec![0, 0, 0], vec![1, 360, 720]),
    )
    .await
    .unwrap()
    .values
    .to_f64();
    let mut seen: Vec<i64> = src.iter().map(|&v| v as i64).collect();
    seen.sort_unstable();
    seen.dedup();
    let dst = GridSpec::from_extent("EPSG:3031", [-4.0e6, 4.0e6, -4.0e6, 4.0e6], [60, 60]);
    let opts = WarpOptions {
        resampling: ResampleAlg::Mode,
        wrap_globe: false,
        ..Default::default()
    };
    for v in run(&ds, &dst, &opts).await {
        assert!(
            v == 0.0 || seen.binary_search(&(v as i64)).is_ok(),
            "{v} is not a source value"
        );
    }
}

/// Aggregating 4 by 4 blocks in the source's own CRS: each target cell is
/// an axis-aligned rectangle of whole source cells, where GDAL's box is
/// exact, so both footprints give the block mean (rounded half up for the
/// integer cells).
#[tokio::test]
async fn exact_and_box_footprints_agree_on_aligned_blocks() {
    let Some(ds) = world().await else { return };
    let src = wxy_source::read_window(
        &ds.levels[0],
        &wxy_source::Window::new(vec![0, 0, 0], vec![1, 360, 720]),
    )
    .await
    .unwrap()
    .values
    .to_f64();
    let dst = GridSpec::from_extent("EPSG:4326", [-180.0, 180.0, -90.0, 90.0], [180, 90]);
    let opts = |footprint| WarpOptions {
        resampling: ResampleAlg::Average,
        footprint,
        wrap_globe: false,
        ..Default::default()
    };
    let boxed = run(&ds, &dst, &opts(Footprint::Box)).await;
    let exact = run(&ds, &dst, &opts(Footprint::Exact { densify: 1 })).await;
    for r in 0..90 {
        for c in 0..180 {
            let mut sum = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    sum += src[(4 * r + i) * 720 + 4 * c + j];
                }
            }
            // a mean at exactly one half may round either way: the
            // weights are fractions summed in floating point
            let mean = sum / 16.0;
            let k = r * 180 + c;
            for (name, got) in [("box", boxed[k]), ("exact", exact[k])] {
                assert!(
                    (got - mean).abs() <= 0.5 + 1e-9,
                    "{name} footprint: cell {k} is {got}, the block mean {mean}"
                );
            }
        }
    }
    let (n, first) = diff(&boxed, &exact);
    assert_eq!(n, 0, "box and exact footprints differ at {first:?}");
}
