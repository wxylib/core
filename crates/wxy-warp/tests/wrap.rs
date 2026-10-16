//! Longitude wrapping for a 0..360, south-up source whose step is not
//! exactly 360 / ncol (as when it is read from float32 coordinates, such as
//! OISST's): the target's western half (longitudes below 0) must read the
//! source's eastern columns, not come back empty.

use std::path::PathBuf;

use wxy_domain::{Affine, Frame, Mapping, Positions};
use wxy_warp::{warp, GridSpec, WarpOptions};

#[tokio::test]
async fn zero_to_360_source_wraps() {
    let root = std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"));
    let Ok(ds) = wxy_tiff::open_path(root.join("cog/world-int16-deflate.tif")).await else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    // The fixture's values on a 0..360, south-up georeferencing.
    let mut domain = ds.collection.domains[0].clone();
    let n = domain.shape().len();
    let gt = [0.0, 0.5 + 1e-9, 0.0, -90.0, 0.0, 0.5];
    let k = domain.add_mapping(Mapping::new(
        vec![n - 2, n - 1],
        Positions::Affine(Affine(gt)),
        Frame::crs("EPSG:4326"),
    ));
    domain.set_primary(k);
    let dst = GridSpec::from_extent("EPSG:3031", [-4.0e6, 4.0e6, -4.0e6, 4.0e6], [100, 100]);
    let opts = WarpOptions {
        nodata: Some(-32768.0),
        ..Default::default()
    };
    let v = warp(&ds.levels[0], &domain, &dst, &opts, None)
        .await
        .unwrap()
        .values
        .to_f64();
    let empty = |cols: std::ops::Range<usize>| {
        (0..100)
            .flat_map(|r| cols.clone().map(move |c| r * 100 + c))
            .filter(|&i| v[i] == -32768.0)
            .count()
    };
    assert_eq!((empty(0..50), empty(50..100)), (0, 0));
}
