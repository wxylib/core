//! S3c: the mdim-refs fixture (wxylib/fixtures, mdimrefs/: three
//! OISST-shaped netCDF4 files, mosaicked and scanned) read straight from
//! its table, with no Icechunk repository, gives the windows GDAL reads
//! from the mosaic; through a cache, a second store over the same table
//! reads nothing from the files and gives the same bytes.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use wxy_domain::{Frame, Positions};
use wxy_refs::{from_refs_table, RefsStore};
use wxy_source::{read_window, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("mdimrefs")
}

/// Every manifest window through `store`, as little-endian bytes.
async fn windows(store: Arc<RefsStore>, manifest: &Value) -> Vec<(Value, Vec<u8>)> {
    let mut out = Vec::new();
    for win in manifest["windows"].as_array().unwrap() {
        let v = |k: &str| -> Vec<u64> {
            win[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let ds = wxy_zarr::open(store.clone(), win["array"].as_str().unwrap())
            .await
            .unwrap();
        let buf = read_window(&ds.source, &Window::new(v("start"), v("count")))
            .await
            .unwrap();
        out.push((win.clone(), buf.values.to_le_bytes()));
    }
    out
}

#[tokio::test]
async fn a_table_read_directly_matches_gdal() {
    let root = fixtures();
    if !root.join("manifest.json").exists() {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    }
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json")).unwrap()).unwrap();
    let arrays = from_refs_table(&root, Some("fixture")).unwrap();
    // sources keep the table's own paths; URLs are where the remap sends them
    let r = &arrays.iter().find(|a| a.path == "/sst").unwrap().refs[0];
    assert!(!r.source.starts_with("file://"), "{}", r.source);
    assert!(r.url.starts_with("file://"), "{}", r.url);

    let store = Arc::new(RefsStore::new(&arrays, None).unwrap());
    for name in ["sst", "ice"] {
        let ds = wxy_zarr::open(store.clone(), name).await.unwrap();
        let d = &ds.collection.domains[0];
        assert_eq!(d.shape(), vec![3, 1, 36, 72], "{name}");
        let m = d.primary_for(2).unwrap();
        let Positions::Affine(a) = &m.positions else {
            panic!("{name}: expected an affine")
        };
        assert_eq!(a.0.to_vec(), vec![0.0, 5.0, 0.0, -90.0, 0.0, 5.0], "{name}");
        assert_eq!(m.frame, Frame::crs("EPSG:4326"), "{name}");
        assert_eq!(ds.collection.fields[0].values.nodata, Some(-999.0));
    }
    let got = windows(store.clone(), &manifest).await;
    assert_eq!(got.len(), 5);
    for (win, bytes) in &got {
        let want = std::fs::read(root.join(win["file"].as_str().unwrap())).unwrap();
        assert_eq!(bytes, &want, "window {win}");
    }
    assert!(store.stats().misses > 0);

    // through a cache: the first store fetches and keeps, the second
    // (a new run in effect) reads only the cache
    let dir = std::env::temp_dir().join(format!("wxy-refs-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let first =
        Arc::new(RefsStore::new(&arrays, Some(wxy_cache::Cache::open(&dir).unwrap())).unwrap());
    let a = windows(first.clone(), &manifest).await;
    // windows share chunks, so the first run already answers repeats
    let s = first.stats();
    assert!(s.misses > 0);
    first.cache().unwrap().flush().unwrap();
    let second =
        Arc::new(RefsStore::new(&arrays, Some(wxy_cache::Cache::open(&dir).unwrap())).unwrap());
    let b = windows(second.clone(), &manifest).await;
    let s2 = second.stats();
    assert_eq!(s2.misses, 0, "{s2:?}");
    assert_eq!(s2.hits, s.hits + s.misses);
    assert_eq!(a, b);
    // rows are keyed on the table's source paths, not the remapped URLs
    let rows: Vec<_> = wxy_cache::fragments(&dir)
        .unwrap()
        .iter()
        .flat_map(|f| wxy_cache::read_fragment(f).unwrap())
        .collect();
    assert_eq!(rows.len() as u64, s.misses);
    assert!(rows.iter().all(|r| !r.source.starts_with("file://")));
    assert!(rows.iter().all(|r| r.url.starts_with("file://")));
    let _ = std::fs::remove_dir_all(&dir);
}
