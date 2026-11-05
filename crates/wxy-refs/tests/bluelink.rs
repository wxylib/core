//! m06 sec 8 point 3: the synthetic Bluelink-shaped table (wxylib/fixtures,
//! mdimrefs/bluelink-synthetic/): two variables, each in two monthly
//! netCDF4 files with many chunks per file, chunks never written, and
//! BRAN's float64 packing. Read straight from the table, every window
//! matches what netCDF-C reads from the files, and the packing comes
//! through unrounded.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use wxy_refs::{from_refs_table, RefsSource, RefsStore};
use wxy_source::{read_window, ChunkKey, ChunkSource, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
        .join("mdimrefs/bluelink-synthetic")
}

fn ints(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn a_many_chunks_per_file_table_matches_netcdf() {
    let root = fixtures();
    if !root.join("manifest.json").exists() {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    }
    let manifest: Value =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json")).unwrap()).unwrap();
    let arrays = from_refs_table(&root, Some("fixture")).unwrap();
    for (name, m) in manifest["arrays"].as_object().unwrap() {
        let a = arrays
            .iter()
            .find(|a| a.path == format!("/{name}"))
            .unwrap();
        assert_eq!(
            a.refs.len() as u64,
            m["present"].as_u64().unwrap(),
            "{name}"
        );
        // many chunks per file
        let per_file = ints(&m["per_file"]);
        assert!(per_file.iter().all(|n| *n > 100), "{name}: {per_file:?}");
        // the packing as the netCDF has it, not rounded
        let attrs = &a.metadata["attributes"];
        assert_eq!(attrs["scale_factor"], m["scale_factor"], "{name}");
        assert_eq!(attrs["add_offset"], m["add_offset"], "{name}");
    }

    let store = Arc::new(RefsStore::from_arrays(arrays, None).unwrap());
    assert_eq!(store.len(), 319 + 304);
    for win in manifest["windows"].as_array().unwrap() {
        let name = win["array"].as_str().unwrap();
        let ds = wxy_zarr::open(store.clone(), name).await.unwrap();
        let src = RefsSource::new(ds.source, store.clone(), name);
        let window = Window::new(ints(&win["start"]), ints(&win["count"]));
        let buf = read_window(&src, &window).await.unwrap();
        let want = std::fs::read(root.join(win["file"].as_str().unwrap())).unwrap();
        assert_eq!(buf.values.to_le_bytes(), want, "window {win}");
        let fill = want
            .chunks_exact(2)
            .filter(|b| i16::from_le_bytes([b[0], b[1]]) == -32768)
            .count() as u64;
        assert_eq!(fill, win["fill_cells"].as_u64().unwrap(), "window {win}");
    }

    // an unwritten chunk is not referenced: it reads nothing
    let ds = wxy_zarr::open(store.clone(), "temp").await.unwrap();
    let src = RefsSource::new(ds.source, store.clone(), "temp");
    assert_eq!(src.stored_bytes(&ChunkKey(vec![2, 3, 1, 0])), Some(0));
    assert!(src.stored_bytes(&ChunkKey(vec![2, 3, 0, 0])).unwrap() > 0);
}
