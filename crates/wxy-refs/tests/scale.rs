//! m06 C4: a table the size of the whole GHRSST MUR series (22.8M
//! references, one file per day, about 2,600 tiles a day) loads and serves
//! its chunks without holding a string per reference.
//!
//! Ignored by default: it writes a table of `WXY_SCALE_REFS` rows
//! (default 22,800,000) to a temporary directory, opens it, builds a
//! [`RefsStore`] and prints the peak resident memory. Run with
//!
//! ```text
//! WXY_SCALE_REFS=22800000 cargo test --release -p wxy-refs --test scale -- --ignored --nocapture
//! ```

use std::sync::Arc;

use arrow_array::{ArrayRef, DictionaryArray, Int64Array, Int8Array, RecordBatch};
use parquet::arrow::ArrowWriter;
use serde_json::json;
use wxy_refs::{chunk_key, from_refs_table, RefsStore};

/// Peak resident set size so far, in MiB (Linux only).
fn peak_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

#[test]
#[ignore]
fn a_series_sized_table_fits() {
    let n: u64 = std::env::var("WXY_SCALE_REFS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(22_800_000);
    // GHRSST MUR shaped: time x lat x lon, 1 x 512 x 512 tiles over
    // 17999 x 36000, so 36 x 71 = 2556 tiles a day
    let (ny, nx) = (36u64, 71u64);
    let per_day = ny * nx;
    let days = n.div_ceil(per_day);
    let dir = std::env::temp_dir().join(format!("wxy-refs-scale-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let table = json!({
        "format_version": "mdim-refs/0.2",
        "remap": [],
        "arrays": [{
            "array": "analysed_sst", "kind": "data", "storage": "referenced",
            "dims": ["time", "lat", "lon"], "shape": [days, 17999, 36000],
            "chunks": [1, 512, 512],
            "dtype": "<i2", "fill": "-32768",
            "codec": {"compressor": {"id": "zstd", "level": 9}, "filters": null},
            "attrs": {}, "affine": null, "values": null,
        }],
    });
    std::fs::write(dir.join("arrays.json"), table.to_string()).unwrap();

    let f = std::fs::File::create(dir.join("refs_analysed_sst.parquet")).unwrap();
    let mut w: Option<ArrowWriter<std::fs::File>> = None;
    let batch_days = 200u64;
    let mut written = 0u64;
    let mut d0 = 0u64;
    while written < n {
        let mut t = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut path = Vec::new();
        let mut off = Vec::new();
        let mut size = Vec::new();
        for d in d0..(d0 + batch_days).min(days) {
            let p = format!(
                "https://data.source.coop/ausantarctic/ghrsst-mur-v2/{d:05}/{d:05}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1_analysed_sst.tif"
            );
            for i in 0..per_day {
                if written == n {
                    break;
                }
                t.push(d as i64);
                y.push((i / nx) as i64);
                x.push((i % nx) as i64);
                path.push(p.clone());
                off.push(16384 + 300_000 * i as i64);
                size.push(250_000 + (i as i64 % 1000));
                written += 1;
            }
        }
        d0 += batch_days;
        let rows = t.len();
        let cols: Vec<(&str, ArrayRef)> = vec![
            ("dim_0", Arc::new(Int64Array::from(t))),
            ("dim_1", Arc::new(Int64Array::from(y))),
            ("dim_2", Arc::new(Int64Array::from(x))),
            ("present", Arc::new(Int8Array::from(vec![1i8; rows]))),
            (
                "path",
                Arc::new(
                    path.iter()
                        .map(|s| Some(s.as_str()))
                        .collect::<DictionaryArray<arrow_array::types::Int32Type>>(),
                ),
            ),
            ("offset", Arc::new(Int64Array::from(off))),
            ("size", Arc::new(Int64Array::from(size))),
        ];
        let batch = RecordBatch::try_from_iter(cols).unwrap();
        w.get_or_insert_with(|| {
            ArrowWriter::try_new(f.try_clone().unwrap(), batch.schema(), None).unwrap()
        })
        .write(&batch)
        .unwrap();
    }
    w.unwrap().close().unwrap();
    eprintln!(
        "table: {n} references over {days} days; peak {:?} MiB",
        peak_mib()
    );

    let t0 = std::time::Instant::now();
    let arrays = from_refs_table(&dir, None).unwrap();
    eprintln!(
        "loaded in {:.1} s; peak {:?} MiB",
        t0.elapsed().as_secs_f64(),
        peak_mib()
    );
    let a = arrays.iter().find(|a| a.path == "/analysed_sst").unwrap();
    assert_eq!(a.refs.len() as u64, n);
    let t0 = std::time::Instant::now();
    let store = RefsStore::from_arrays(arrays, None).unwrap();
    eprintln!(
        "store in {:.1} s; peak {:?} MiB",
        t0.elapsed().as_secs_f64(),
        peak_mib()
    );
    // the last reference, and a key past it, are found
    let last = n - 1;
    let (d, i) = (last / per_day, last % per_day);
    let k = chunk_key("analysed_sst", &[d, i / nx, i % nx]);
    let r = store.stored_range(&k).unwrap();
    assert_eq!(r.start, 16384 + 300_000 * i);
    assert!(store
        .stored_range(&chunk_key("analysed_sst", &[days, 0, 0]))
        .is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
