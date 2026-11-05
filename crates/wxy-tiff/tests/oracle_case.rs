//! The cog/ windows through the oracle-case reader: every raster.read case
//! of wxylib/fixtures `cog/cases.json`, against every GDAL answer set
//! present (3.12.2 as made, 3.8.4 for the browser builds, and HEAD once
//! the answers workflow has run). Skips when `WXY_FIXTURES` is unset or
//! the checkout predates cases.json.

use oracle_case::{compare, Set};
use wxy_source::{read_window, Window};

#[tokio::test]
async fn cog_windows_match_every_answer_set() {
    let Some(set) = Set::from_env("WXY_FIXTURES", "cog") else {
        return;
    };
    let ids = set.oracle_ids();
    assert!(!ids.is_empty(), "no answer sets");
    for id in &ids {
        let answers = set.answers(id).unwrap();
        let mut n = 0;
        for case in set.cases_with_op("raster.read") {
            let ds = wxy_tiff::open_path(set.resolve(&case.dsn)).await.unwrap();
            let level0 = &ds.levels[0];
            let shape = ds.collection.domains[0].shape();
            let [col, row, w, h] = case.window.expect("cog cases name a window");
            let bands = shape[0];
            let win = Window::new(vec![0, row, col], vec![bands, h, w]);
            let got = read_window(level0, &win).await.unwrap();
            let (v, want_bytes) = answers.values_bytes(&case.id).unwrap();
            // Bytes first (the strict check), then values for the report.
            if got.values.to_le_bytes() != want_bytes {
                let want = oracle_case::decode_f64(&want_bytes, &v.dtype).unwrap();
                let got = oracle_case::decode_f64(&got.values.to_le_bytes(), &v.dtype).unwrap();
                let o = compare(&got, &want, &case.compare);
                panic!("{id} {}: {o}", case.id);
            }
            n += 1;
        }
        eprintln!("{id}: {n} windows identical");
        assert!(n >= 15);
    }
}
