# oracle-case

Reads oracle-case fixture sets and compares an implementation's values
with their recorded answers. The format (`oracle-case/0`) is defined in
wxylib/fixtures [ORACLE.md](https://github.com/wxylib/fixtures/blob/main/ORACLE.md):
a set's `cases.json` holds the questions (open this dsn, read this window,
this multidimensional slab, these features) and `answers/<oracle-id>/`
holds one reference run's answers, so GDAL 3.8, 3.12 and HEAD answer the
same questions side by side. Answers are recorded with
`scripts/oraclecase.py` in that repository (osgeo.gdal).

The crate has no geo dependencies and does no I/O beyond the set's own
files: a test reads a case, asks its own implementation, and hands the
values to `compare`, which applies the case's tolerance (exact, absolute
or relative, with a count of cells allowed to differ). A refusal by the
oracle is an answer too (`Status::Error`).

```rust
use oracle_case::{compare, Set};

let Some(set) = Set::from_env("WXY_FIXTURES", "cog") else { return };
let answers = set.answers_default()?; // ORACLE_ID, or the newest GDAL
for case in set.cases_with_op("raster.read") {
    let want = answers.values_f64(&case.id)?;
    let got = my_read(&set.resolve(&case.dsn), case.window)?;
    let o = compare(&got, &want, &case.compare);
    assert!(o.passed(), "{}: {o}", case.id);
}
```

Design: wxylib/design `docs/m26-oracle-harness.md`. It lives in the
wxylib/core workspace with no wxy dependencies, to be extracted when
another project uses it.
