# vrt-model

A typed model of virtual dataset documents and a planner over it, with no
I/O and no pixel arithmetic (design m25, wxylib/design). It has no wxy
dependencies and builds for `wasm32-unknown-unknown`, so it can sit behind
a browser reader as well as behind wxy. Working name; it moves to its own
repository when another project uses it.

- `dsn`: source names, the grammar the VRT schemas leave as a plain
  string: paths, URLs, `/vsi` chains with `{...}` nesting, driver prefixes
  with subdatasets, `vrt://` strings (split at the last `?`, so they nest),
  inline VRT. Strings round-trip, including those the R package dsn writes.
- `xml`: an element tree that reads and writes any VRT family.
- `classic`: classic raster VRT as typed values (grid, bands, Simple and
  Complex sources with windows, NODATA, scaling, LUT); every element it does
  not interpret is kept and written back, and `Dataset::unknown()` lists them.
- `vrtcon`: `vrt://` strings lowered to the VRT a translate would write
  (`bands`, `srcwin`, `outsize`, `a_nodata`, `a_srs`, `a_ullr`, `a_gt` so far;
  other keys are an error that names them).
- `plan`: for a read request, which sources, which source windows, where they
  land and which operations apply; through nested VRTs to leaf sources. The
  window arithmetic and the nearest sampling follow the reference library.

## Checked against GDAL

`tests/oracle.rs` plans every case of the wxylib/fixtures `vrt/` corpus,
evaluates the plan with a small reference reader that lives in the test, and
compares with GDAL's read byte for byte: 14 cases (9 VRT documents, 5 `vrt://`
strings), 32 requests (full, windowed, downsampled), all identical to GDAL
3.12.2.

Not yet: multidimensional and vector VRT, averaged and kernel sources,
warped and derived bands (planned, evaluated elsewhere), resampling other
than nearest.
