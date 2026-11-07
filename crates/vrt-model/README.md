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
- `footprint`: where each source is placed (its destination rectangle after
  the `SrcRect` and VRT-edge clips) in pixels and in the VRT's coordinates,
  and which sources meet a box. Any source kind and any resampling: this is
  the catalogue view a map client needs ("which files cover this viewport"),
  not a read plan. Boxes meet by interval overlap, so a tall narrow query
  through a wide tile is found.
- `url`: source names as plain URLs a client fetches (`/vsicurl/`, `/vsis3/`,
  `/vsigs/`, `/vsiaz/`, `s3://`, `gs://`), with object store hosts set from
  the reference library's option names; archives, subdatasets, nested VRTs
  and local paths are an error that says which.

`examples/footprints.rs` lists the sources of a VRT that meet a box, with
their URLs. On the USGS 1/3 arc-second VRT (1,449 COGs), a 0.1 by 10 degree
strip at -120.6 meets 11 tiles; a corner-in-box test finds 3 of them (design
`docs/watch/vrtmate.md`).

## Checked against GDAL

`tests/oracle.rs` plans every case of the wxylib/fixtures `vrt/` corpus,
evaluates the plan with a small reference reader that lives in the test, and
compares with GDAL's read byte for byte: 14 cases (9 VRT documents, 5 `vrt://`
strings), 32 requests (full, windowed, downsampled), all identical to GDAL
3.12.2. `tests/footprint.rs` checks over the same corpus that the sources
whose footprint meets a window's box are the sources the planner reads for
that window (54 windows).

Not yet: multidimensional and vector VRT, averaged and kernel sources,
warped and derived bands (planned, evaluated elsewhere), resampling other
than nearest.
