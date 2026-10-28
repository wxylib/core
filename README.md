# wxylib/core

The wxylib Cargo workspace. Design, decisions and roadmap are in
[wxylib/design](https://github.com/wxylib/design). New here? Start at
**https://wxylib.org/docs/**. API docs for every crate:
[wxylib.org/core](https://wxylib.org/core/).

| Crate | Stage | What it is |
|-------|-------|------------|
| [`wxy-domain`](crates/wxy-domain) | S1 | Domains, mappings and fields: one data model for grids, meshes and features. No I/O. |
| [`wxy-source`](crates/wxy-source) | S2 | Chunk sources: plan, fetch, decode, with chunking in index space. |
| [`wxy-tiff`](crates/wxy-tiff) | S2 | GeoTIFF and COG through async-tiff: a Collection plus one chunk source per level. |
| [`wxy-geozarr`](crates/wxy-geozarr) | S3 | The GeoZarr `spatial:` and `proj:` conventions (v0.1) over JSON attributes; no other wxylib dependency, so any Zarr tool can use it. |
| [`wxy-zarr`](crates/wxy-zarr) | S3 | Zarr v2 and v3 through zarrs, from a directory or by URL: georeferenced by the GeoZarr `spatial:`/`proj:` conventions or CF coordinates; writes GeoZarr grids. |
| [`wxy-virtual`](crates/wxy-virtual) | S3b | Scanners to virtual Zarr: TIFF tile byte ranges as Icechunk virtual references, read back through zarrs. |
| [`wxy-warp`](crates/wxy-warp) | S4 to S5b | Plan, fetch, resample: GDAL's chunked warp (3.14) with [rwarp](https://github.com/hypertidy/rwarp)'s kernels on each chunk, or one rwarp pass; regular, rectilinear and curvilinear sources, extra dimensions streamed under one plan; values at points and cells under polygons, each chunk fetched once. |
| [`wxy-refs`](crates/wxy-refs) | S3c | Byte-reference tables (the mdim-refs Parquet export, or kerchunk Parquet references) read straight as Zarr storage: chunks fetched by range from `file://`, `http(s)://` or `s3://` (credentials and endpoint from `AWS_*`), through `wxy-cache` when given; no Icechunk repository in between. |
| [`wxy-predictor`](crates/wxy-predictor) | S3c | The TIFF horizontal predictor (Predictor=2) as a zarrs codec, so tiles of predicted COGs read as virtual Zarr chunks. |
| [`wxy-cache`](crates/wxy-cache) | m05 C1 | A persistent byte-range cache: wraps a reader, keeps what it fetches (still encoded) as append-only Parquet rows `(source, offset, size, url, data, hash, validator)`, partitioned by source: a reference table with the bytes. |
| [`wxy-cli`](crates/wxy-cli) | S4, S4c | The `wxy` command: `warp` (to a GeoZarr grid), `points` and `zonal` (CSV out), from a GeoTIFF/COG, a Zarr store or a byte-reference table, local or remote. |

```sh
git clone https://github.com/wxylib/fixtures ../fixtures   # GDAL oracle data
cargo test
```

```sh
# warp to Antarctic polar stereographic, written as GeoZarr
cargo run -p wxy-cli -- warp ../fixtures/cog/world-int16-deflate.tif out.zarr \
  --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 145 145 -r bilinear

# values at lon/lat points (x,y lines on stdin), as CSV
printf '147.3,-42.9\n-70.6,-33.4\n' |
  cargo run -p wxy-cli -- points ../fixtures/cog/world-int16-deflate.tif

# statistics over GeoJSON polygons, weighted by exact cell coverage
echo '{"type":"Polygon","coordinates":[[[140,-45],[150,-45],[150,-38],[140,-38],[140,-45]]]}' > box.geojson
cargo run -p wxy-cli -- zonal ../fixtures/cog/world-int16-deflate.tif box.geojson --coverage

# a COG straight from a URL, by range requests: only the header and the
# tiles of the overview the target needs are fetched (here 9 tiles,
# about 4 MB, of a 284 MB Sentinel-2 image)
cargo run -p wxy-cli -- warp \
  https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/G/EN/2024/2/S2B_55GEN_20240203_0_L2A/TCI.tif \
  hobart.zarr --crs EPSG:4326 --extent 147.1 -43.1 147.6 -42.75 --size 1000 700

# any of them with --dry-run: the plan as JSON (every chunk it would
# fetch, the bytes that costs, how the warp is chunked), reading no data
cargo run -p wxy-cli -- zonal ../fixtures/cog/world-int16-deflate.tif box.geojson --dry-run

# keep the bytes a warp reads; the second run reads nothing from the file,
# and --dry-run --cache says how much of a plan the cache already holds
cargo run -p wxy-cli -- warp ../fixtures/cog/world-int16-deflate.tif out.zarr \
  --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 145 145 --cache cache --overwrite

# a byte-reference table as the source, read straight from the table;
# --remap picks the table's remap target (here, the fixture's local copies)
cargo run -p wxy-cli -- warp ../fixtures/mdimrefs sst.zarr --source-array /sst \
  --remap fixture --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 100 100 --cache cache
```

The tests compare every read, warp, point read and polygon with what GDAL
gives for the same input, and require identical numbers; the
[testing explainer](https://wxylib.org/docs/explainers/testing.html) lists the
cases. They look for wxylib/fixtures at `../fixtures`, or at
`WXY_FIXTURES`, and skip when it is missing.

Worked examples on real data: the
[OISST walk-through](https://wxylib.org/docs/oisst-virtual.html)
reads a month of NOAA sea surface temperature as virtual Zarr
(`cargo run -p wxy-virtual --example oisst_refs`), and
[a year of BRAN2023](https://wxylib.org/docs/examples/bran/README.html)
reads an ocean reanalysis on THREDDS through its byte table
(the `wxy-virtual` example `bran_refs`).

License: MIT OR Apache-2.0.
