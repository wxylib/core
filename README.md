# wxylib/core

The wxylib Cargo workspace. Design, decisions and roadmap are in
[wxylib/design](https://github.com/wxylib/design). New here? Start with the
[intro page](https://github.com/wxylib/docs#readme). API docs for every
crate: [wxylib.github.io/core](https://wxylib.github.io/core/).

| Crate | Stage | What it is |
|-------|-------|------------|
| [`wxy-domain`](crates/wxy-domain) | S1 | Domains, mappings and fields: one data model for grids, meshes and features. No I/O. |
| [`wxy-source`](crates/wxy-source) | S2 | Chunk sources: plan, fetch, decode, with chunking in index space. |
| [`wxy-tiff`](crates/wxy-tiff) | S2 | GeoTIFF and COG through async-tiff: a Collection plus one chunk source per level. |
| [`wxy-geozarr`](crates/wxy-geozarr) | S3 | The GeoZarr `spatial:` and `proj:` conventions (v0.1) over JSON attributes; no other wxylib dependency, so any Zarr tool can use it. |
| [`wxy-zarr`](crates/wxy-zarr) | S3 | Zarr through zarrs: georeferenced by the GeoZarr `spatial:`/`proj:` conventions or CF coordinates; writes GeoZarr grids. |
| [`wxy-virtual`](crates/wxy-virtual) | S3b | Scanners to virtual Zarr: TIFF tile byte ranges as Icechunk virtual references, read back through zarrs. |
| [`wxy-resample`](crates/wxy-resample) | S5 | GDAL's warp kernels in its arithmetic order, scaled when the target is coarser than the source. No transforms, no I/O. |
| [`wxy-warp`](crates/wxy-warp) | S4 to S5b | Plan, fetch, resample: GDAL's chunked warp with `wxy-resample` kernels, or one pass through [rwarp](https://github.com/hypertidy/rwarp); regular, rectilinear and curvilinear sources, extra dimensions streamed under one plan; values at points and cells under polygons, each chunk fetched once. |
| [`wxy-cache`](crates/wxy-cache) | m05 C1 | A persistent byte-range cache: wraps a reader, keeps what it fetches (still encoded) as append-only Parquet rows `(source, offset, size, url, data, hash, validator)`, partitioned by source: a reference table with the bytes. |
| [`wxy-cli`](crates/wxy-cli) | S4, S4c | The `wxy` command: `warp` (to a GeoZarr grid), `points` and `zonal` (CSV out). |

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

# any of them with --dry-run: the plan as JSON (every chunk it would
# fetch, the bytes that costs, how the warp is chunked), reading no data
cargo run -p wxy-cli -- zonal ../fixtures/cog/world-int16-deflate.tif box.geojson --dry-run

# keep the bytes a warp reads; the second run reads nothing from the file,
# and --dry-run --cache says how much of a plan the cache already holds
cargo run -p wxy-cli -- warp ../fixtures/cog/world-int16-deflate.tif out.zarr \
  --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 145 145 --cache cache --overwrite
```

The tests compare every read, warp, point read and polygon with what GDAL
gives for the same input, and require identical numbers; the
[intro page](https://github.com/wxylib/docs#what-the-tests-check) lists the
cases. They look for wxylib/fixtures at `../fixtures`, or at
`WXY_FIXTURES`, and skip when it is missing.

A worked example on real data: the
[OISST walk-through](https://github.com/wxylib/docs/blob/main/oisst-virtual.md)
reads a month of NOAA sea surface temperature as virtual Zarr
(`cargo run -p wxy-virtual --example oisst_refs`).

License: MIT OR Apache-2.0.
