# wxylib/core

The wxylib Cargo workspace. Design, decisions and roadmap are in
[wxylib/design](https://github.com/wxylib/design). New here? Start with the
[intro page](https://github.com/wxylib/docs#readme).

| Crate | Stage | What it is |
|-------|-------|------------|
| [`wxy-domain`](crates/wxy-domain) | S1 | Domains, mappings and fields: one data model for grids, meshes and features. No I/O. |
| [`wxy-source`](crates/wxy-source) | S2 | Chunk sources: plan, fetch, decode, with chunking in index space. |
| [`wxy-tiff`](crates/wxy-tiff) | S2 | GeoTIFF and COG through async-tiff: a Collection plus one chunk source per level. |
| [`wxy-geozarr`](crates/wxy-geozarr) | S3 | The GeoZarr `spatial:` and `proj:` conventions (v0.1) over JSON attributes; no other wxylib dependency, so any Zarr tool can use it. |
| [`wxy-zarr`](crates/wxy-zarr) | S3 | Zarr through zarrs: georeferenced by the GeoZarr `spatial:`/`proj:` conventions or CF coordinates; writes GeoZarr grids. |
| [`wxy-virtual`](crates/wxy-virtual) | S3b | Scanners to virtual Zarr: TIFF tile byte ranges as Icechunk virtual references, read back through zarrs. |
| [`wxy-warp`](crates/wxy-warp) | S4 | Warp a source onto a target grid through [rwarp](https://github.com/hypertidy/rwarp): plan the source window, read it, resample. |
| [`wxy-cli`](crates/wxy-cli) | S4 | The `wxy` command. `wxy warp` warps a GeoTIFF or Zarr array to a GeoZarr grid. |

```sh
git clone https://github.com/wxylib/fixtures ../fixtures   # GDAL oracle data
cargo test
```

```sh
cargo run -p wxy-cli -- warp ../fixtures/cog/world-int16-deflate.tif out.zarr \
  --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 145 145
```

The `wxy-tiff`, `wxy-zarr` and `wxy-virtual` tests compare every window byte for byte with what GDAL
reads. The `wxy-warp` and `wxy-cli` tests compare warps with what GDAL's warper writes: identical
for nearest neighbour, and for every method near 1:1 (interpolated downsampling waits on stage S5). They look for wxylib/fixtures at `../fixtures`, or at `WXY_FIXTURES`,
and skip when it is missing.

License: MIT OR Apache-2.0.
