# wxylib/core

The wxylib Cargo workspace. Design, decisions and roadmap are in
[wxylib/design](https://github.com/wxylib/design).

| Crate | Stage | What it is |
|-------|-------|------------|
| [`wxy-domain`](crates/wxy-domain) | S1 | Domains, mappings and fields: one data model for grids, meshes and features. No I/O. |
| [`wxy-source`](crates/wxy-source) | S2 | Chunk sources: plan, fetch, decode, with chunking in index space. |
| [`wxy-tiff`](crates/wxy-tiff) | S2 | GeoTIFF and COG through async-tiff: a Collection plus one chunk source per level. |

```sh
git clone https://github.com/wxylib/fixtures ../fixtures   # GDAL oracle data
cargo test
```

The `wxy-tiff` tests compare every window byte for byte with what GDAL
reads. They look for wxylib/fixtures at `../fixtures`, or at `WXY_FIXTURES`,
and skip when it is missing.

License: MIT OR Apache-2.0.
