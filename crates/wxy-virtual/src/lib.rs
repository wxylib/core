//! # wxy-virtual
//!
//! Virtual Zarr for wxylib (design doc 03, A6 and S3b): a *scanner* turns a
//! file into chunk references (url, byte offset, length) plus Zarr array
//! metadata, Icechunk stores them, and every read goes through zarrs and
//! [`wxy_zarr`]. No reference format of our own: Icechunk is the store
//! (landscape survey, LS3), and the metadata carries the GeoZarr
//! `spatial:` and `proj:` conventions.
//!
//! Stage S3b scans tiled TIFFs whose tiles are plain Zarr chunks: no
//! compression, Deflate (as `numcodecs.zlib`) or ZSTD, no predictor, chunky
//! or planar. Chunky multi-band tiles are pixel-interleaved, which is a
//! Zarr `transpose` codec over `(band, y, x)`. Other TIFFs are reported as
//! unsupported rather than referenced wrongly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_tiff::reader::Endianness;
use async_tiff::tags::{Compression, PlanarConfiguration, Predictor};
use icechunk::format::manifest::{VirtualChunkLocation, VirtualChunkRef};
use icechunk::format::ChunkIndices;
use icechunk::repository::VersionInfo;
use icechunk::virtual_chunks::VirtualChunkContainer;
use icechunk::{ObjectStoreConfig, Repository, RepositoryConfig, Store};
use serde_json::{json, Value};
use wxy_domain::{Frame, Positions};
use wxy_source::{ChunkSource, Error, Result};
use zarrs::storage::AsyncReadableStorageTraits;

/// A reference to one stored chunk: its key in the Zarr chunk grid and the
/// bytes that hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub key: Vec<u64>,
    pub url: String,
    pub offset: u64,
    pub length: u64,
}

/// One virtual array: where it goes in the hierarchy, its Zarr v3
/// metadata (`zarr.json`), and its chunk references. Chunks with no
/// reference read as the fill value.
#[derive(Debug, Clone)]
pub struct VirtualArray {
    pub path: String,
    pub metadata: Value,
    pub refs: Vec<ChunkRef>,
}

/// Scan level 0 of a local TIFF into a virtual array at `/data`.
pub async fn scan_tiff(path: impl AsRef<Path>) -> Result<VirtualArray> {
    let path = std::fs::canonicalize(path.as_ref()).map_err(|e| Error::Fetch(e.to_string()))?;
    let url = format!("file://{}", path.display());
    let ds = wxy_tiff::open_path(&path).await?;
    let level = &ds.levels[0];
    let ifd = level.ifd();

    match ifd.predictor() {
        None | Some(Predictor::None) => {}
        Some(_) => return Err(Error::Unsupported("TIFF predictor".into())),
    }
    let compressor = match ifd.compression() {
        Compression::None => None,
        Compression::Deflate | Compression::OldDeflate => {
            Some(json!({"name": "numcodecs.zlib", "configuration": {"level": 6}}))
        }
        Compression::ZSTD => {
            Some(json!({"name": "zstd", "configuration": {"level": 0, "checksum": false}}))
        }
        c => return Err(Error::Unsupported(format!("TIFF compression {c:?}"))),
    };
    let (offsets, counts) = match (ifd.tile_offsets(), ifd.tile_byte_counts()) {
        (Some(o), Some(c)) => (o, c),
        _ => return Err(Error::Unsupported("stripped TIFF".into())),
    };

    let grid = level.grid();
    let shape = grid.shape.clone();
    let chunk = grid.chunk.clone();
    let (bands, th, tw) = (shape[0], chunk[1], chunk[2]);
    let counts_yx = grid.counts();
    let (ny, nx) = (counts_yx[1], counts_yx[2]);
    let planar = ifd.planar_configuration() == PlanarConfiguration::Planar;

    // Chunk shape and codecs. Chunky tiles store (y, x, band); planar
    // tiles store one band's (y, x).
    let endian = match level.endianness() {
        Endianness::LittleEndian => "little",
        Endianness::BigEndian => "big",
    };
    let mut codecs = Vec::new();
    let chunk_shape = if planar || bands == 1 {
        vec![1, th, tw]
    } else {
        codecs.push(json!({"name": "transpose", "configuration": {"order": [1, 2, 0]}}));
        vec![bands, th, tw]
    };
    codecs.push(json!({"name": "bytes", "configuration": {"endian": endian}}));
    codecs.extend(compressor);

    let mut refs = Vec::new();
    let planes = if planar { bands } else { 1 };
    for p in 0..planes {
        for ty in 0..ny {
            for tx in 0..nx {
                let i = (p * ny * nx + ty * nx + tx) as usize;
                let (offset, length) = (offsets[i], counts[i]);
                if length == 0 {
                    continue; // sparse tile: fill value
                }
                refs.push(ChunkRef {
                    key: vec![p, ty, tx],
                    url: url.clone(),
                    offset,
                    length,
                });
            }
        }
    }

    let field = &ds.collection.fields[0];
    let fill = field.values.nodata.unwrap_or(0.0);
    let mut attrs = serde_json::Map::new();
    if let Some(m) = ds.collection.domains[0].primary_for(2) {
        if let Positions::Affine(a) = &m.positions {
            wxy_geozarr::Spatial::from_geotransform(a.0, ["y", "x"], [shape[1], shape[2]])
                .to_attrs(&mut attrs);
            if let Frame::Crs(c) = &m.frame {
                wxy_geozarr::Proj {
                    code: Some(c.clone()),
                    ..Default::default()
                }
                .to_attrs(&mut attrs);
            }
        }
    }
    if let Some(n) = field.values.nodata {
        attrs.insert("_FillValue".into(), json!(n));
    }

    let metadata = json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": wxy_zarr::write::data_type_name(level.dtype())?,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": wxy_zarr::write::fill_value_json(level.dtype(), fill),
        "codecs": codecs,
        "dimension_names": ["band", "y", "x"],
        "attributes": attrs,
    });
    Ok(VirtualArray {
        path: "/data".into(),
        metadata,
        refs,
    })
}

/// The URL prefix a set of arrays reads from: the directory of each file.
fn container_prefixes(arrays: &[VirtualArray]) -> Vec<String> {
    let mut out: Vec<String> = arrays
        .iter()
        .flat_map(|a| a.refs.iter())
        .filter_map(|r| r.url.rfind('/').map(|k| r.url[..=k].to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

fn config(
    prefixes: &[String],
) -> Result<(
    RepositoryConfig,
    HashMap<String, Option<icechunk::config::Credentials>>,
)> {
    let mut config = RepositoryConfig::default();
    let mut creds = HashMap::new();
    for p in prefixes {
        let store = if p.starts_with("file://") {
            ObjectStoreConfig::LocalFileSystem(Default::default())
        } else {
            return Err(Error::Unsupported(format!("virtual chunk location {p}")));
        };
        let c = VirtualChunkContainer::new(p.clone(), store).map_err(Error::Fetch)?;
        config
            .set_virtual_chunk_container(c)
            .map_err(|e| Error::Fetch(e.to_string()))?;
        creds.insert(p.clone(), None);
    }
    Ok((config, creds))
}

fn ice(e: impl std::fmt::Display) -> Error {
    Error::Fetch(e.to_string())
}

/// Write arrays into a new Icechunk repository at `repo` (a local
/// directory), with a root group, and commit them to `main`.
pub async fn write_icechunk(repo: impl AsRef<Path>, arrays: &[VirtualArray]) -> Result<()> {
    let storage = icechunk::new_local_filesystem_storage(repo.as_ref())
        .await
        .map_err(ice)?;
    let (config, creds) = config(&container_prefixes(arrays))?;
    let repo = Repository::create(storage)
        .config(config)
        .authorize_virtual_chunk_access(creds)
        .execute()
        .await
        .map_err(ice)?;
    let session = Arc::new(tokio::sync::RwLock::new(
        repo.writable_session("main").await.map_err(ice)?,
    ));
    let store = Store::from_session(session.clone()).await;
    store
        .set(
            "zarr.json",
            r#"{"zarr_format":3,"node_type":"group"}"#.into(),
        )
        .await
        .map_err(ice)?;
    for a in arrays {
        let key = format!("{}/zarr.json", a.path.trim_start_matches('/'));
        store
            .set(&key, serde_json::to_vec(&a.metadata).map_err(ice)?.into())
            .await
            .map_err(ice)?;
        let path: icechunk::format::Path = a.path.as_str().try_into().map_err(ice)?;
        let mut refs = Vec::with_capacity(a.refs.len());
        for r in &a.refs {
            let location = VirtualChunkLocation::from_url(&r.url).map_err(ice)?;
            let key = ChunkIndices(r.key.iter().map(|&k| k as u32).collect());
            refs.push((
                key,
                VirtualChunkRef {
                    location,
                    offset: r.offset,
                    length: r.length,
                    checksum: None,
                },
            ));
        }
        let result = store
            .set_virtual_refs(&path, true, refs)
            .await
            .map_err(ice)?;
        if let icechunk::store::SetVirtualRefsResult::FailedRefs(f) = result {
            return Err(Error::Fetch(format!("{} references failed", f.len())));
        }
    }
    session
        .write()
        .await
        .commit("wxy-virtual scan")
        .execute()
        .await
        .map_err(ice)?;
    Ok(())
}

/// Open the tip of `main` in a local Icechunk repository as zarrs storage,
/// authorising reads under `prefixes` (such as `file:///data/cogs/`).
pub async fn open_icechunk(
    repo: impl AsRef<Path>,
    prefixes: &[String],
) -> Result<Arc<dyn AsyncReadableStorageTraits>> {
    let storage = icechunk::new_local_filesystem_storage(repo.as_ref())
        .await
        .map_err(ice)?;
    let (config, creds) = config(prefixes)?;
    let repo = Repository::open(storage)
        .config(config)
        .authorize_virtual_chunk_access(creds)
        .execute()
        .await
        .map_err(ice)?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".into()))
        .await
        .map_err(ice)?;
    Ok(Arc::new(zarrs_icechunk::AsyncIcechunkStore::new(session)))
}

/// The container prefixes an array set needs, for [`open_icechunk`].
pub fn prefixes(arrays: &[VirtualArray]) -> Vec<String> {
    container_prefixes(arrays)
}
