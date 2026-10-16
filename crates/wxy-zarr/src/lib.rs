//! # wxy-zarr
//!
//! Zarr for wxylib, through [zarrs](https://zarrs.dev). A thin adapter
//! (design doc 03, decision A4): an array's metadata becomes a
//! [`Collection`] with one domain, and the array becomes a [`ChunkSource`]
//! whose chunks are its Zarr chunks. zarrs does the storage and the codec
//! chain; this crate only maps metadata.
//!
//! Georeferencing is read from what already exists, not from a scheme of
//! our own (gate 1: "lean on existing crate support"):
//!
//! - the GeoZarr conventions `spatial:` (affine transform, registration)
//!   and `proj:` (CRS), v0.1, from the array or inherited from its parent
//!   group, with array attributes taking precedence;
//! - otherwise CF-style 1D coordinate arrays: a sibling array named after a
//!   dim, with that dim's length, becomes a [`Positions::Rectilinear`]
//!   mapping. These are stored positions; [`wxy_domain::normalise`]
//!   recovers a regular grid when there is one.
//!
//! Stage S3 reads regular chunk grids of fixed-size numeric types.
//! [`write`] writes one georeferenced grid the same way (stage S4).

use std::borrow::Cow;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};
use wxy_domain::{
    Affine, CellConvention, Collection, DataType, Dim, Domain, Field, Frame, Mapping, Positions,
};
use wxy_source::{
    read_window, Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Error, Result, Values, Window,
};
use zarrs::array::data_type as zdt;
use zarrs::array::{Array, ArrayBytes, ArrayToBytesCodecTraits, CodecOptions, FromArrayBytes};
use zarrs::group::Group;
use zarrs::storage::AsyncReadableStorageTraits;

pub mod write;
pub use write::{write_path, write_store, GridArray};

type Storage = dyn AsyncReadableStorageTraits;

/// An opened Zarr array: its collection and its chunk source.
pub struct ZarrDataset {
    pub collection: Collection,
    pub source: ZarrSource,
}

/// A Zarr array as a chunk source.
pub struct ZarrSource {
    array: Array<Storage>,
    grid: ChunkGrid,
    dtype: DataType,
}

/// Open array `array` (a path such as `"/data"`) in a local store.
#[cfg(not(target_arch = "wasm32"))]
pub async fn open_path(root: impl AsRef<std::path::Path>, array: &str) -> Result<ZarrDataset> {
    let root = std::fs::canonicalize(root.as_ref()).map_err(|e| Error::Fetch(e.to_string()))?;
    let fs = object_store::local::LocalFileSystem::new_with_prefix(root)
        .map_err(|e| Error::Fetch(e.to_string()))?;
    open_store(Arc::new(fs), array).await
}

/// Open array `array` in any object store, rooted at the Zarr hierarchy.
pub async fn open_store(
    store: Arc<dyn object_store::ObjectStore>,
    array: &str,
) -> Result<ZarrDataset> {
    let storage: Arc<Storage> = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
    open(storage, array).await
}

/// Open array `path` in any zarrs async storage.
pub async fn open(storage: Arc<Storage>, path: &str) -> Result<ZarrDataset> {
    let path = normalise_path(path);
    let array = Array::async_open(storage.clone(), &path)
        .await
        .map_err(|e| Error::Fetch(e.to_string()))?;
    let source = ZarrSource::new(array)?;

    let names = dim_names(&source.array);
    let shape = source.array.shape().to_vec();
    let mut domain = Domain::new(
        names
            .iter()
            .zip(&shape)
            .map(|(n, &s)| Dim::new(n.clone(), s))
            .collect(),
    );

    // Array attributes over parent group attributes (convention inheritance).
    let parent = parent_path(&path);
    let group = match Group::async_open(storage.clone(), &parent).await {
        Ok(g) => g.attributes().clone(),
        Err(_) => Map::new(),
    };
    let attrs = wxy_geozarr::inherit(&group, source.array.attributes());
    // spatial:dimensions is required on arrays; a group's value is a default.
    let frame = proj_frame(&attrs);

    if let Some(m) = spatial_mapping(&attrs, &domain, frame.clone())? {
        let k = domain.add_mapping(m);
        domain.set_primary(k);
    } else {
        for (d, name) in names.iter().enumerate() {
            if let Some(c) = coordinate(&storage, &parent, name, shape[d]).await? {
                let f = frame
                    .clone()
                    .unwrap_or_else(|| Frame::Other("cf:unspecified".into()));
                domain.add_mapping(Mapping::new(vec![d], Positions::Rectilinear(vec![c]), f));
            }
        }
    }

    let mut collection = Collection::new(path.trim_start_matches('/'));
    collection.add_domain(domain);
    let mut field = Field::new(path.rsplit('/').next().unwrap_or("data"), source.dtype, 0);
    field.values.nodata = attrs.get("_FillValue").and_then(Value::as_f64);
    if let Some(s) = attrs.get("scale_factor").and_then(Value::as_f64) {
        field.values.scale = Some(s);
    }
    if let Some(o) = attrs.get("add_offset").and_then(Value::as_f64) {
        field.values.offset = Some(o);
    }
    if let Some(u) = attrs.get("units").and_then(Value::as_str) {
        field.values.units = Some(u.to_string());
    }
    collection.add_field(field);
    collection
        .attrs
        .insert("zarr:fill_value".into(), fill_value_text(&source.array));
    Ok(ZarrDataset { collection, source })
}

impl ZarrSource {
    fn new(array: Array<Storage>) -> Result<Self> {
        let dtype = dtype(&array)?;
        let shape = array.shape().to_vec();
        let n = shape.len();
        let chunk: Vec<u64> = array
            .chunk_shape(&vec![0; n])
            .map_err(|e| Error::Unsupported(e.to_string()))?
            .iter()
            .map(|c| c.get())
            .collect();
        let grid = ChunkGrid::new(shape, chunk);
        if grid.counts() != array.chunk_grid_shape() {
            return Err(Error::Unsupported(
                "chunk grid is not regular (S3 reads regular grids)".into(),
            ));
        }
        Ok(Self { array, grid, dtype })
    }

    /// The underlying zarrs array.
    pub fn array(&self) -> &Array<Storage> {
        &self.array
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl ChunkSource for ZarrSource {
    fn grid(&self) -> &ChunkGrid {
        &self.grid
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let bytes = self
                .array
                .async_retrieve_encoded_chunk(&key.0)
                .await
                .map_err(|e| Error::Fetch(e.to_string()))?;
            // `None` is a chunk that was never written: fill value.
            out.push(Encoded {
                key: key.clone(),
                payload: Box::new(bytes),
            });
        }
        Ok(out)
    }

    fn decode(&self, encoded: Encoded) -> Result<Chunk> {
        let bytes = encoded
            .payload
            .downcast::<Option<zarrs::storage::Bytes>>()
            .map_err(|_| Error::Decode("payload is not a Zarr chunk".into()))?;
        let a = &self.array;
        let chunk_shape = a
            .chunk_shape(&encoded.key.0)
            .map_err(|e| Error::Decode(e.to_string()))?;
        let stored: Vec<u64> = chunk_shape.iter().map(|c| c.get()).collect();
        let n: u64 = stored.iter().product();
        let decoded: ArrayBytes<'static> = match *bytes {
            Some(b) => a
                .codecs()
                .decode(
                    Cow::Owned(b.into()),
                    &chunk_shape,
                    a.data_type(),
                    a.fill_value(),
                    &CodecOptions::default(),
                )
                .map_err(|e| Error::Decode(e.to_string()))?
                .into_owned(),
            None => ArrayBytes::new_fill_value(a.data_type(), n, a.fill_value())
                .map_err(|e| Error::Decode(e.to_string()))?,
        };
        let values = to_values(decoded, &stored, a)?;
        Ok(Chunk {
            window: self.grid.window(&encoded.key),
            key: encoded.key,
            stored,
            values,
        })
    }
}

fn to_values(bytes: ArrayBytes<'static>, shape: &[u64], a: &Array<Storage>) -> Result<Values> {
    fn get<T: FromArrayBytes>(b: ArrayBytes<'static>, s: &[u64], a: &Array<Storage>) -> Result<T> {
        T::from_array_bytes(b, s, a.data_type()).map_err(|e| Error::Decode(e.to_string()))
    }
    let dt = a.data_type();
    Ok(if dt.is::<zdt::UInt8DataType>() {
        Values::U8(get(bytes, shape, a)?)
    } else if dt.is::<zdt::UInt16DataType>() {
        Values::U16(get(bytes, shape, a)?)
    } else if dt.is::<zdt::UInt32DataType>() {
        Values::U32(get(bytes, shape, a)?)
    } else if dt.is::<zdt::UInt64DataType>() {
        Values::U64(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Int8DataType>() {
        Values::I8(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Int16DataType>() {
        Values::I16(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Int32DataType>() {
        Values::I32(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Int64DataType>() {
        Values::I64(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Float32DataType>() {
        Values::F32(get(bytes, shape, a)?)
    } else if dt.is::<zdt::Float64DataType>() {
        Values::F64(get(bytes, shape, a)?)
    } else {
        return Err(Error::Unsupported("Zarr data type".into()));
    })
}

fn dtype(a: &Array<Storage>) -> Result<DataType> {
    let dt = a.data_type();
    Ok(if dt.is::<zdt::BoolDataType>() {
        DataType::Bool
    } else if dt.is::<zdt::UInt8DataType>() {
        DataType::U8
    } else if dt.is::<zdt::UInt16DataType>() {
        DataType::U16
    } else if dt.is::<zdt::UInt32DataType>() {
        DataType::U32
    } else if dt.is::<zdt::UInt64DataType>() {
        DataType::U64
    } else if dt.is::<zdt::Int8DataType>() {
        DataType::I8
    } else if dt.is::<zdt::Int16DataType>() {
        DataType::I16
    } else if dt.is::<zdt::Int32DataType>() {
        DataType::I32
    } else if dt.is::<zdt::Int64DataType>() {
        DataType::I64
    } else if dt.is::<zdt::Float32DataType>() {
        DataType::F32
    } else if dt.is::<zdt::Float64DataType>() {
        DataType::F64
    } else {
        return Err(Error::Unsupported(
            "Zarr data type (S3 reads fixed-size numeric types)".into(),
        ));
    })
}

fn fill_value_text(a: &Array<Storage>) -> String {
    serde_json::to_value(a.metadata())
        .ok()
        .and_then(|m| m.get("fill_value").cloned())
        .map_or_else(String::new, |v| v.to_string())
}

/// Dim names from `dimension_names`, or `dim_<k>` where unnamed.
fn dim_names(a: &Array<Storage>) -> Vec<String> {
    let names = a.dimension_names().clone().unwrap_or_default();
    (0..a.dimensionality())
        .map(|k| {
            names
                .get(k)
                .and_then(|n| n.as_ref().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("dim_{k}"))
        })
        .collect()
}

fn normalise_path(p: &str) -> String {
    let p = p.trim_end_matches('/');
    if p.starts_with('/') {
        p.to_string()
    } else {
        format!("/{p}")
    }
}

fn parent_path(p: &str) -> String {
    match p.rfind('/') {
        Some(0) | None => "/".into(),
        Some(k) => p[..k].to_string(),
    }
}

/// The frame from the `proj:` convention.
fn proj_frame(attrs: &Map<String, Value>) -> Option<Frame> {
    wxy_geozarr::Proj::from_attrs(attrs)
        .and_then(|p| p.definition())
        .map(Frame::crs)
}

/// An affine mapping from the `spatial:` convention, in GDAL order on
/// `[y_dim, x_dim]`, placing cell corners (node registration is shifted by
/// [`wxy_geozarr`]). Unknown transform types are skipped, as the
/// convention asks.
fn spatial_mapping(
    attrs: &Map<String, Value>,
    domain: &Domain,
    frame: Option<Frame>,
) -> Result<Option<Mapping>> {
    let spatial =
        wxy_geozarr::Spatial::from_attrs(attrs).map_err(|e| Error::Decode(e.to_string()))?;
    let Some(spatial) = spatial else {
        return Ok(None);
    };
    let Some(gt) = spatial.geotransform() else {
        return Ok(None);
    };
    let [ydim, xdim] = spatial
        .dimensions
        .as_ref()
        .ok_or_else(|| Error::Decode("spatial:dimensions is required on arrays".into()))?;
    let y = domain.dim_index(ydim).ok_or_else(|| missing(ydim))?;
    let x = domain.dim_index(xdim).ok_or_else(|| missing(xdim))?;
    let frame = frame.unwrap_or(Frame::Index);
    Ok(Some(
        Mapping::new(vec![y, x], Positions::Affine(Affine(gt)), frame)
            .with_cell(CellConvention::Edge),
    ))
}

fn missing(name: &str) -> Error {
    Error::Decode(format!(
        "spatial:dimensions names `{name}`, not a dim of the array"
    ))
}

/// A CF-style coordinate: sibling array `name`, 1D, of length `len`.
async fn coordinate(
    storage: &Arc<Storage>,
    parent: &str,
    name: &str,
    len: u64,
) -> Result<Option<Vec<f64>>> {
    let path = if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    };
    let Ok(array) = Array::async_open(storage.clone(), &path).await else {
        return Ok(None);
    };
    if array.shape() != [len] {
        return Ok(None);
    }
    let Ok(source) = ZarrSource::new(array) else {
        return Ok(None);
    };
    let buf = read_window(&source, &Window::new(vec![0], vec![len])).await?;
    Ok(Some(buf.values.to_f64()))
}
