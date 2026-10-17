//! Writing a georeferenced grid as Zarr v3, with the GeoZarr `spatial:` and
//! `proj:` conventions on the array, so that [`open`](crate::open) (and any
//! GeoZarr reader) finds its georeferencing. One array of a fixed-size
//! numeric type, chunked regularly, compressed with zstd.

use std::sync::Arc;

use serde_json::{json, Map, Value};
use wxy_domain::DataType;
use wxy_source::{Error, Result, Values};
use zarrs::array::{Array, ArrayMetadata, ArraySubset};
use zarrs::group::{Group, GroupMetadata};
use zarrs::storage::AsyncReadableWritableStorageTraits;

pub use wxy_geozarr::Proj;

type WritableStorage = dyn AsyncReadableWritableStorageTraits;

/// What to write: dims (the last two are y and x), shape, chunk shape,
/// GDAL geotransform of the last two dims, CRS and nodata.
#[derive(Debug, Clone, PartialEq)]
pub struct GridArray {
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
    pub chunk: Vec<u64>,
    pub geotransform: [f64; 6],
    pub crs: Option<String>,
    pub nodata: Option<f64>,
}

impl GridArray {
    /// A grid of `shape` with dims `y, x`, `band, y, x`, or `dim_0, ...,
    /// y, x` for more, chunked one slice of each leading dim and 512 by
    /// 512 cells.
    pub fn new(shape: Vec<u64>, geotransform: [f64; 6]) -> Self {
        let lead = shape.len().saturating_sub(2);
        let dims: Vec<String> = match lead {
            0 => vec![],
            1 => vec!["band".to_string()],
            _ => (0..lead).map(|i| format!("dim_{i}")).collect(),
        }
        .into_iter()
        .chain(["y".into(), "x".into()])
        .collect();
        let n = shape.len();
        let chunk = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i + 2 >= n { s.min(512) } else { 1 })
            .collect();
        Self {
            dims,
            shape,
            chunk,
            geotransform,
            crs: None,
            nodata: None,
        }
    }

    /// The array's attributes: `spatial:`, `proj:` (when the CRS has a
    /// `proj:` form) and `_FillValue`.
    pub fn attributes(&self) -> Result<Map<String, Value>> {
        let n = self.dims.len();
        if n < 2 || self.shape.len() != n || self.chunk.len() != n {
            return Err(Error::Unsupported(
                "dims, shape and chunk must agree, with at least y and x".into(),
            ));
        }
        let mut attrs = Map::new();
        wxy_geozarr::Spatial::from_geotransform(
            self.geotransform,
            [&self.dims[n - 2], &self.dims[n - 1]],
            [self.shape[n - 2], self.shape[n - 1]],
        )
        .to_attrs(&mut attrs);
        if let Some(c) = &self.crs {
            let p = wxy_geozarr::Proj::from_definition(c).ok_or_else(|| {
                Error::Unsupported(format!(
                    "CRS `{c}` has no proj: form (use AUTH:CODE, WKT2 or PROJJSON)"
                ))
            })?;
            p.to_attrs(&mut attrs);
        }
        if let Some(v) = self.nodata {
            attrs.insert("_FillValue".into(), float_json(v));
        }
        Ok(attrs)
    }

    /// Zarr v3 array metadata for values of `dtype`.
    pub fn metadata(&self, dtype: DataType) -> Result<Value> {
        Ok(json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": self.shape,
            "data_type": data_type_name(dtype)?,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": self.chunk}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": fill_value_json(dtype, self.nodata.unwrap_or(0.0)),
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 3, "checksum": false}}
            ],
            "dimension_names": self.dims,
            "attributes": self.attributes()?,
        }))
    }
}

/// Write `values` (C order over `grid.shape`) as array `array` (such as
/// `"/data"`) in a local directory, creating it.
#[cfg(not(target_arch = "wasm32"))]
pub async fn write_path(
    root: impl AsRef<std::path::Path>,
    array: &str,
    values: &Values,
    grid: &GridArray,
) -> Result<()> {
    let w = create_path(root, array, values.dtype(), grid).await?;
    w.write_all(values).await
}

/// Write to any object store, rooted at the Zarr hierarchy.
pub async fn write_store(
    store: Arc<dyn object_store::ObjectStore>,
    array: &str,
    values: &Values,
    grid: &GridArray,
) -> Result<()> {
    let w = create_store(store, array, values.dtype(), grid).await?;
    w.write_all(values).await
}

/// Write to any zarrs async storage. Every group above the array is
/// written too, without attributes.
pub async fn write(
    storage: Arc<WritableStorage>,
    array: &str,
    values: &Values,
    grid: &GridArray,
) -> Result<()> {
    let w = create(storage, array, values.dtype(), grid).await?;
    w.write_all(values).await
}

/// An array whose metadata is written and whose values are written
/// window by window ([`GridWriter::write_window`]), so that a result
/// larger than memory can be streamed into it.
pub struct GridWriter {
    array: Array<WritableStorage>,
    shape: Vec<u64>,
}

/// Create array `array` of `dtype` in a local directory (creating the
/// directory), with no values written yet.
#[cfg(not(target_arch = "wasm32"))]
pub async fn create_path(
    root: impl AsRef<std::path::Path>,
    array: &str,
    dtype: DataType,
    grid: &GridArray,
) -> Result<GridWriter> {
    let root = root.as_ref();
    std::fs::create_dir_all(root).map_err(|e| Error::Fetch(e.to_string()))?;
    let fs = object_store::local::LocalFileSystem::new_with_prefix(root)
        .map_err(|e| Error::Fetch(e.to_string()))?;
    create_store(Arc::new(fs), array, dtype, grid).await
}

/// Create an array in any object store, rooted at the Zarr hierarchy.
pub async fn create_store(
    store: Arc<dyn object_store::ObjectStore>,
    array: &str,
    dtype: DataType,
    grid: &GridArray,
) -> Result<GridWriter> {
    let storage: Arc<WritableStorage> = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
    create(storage, array, dtype, grid).await
}

/// Create an array in any zarrs async storage, with every group above it.
pub async fn create(
    storage: Arc<WritableStorage>,
    array: &str,
    dtype: DataType,
    grid: &GridArray,
) -> Result<GridWriter> {
    let path = crate::normalise_path(array);
    let fail = |e: &dyn std::fmt::Display| Error::Fetch(e.to_string());

    let mut parent = String::new();
    let mut groups = vec!["/".to_string()];
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    for s in &segments[..segments.len() - 1] {
        parent.push('/');
        parent.push_str(s);
        groups.push(parent.clone());
    }
    for g in groups {
        let meta: GroupMetadata =
            serde_json::from_value(json!({"zarr_format": 3, "node_type": "group"}))
                .map_err(|e| fail(&e))?;
        Group::new_with_metadata(storage.clone(), &g, meta)
            .map_err(|e| fail(&e))?
            .async_store_metadata()
            .await
            .map_err(|e| fail(&e))?;
    }

    let meta: ArrayMetadata =
        serde_json::from_value(grid.metadata(dtype)?).map_err(|e| fail(&e))?;
    let out = Array::new_with_metadata(storage.clone(), &path, meta).map_err(|e| fail(&e))?;
    out.async_store_metadata().await.map_err(|e| fail(&e))?;
    xy_coordinates(&storage, &parent, &path, grid).await?;
    Ok(GridWriter {
        array: out,
        shape: grid.shape.clone(),
    })
}

/// CF 1D coordinates for the y and x dims (cell centres), beside the
/// array, so that CF readers such as xarray see the grid as well as
/// GeoZarr readers do. Skipped for a rotated grid, and where the name is
/// the array's own.
async fn xy_coordinates(
    storage: &Arc<WritableStorage>,
    parent: &str,
    path: &str,
    grid: &GridArray,
) -> Result<()> {
    let fail = |e: &dyn std::fmt::Display| Error::Fetch(e.to_string());
    let gt = grid.geotransform;
    let n = grid.shape.len();
    if n < 2 || gt[2] != 0.0 || gt[4] != 0.0 {
        return Ok(());
    }
    let geographic = grid.crs.as_deref().is_some_and(|c| {
        [
            "EPSG:4326",
            "EPSG:4258",
            "EPSG:4269",
            "EPSG:4283",
            "EPSG:7844",
        ]
        .iter()
        .any(|g| c.eq_ignore_ascii_case(g))
    });
    for (d, origin, step, axis) in [(n - 2, gt[3], gt[5], "Y"), (n - 1, gt[0], gt[1], "X")] {
        let name = &grid.dims[d];
        let at = format!("{}/{name}", parent.trim_end_matches('/'));
        if at == path {
            continue;
        }
        let len = grid.shape[d];
        let values: Vec<f64> = (0..len).map(|i| origin + (i as f64 + 0.5) * step).collect();
        let mut attrs = json!({"axis": axis});
        match (geographic, axis) {
            (true, "Y") => {
                attrs["standard_name"] = json!("latitude");
                attrs["units"] = json!("degrees_north");
            }
            (true, _) => {
                attrs["standard_name"] = json!("longitude");
                attrs["units"] = json!("degrees_east");
            }
            (false, "Y") => attrs["standard_name"] = json!("projection_y_coordinate"),
            (false, _) => attrs["standard_name"] = json!("projection_x_coordinate"),
        }
        let meta: ArrayMetadata = serde_json::from_value(json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [len],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [len.max(1)]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "dimension_names": [name],
            "attributes": attrs,
        }))
        .map_err(|e| fail(&e))?;
        let a = Array::new_with_metadata(storage.clone(), &at, meta).map_err(|e| fail(&e))?;
        a.async_store_metadata().await.map_err(|e| fail(&e))?;
        let all = ArraySubset::new_with_start_shape(vec![0], vec![len]).map_err(|e| fail(&e))?;
        a.async_store_array_subset(&all, &values)
            .await
            .map_err(|e| fail(&e))?;
    }
    Ok(())
}

impl GridWriter {
    /// Write `values` (C order over `window`). Windows aligned to the
    /// array's chunks write whole chunks; others read and rewrite the
    /// chunks they share.
    pub async fn write_window(&self, window: &wxy_source::Window, values: &Values) -> Result<()> {
        let fail = |e: &dyn std::fmt::Display| Error::Fetch(e.to_string());
        let n = self.shape.len();
        let fits = window.offset.len() == n
            && window.size.len() == n
            && (0..n).all(|d| window.offset[d] + window.size[d] <= self.shape[d]);
        if !fits || values.len() as u64 != window.len() {
            return Err(Error::Unsupported(format!(
                "{} values for window {:?} + {:?} of shape {:?}",
                values.len(),
                window.offset,
                window.size,
                self.shape
            )));
        }
        let subset = ArraySubset::new_with_start_shape(window.offset.clone(), window.size.clone())
            .map_err(|e| fail(&e))?;
        macro_rules! store {
            ($v:expr) => {
                self.array
                    .async_store_array_subset(&subset, $v)
                    .await
                    .map_err(|e| fail(&e))?
            };
        }
        match values {
            Values::U8(v) => store!(v),
            Values::U16(v) => store!(v),
            Values::U32(v) => store!(v),
            Values::U64(v) => store!(v),
            Values::I8(v) => store!(v),
            Values::I16(v) => store!(v),
            Values::I32(v) => store!(v),
            Values::I64(v) => store!(v),
            Values::F32(v) => store!(v),
            Values::F64(v) => store!(v),
        }
        Ok(())
    }

    /// Write every value of the array at once.
    pub async fn write_all(&self, values: &Values) -> Result<()> {
        let all = wxy_source::Window::new(vec![0; self.shape.len()], self.shape.clone());
        self.write_window(&all, values).await
    }
}

/// The Zarr v3 `data_type` name of a wxylib data type.
pub fn data_type_name(d: DataType) -> Result<&'static str> {
    use DataType as D;
    Ok(match d {
        D::Bool => "bool",
        D::U8 => "uint8",
        D::U16 => "uint16",
        D::U32 => "uint32",
        D::U64 => "uint64",
        D::I8 => "int8",
        D::I16 => "int16",
        D::I32 => "int32",
        D::I64 => "int64",
        D::F32 => "float32",
        D::F64 => "float64",
        other => return Err(Error::Unsupported(format!("data type {other:?}"))),
    })
}

/// A Zarr v3 `fill_value` of `dtype` for `v`.
pub fn fill_value_json(d: DataType, v: f64) -> Value {
    use DataType as D;
    match d {
        D::F32 | D::F64 => float_json(v),
        D::Bool => json!(v != 0.0),
        _ => json!(v as i64),
    }
}

/// A float as JSON, with Zarr v3's strings for the values JSON has no
/// number for (`json!(f64::NAN)` would be `null`).
pub fn float_json(v: f64) -> Value {
    if v.is_nan() {
        json!("NaN")
    } else if v == f64::INFINITY {
        json!("Infinity")
    } else if v == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(v)
    }
}

/// A float from JSON written by [`float_json`] (or any number).
pub fn json_float(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => v.as_f64(),
    }
}
