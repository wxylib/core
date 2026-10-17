//! Byte-reference tables as a source of virtual arrays.
//!
//! An mdim-refs/0.2 table (hypertidy/zaro, `mdimrefs://`) holds one row
//! per chunk of an array that a mosaic has already placed in its global
//! chunk grid: `dim_0 .. dim_{n-1}, present, path, offset, size, info`.
//! Array metadata (dims, shape, chunks, numpy dtype, fill, Zarr v2 codec
//! ids, CF attributes) and coordinates (affine `{start, step}` or inline
//! values) sit beside it. This module reads the table's exported form, one
//! `refs_<array>.parquet` per referenced array plus `arrays.json` (the
//! `arrays`, `remap` and `meta` tables), and turns it into
//! [`VirtualArray`]s for [`crate::write_icechunk`]. The table is not
//! copied into a format of our own: Icechunk is where it is written.
//!
//! - v2 codecs become Zarr v3: `bytes` with the dtype's byte order, then
//!   `shuffle` as `numcodecs.shuffle`, then `zlib` as `numcodecs.zlib`
//!   (`gzip` and `zstd` map too); anything else is unsupported.
//! - Affine coordinates on an array's last two dims become GeoZarr
//!   `spatial:` attributes (pixel registration, so the start moves back by
//!   half a step), with `proj:code` EPSG:4326 when those dims are latitude
//!   and longitude. Every coordinate is also written as a small CF
//!   coordinate array with its values inline.
//! - Rows with `present = 1` become [`ChunkRef`]s, or inline chunks when
//!   the table holds their bytes (`offset` null, `data` set); absent rows
//!   are left out and read as the fill value.
//!
//! Spec: wxylib/design `specs/mdim-refs-0.2.md`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow_array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::DataType as ArrowType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Map, Value};
use wxy_domain::DataType;
use wxy_source::{Error, Result};

use crate::{ChunkRef, VirtualArray};

/// The table format versions this reader takes.
pub const FORMAT_PREFIX: &str = "mdim-refs/";

/// Read the exported table in `dir` (`arrays.json` and
/// `refs_<array>.parquet`) as virtual arrays, one per array at `/<name>`:
/// referenced arrays with their chunk references, coordinates with their
/// values inline. Source paths are rewritten by the `remap` rows of
/// `target` (longest prefix first); a path that is then relative is taken
/// relative to `dir`, and a local path becomes a `file://` URL.
pub fn from_refs_table(dir: impl AsRef<Path>, target: Option<&str>) -> Result<Vec<VirtualArray>> {
    let dir = std::fs::canonicalize(dir.as_ref()).map_err(io)?;
    let table: Value = serde_json::from_reader(File::open(dir.join("arrays.json")).map_err(io)?)
        .map_err(|e| Error::Decode(format!("arrays.json: {e}")))?;
    let format = table["format_version"].as_str().unwrap_or("");
    if !format.starts_with(FORMAT_PREFIX) {
        return Err(Error::Unsupported(format!(
            "not an mdim-refs table: {format:?}"
        )));
    }
    let remap = remap_rules(&table, target);
    let rows: Vec<ArrayRow> = table["arrays"]
        .as_array()
        .ok_or_else(|| Error::Decode("arrays.json: no arrays".into()))?
        .iter()
        .map(ArrayRow::parse)
        .collect::<Result<_>>()?;
    let coords: HashMap<&str, &ArrayRow> = rows
        .iter()
        .filter(|r| r.storage != "referenced" && r.dims.len() == 1)
        .map(|r| (r.dims[0].as_str(), r))
        .collect();

    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        out.push(if row.storage == "referenced" {
            let (refs, inline) =
                read_refs(&dir.join(format!("refs_{}.parquet", row.name)), row, &|p| {
                    resolve(p, &remap, &dir)
                })?;
            referenced(row, &coords, refs, inline)?
        } else {
            coordinate(row)?
        });
    }
    Ok(out)
}

/// One row of the `arrays` table.
struct ArrayRow {
    name: String,
    storage: String,
    dims: Vec<String>,
    shape: Vec<u64>,
    chunks: Vec<u64>,
    dtype: Numpy,
    fill: Option<f64>,
    codec: Value,
    attrs: Map<String, Value>,
    affine: Option<(f64, f64)>,
    values: Option<Vec<f64>>,
}

impl ArrayRow {
    fn parse(v: &Value) -> Result<Self> {
        let name = v["array"]
            .as_str()
            .ok_or_else(|| bad("array", v))?
            .to_string();
        let u64s = |k: &str| -> Result<Vec<u64>> {
            v[k].as_array()
                .ok_or_else(|| bad(k, v))?
                .iter()
                .map(|x| x.as_u64().ok_or_else(|| bad(k, v)))
                .collect()
        };
        let shape = u64s("shape")?;
        let chunks = if v["chunks"].is_null() {
            shape.clone()
        } else {
            u64s("chunks")?
        };
        let fill = match &v["fill"] {
            Value::String(s) => Some(s.parse().map_err(|_| bad("fill", v))?),
            Value::Number(n) => n.as_f64(),
            _ => None,
        };
        let affine = match &v["affine"] {
            Value::Object(a) => Some((
                a.get("start")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| bad("affine", v))?,
                a.get("step")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| bad("affine", v))?,
            )),
            _ => None,
        };
        let values = match &v["values"] {
            Value::Array(a) => {
                let x: Vec<f64> = a
                    .iter()
                    .map(|x| match x {
                        Value::String(s) => s.parse().ok(),
                        _ => x.as_f64(),
                    })
                    .collect::<Option<_>>()
                    .ok_or_else(|| bad("values", v))?;
                if Some(&(x.len() as u64)) != shape.first() || shape.len() != 1 {
                    return Err(bad("values", v));
                }
                Some(x)
            }
            _ => None,
        };
        Ok(ArrayRow {
            storage: v["storage"].as_str().unwrap_or("").to_string(),
            dims: v["dims"]
                .as_array()
                .ok_or_else(|| bad("dims", v))?
                .iter()
                .map(|d| d.as_str().unwrap_or("").to_string())
                .collect(),
            shape,
            chunks,
            dtype: Numpy::parse(v["dtype"].as_str().ok_or_else(|| bad("dtype", v))?)?,
            fill,
            codec: v["codec"].clone(),
            attrs: v["attrs"].as_object().cloned().unwrap_or_default(),
            affine,
            values,
            name,
        })
    }

    /// The coordinate's values (affine expanded).
    fn coordinate_values(&self) -> Option<Vec<f64>> {
        if let Some(v) = &self.values {
            return Some(v.clone());
        }
        let (start, step) = self.affine?;
        Some(
            (0..self.shape[0])
                .map(|i| start + step * i as f64)
                .collect(),
        )
    }
}

fn bad(what: &str, row: &Value) -> Error {
    Error::Decode(format!("arrays.json: bad `{what}` in {row}"))
}

fn io(e: std::io::Error) -> Error {
    Error::Fetch(e.to_string())
}

/// A numpy dtype string such as `<i2`.
#[derive(Debug, Clone, Copy)]
struct Numpy {
    dtype: DataType,
    big_endian: bool,
}

impl Numpy {
    fn parse(s: &str) -> Result<Self> {
        let unsupported = || Error::Unsupported(format!("numpy dtype {s}"));
        let b = s.as_bytes();
        if b.len() < 3 {
            return Err(unsupported());
        }
        let dtype = match (b[1], &s[2..]) {
            (b'b', "1") => DataType::Bool,
            (b'u', "1") => DataType::U8,
            (b'u', "2") => DataType::U16,
            (b'u', "4") => DataType::U32,
            (b'u', "8") => DataType::U64,
            (b'i', "1") => DataType::I8,
            (b'i', "2") => DataType::I16,
            (b'i', "4") => DataType::I32,
            (b'i', "8") => DataType::I64,
            (b'f', "4") => DataType::F32,
            (b'f', "8") => DataType::F64,
            _ => return Err(unsupported()),
        };
        Ok(Numpy {
            dtype,
            big_endian: b[0] == b'>',
        })
    }

    fn size(&self) -> u64 {
        self.dtype.size().unwrap_or(1) as u64
    }

    fn endian(&self) -> &'static str {
        if self.big_endian {
            "big"
        } else {
            "little"
        }
    }
}

/// Zarr v3 codecs for a v2 `{compressor, filters}` pair: `bytes`, then the
/// filters in order, then the compressor.
fn codecs(codec: &Value, dtype: Numpy) -> Result<Vec<Value>> {
    let mut out = vec![json!({"name": "bytes", "configuration": {"endian": dtype.endian()}})];
    let filters = codec["filters"].as_array().cloned().unwrap_or_default();
    let compressor = Some(codec["compressor"].clone()).filter(|c| !c.is_null());
    for c in filters.iter().chain(compressor.iter()) {
        let level = |d: i64| c["level"].as_i64().unwrap_or(d);
        out.push(match c["id"].as_str().unwrap_or("") {
            "shuffle" => json!({"name": "numcodecs.shuffle", "configuration": {
                "elementsize": c["elementsize"].as_u64().unwrap_or(dtype.size())}}),
            "zlib" => json!({"name": "numcodecs.zlib", "configuration": {"level": level(1)}}),
            "gzip" => json!({"name": "gzip", "configuration": {"level": level(1)}}),
            "zstd" => {
                json!({"name": "zstd", "configuration": {"level": level(0), "checksum": false}})
            }
            _ => return Err(Error::Unsupported(format!("Zarr v2 codec {c}"))),
        });
    }
    Ok(out)
}

fn metadata(row: &ArrayRow, codecs: Vec<Value>, attrs: Map<String, Value>) -> Result<Value> {
    let fill = row.fill.unwrap_or(0.0);
    Ok(json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": row.shape,
        "data_type": wxy_zarr::write::data_type_name(row.dtype.dtype)?,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": row.chunks}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": wxy_zarr::write::fill_value_json(row.dtype.dtype, fill),
        "codecs": codecs,
        "dimension_names": row.dims,
        "attributes": attrs,
    }))
}

// By CF `standard_name`, `units` (any spelling, so BRAN2023's `yt_ocean` in
// `degrees_N` counts) or name, as wxy-zarr reads CF stores.
fn is_lat(c: &ArrayRow) -> bool {
    wxy_zarr::cf_axis(&c.name, &c.attrs) == Some(wxy_zarr::Axis::Lat)
}

fn is_lon(c: &ArrayRow) -> bool {
    wxy_zarr::cf_axis(&c.name, &c.attrs) == Some(wxy_zarr::Axis::Lon)
}

fn referenced(
    row: &ArrayRow,
    coords: &HashMap<&str, &ArrayRow>,
    refs: Vec<ChunkRef>,
    inline: Vec<(Vec<u64>, Vec<u8>)>,
) -> Result<VirtualArray> {
    let mut attrs = row.attrs.clone();
    if let Some(f) = row.fill {
        attrs.insert("_FillValue".into(), wxy_zarr::write::float_json(f));
    }
    let n = row.dims.len();
    if n >= 2 {
        let (ydim, xdim) = (&row.dims[n - 2], &row.dims[n - 1]);
        if let (Some(y), Some(x)) = (coords.get(ydim.as_str()), coords.get(xdim.as_str())) {
            if let (Some((y0, dy)), Some((x0, dx))) = (y.affine, x.affine) {
                // Centres to corners: GDAL order [x0, dx, 0, y0, 0, dy].
                let gt = [x0 - dx / 2.0, dx, 0.0, y0 - dy / 2.0, 0.0, dy];
                wxy_geozarr::Spatial::from_geotransform(
                    gt,
                    [ydim.as_str(), xdim.as_str()],
                    [row.shape[n - 2], row.shape[n - 1]],
                )
                .to_attrs(&mut attrs);
                // An assumption: CF latitude and longitude name no datum,
                // so WGS 84 is taken (right for OISST).
                if is_lat(y) && is_lon(x) {
                    wxy_geozarr::Proj {
                        code: Some("EPSG:4326".into()),
                        ..Default::default()
                    }
                    .to_attrs(&mut attrs);
                }
            }
        }
    }
    Ok(VirtualArray {
        path: format!("/{}", row.name),
        metadata: metadata(row, codecs(&row.codec, row.dtype)?, attrs)?,
        refs,
        inline,
    })
}

/// A coordinate as a one-chunk array whose values are stored inline.
fn coordinate(row: &ArrayRow) -> Result<VirtualArray> {
    let values = row
        .coordinate_values()
        .ok_or_else(|| Error::Unsupported(format!("coordinate {} has no values", row.name)))?;
    if row.dtype.dtype != DataType::F64 && row.dtype.dtype != DataType::F32 {
        return Err(Error::Unsupported(format!(
            "coordinate dtype for {}",
            row.name
        )));
    }
    let mut bytes = Vec::with_capacity(values.len() * row.dtype.size() as usize);
    for v in &values {
        match (row.dtype.dtype, row.dtype.big_endian) {
            (DataType::F32, false) => bytes.extend((*v as f32).to_le_bytes()),
            (DataType::F32, true) => bytes.extend((*v as f32).to_be_bytes()),
            (_, false) => bytes.extend(v.to_le_bytes()),
            (_, true) => bytes.extend(v.to_be_bytes()),
        }
    }
    let one = ArrayRow {
        chunks: row.shape.clone(),
        fill: None,
        name: row.name.clone(),
        storage: row.storage.clone(),
        dims: row.dims.clone(),
        shape: row.shape.clone(),
        dtype: row.dtype,
        codec: Value::Null,
        attrs: row.attrs.clone(),
        affine: row.affine,
        values: None,
    };
    let codecs = vec![json!({"name": "bytes", "configuration": {"endian": row.dtype.endian()}})];
    Ok(VirtualArray {
        path: format!("/{}", row.name),
        metadata: metadata(&one, codecs, row.attrs.clone())?,
        refs: Vec::new(),
        inline: vec![(vec![0; row.dims.len()], bytes)],
    })
}

/// `(prefix, replacement)` pairs for `target`, longest prefix first.
fn remap_rules(table: &Value, target: Option<&str>) -> Vec<(String, String)> {
    let Some(target) = target else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = table["remap"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| r["target"].as_str() == Some(target))
        .filter_map(|r| {
            Some((
                r["prefix"].as_str()?.into(),
                r["replacement"].as_str()?.into(),
            ))
        })
        .collect();
    out.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
    out
}

/// A source path as a URL: remapped, then made absolute against `dir`.
fn resolve(path: &str, remap: &[(String, String)], dir: &Path) -> Result<String> {
    let mut p = path.to_string();
    if let Some((prefix, rep)) = remap
        .iter()
        .find(|(prefix, _)| path.starts_with(prefix.as_str()))
    {
        p = format!("{rep}{}", &path[prefix.len()..]);
    }
    if p.contains("://") {
        return Ok(p);
    }
    let local = if Path::new(&p).is_absolute() {
        PathBuf::from(&p)
    } else {
        dir.join(&p)
    };
    let local = std::fs::canonicalize(&local)
        .map_err(|e| Error::Fetch(format!("{}: {e}", local.display())))?;
    Ok(format!("file://{}", local.display()))
}

/// Encoded chunk bytes held in the table itself, by chunk key.
type Inline = Vec<(Vec<u64>, Vec<u8>)>;

/// The present rows of one `refs_<array>` export: referenced rows as chunk
/// references, inline rows (`offset` null, `data` set) as encoded bytes.
fn read_refs(
    file: &Path,
    row: &ArrayRow,
    url: &dyn Fn(&str) -> Result<String>,
) -> Result<(Vec<ChunkRef>, Inline)> {
    let pq = |e: parquet::errors::ParquetError| Error::Decode(format!("{}: {e}", file.display()));
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(file).map_err(io)?)
        .map_err(pq)?
        .build()
        .map_err(pq)?;
    let n = row.dims.len();
    let counts: Vec<u64> = row
        .shape
        .iter()
        .zip(&row.chunks)
        .map(|(s, c)| s.div_ceil(*c))
        .collect();
    let mut urls: HashMap<String, String> = HashMap::new();
    let mut out = Vec::new();
    let mut inline = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| Error::Decode(e.to_string()))?;
        let dims: Vec<Int64Array> = (0..n)
            .map(|i| int_column(&batch, &format!("dim_{i}")))
            .collect::<Result<_>>()?;
        let present = int_column(&batch, "present")?;
        let offset = int_column(&batch, "offset")?;
        let size = int_column(&batch, "size")?;
        let path = column(&batch, "path")?;
        let path = path
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Decode("refs: `path` is not a string column".into()))?;
        let data = match batch.column_by_name("data") {
            Some(c) => Some(
                arrow_cast::cast(c, &ArrowType::Binary)
                    .map_err(|e| Error::Decode(format!("refs: `data`: {e}")))?,
            ),
            None => None,
        };
        let data = data
            .as_ref()
            .and_then(|d| d.as_any().downcast_ref::<BinaryArray>());
        for r in 0..batch.num_rows() {
            if present.is_null(r) || present.value(r) == 0 {
                continue;
            }
            let key: Vec<u64> = dims.iter().map(|d| d.value(r) as u64).collect();
            if key.iter().zip(&counts).any(|(k, c)| k >= c) {
                return Err(Error::Decode(format!(
                    "refs_{}: chunk {key:?} is outside the grid",
                    row.name
                )));
            }
            if offset.is_null(r) {
                match data.filter(|d| !d.is_null(r)) {
                    Some(d) => inline.push((key, d.value(r).to_vec())),
                    None => {
                        return Err(Error::Decode(format!(
                            "refs_{}: chunk {key:?} is present with no offset and no data",
                            row.name
                        )))
                    }
                }
                continue;
            }
            let p = path.value(r);
            let u = match urls.get(p) {
                Some(u) => u.clone(),
                None => {
                    let u = url(p)?;
                    urls.insert(p.to_string(), u.clone());
                    u
                }
            };
            out.push(ChunkRef {
                key,
                url: u,
                offset: offset.value(r) as u64,
                length: size.value(r) as u64,
            });
        }
    }
    Ok((out, inline))
}

fn column(batch: &RecordBatch, name: &str) -> Result<arrow_array::ArrayRef> {
    batch
        .column_by_name(name)
        .cloned()
        .ok_or_else(|| Error::Decode(format!("refs: no `{name}` column")))
}

/// An integer column of any width as Int64.
fn int_column(batch: &RecordBatch, name: &str) -> Result<Int64Array> {
    let c = arrow_cast::cast(&column(batch, name)?, &ArrowType::Int64)
        .map_err(|e| Error::Decode(format!("refs: `{name}`: {e}")))?;
    Ok(c.as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to Int64")
        .clone())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BinaryArray, Int64Array, Int8Array, StringArray};
    use parquet::arrow::ArrowWriter;

    use super::*;

    /// A 1 x 3 chunk grid: one referenced chunk (remapped), one inline,
    /// one absent.
    #[test]
    fn rows_split_into_refs_and_inline() {
        let dir = std::env::temp_dir().join(format!("wxy-refs-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("local")).unwrap();
        std::fs::write(dir.join("local/a.bin"), [0u8; 16]).unwrap();
        let table = json!({
            "format_version": "mdim-refs/0.2",
            "remap": [
                {"target": "t", "prefix": "/src/", "replacement": "elsewhere/"},
                {"target": "t", "prefix": "/src/a", "replacement": "local/a"},
            ],
            "arrays": [{
                "array": "v", "kind": "data", "storage": "referenced",
                "dims": ["y", "x"], "shape": [2, 6], "chunks": [2, 2],
                "dtype": "<i4", "fill": "-1",
                "codec": {"compressor": {"id": "zlib", "level": 1}, "filters": null},
                "attrs": {}, "affine": null, "values": null,
            }],
        });
        std::fs::write(dir.join("arrays.json"), table.to_string()).unwrap();
        let cols: Vec<(&str, ArrayRef)> = vec![
            ("dim_0", Arc::new(Int64Array::from(vec![0, 0, 0]))),
            ("dim_1", Arc::new(Int64Array::from(vec![0, 1, 2]))),
            ("present", Arc::new(Int8Array::from(vec![1, 1, 0]))),
            (
                "path",
                Arc::new(StringArray::from(vec![Some("/src/a.bin"), None, None])),
            ),
            (
                "offset",
                Arc::new(Int64Array::from(vec![Some(4), None, None])),
            ),
            (
                "size",
                Arc::new(Int64Array::from(vec![Some(8), None, None])),
            ),
            (
                "data",
                Arc::new(BinaryArray::from(vec![None, Some(&b"xyz"[..]), None])),
            ),
        ];
        let batch = RecordBatch::try_from_iter(cols).unwrap();
        let f = File::create(dir.join("refs_v.parquet")).unwrap();
        let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let arrays = from_refs_table(&dir, Some("t")).unwrap();
        assert_eq!(arrays.len(), 1);
        let a = &arrays[0];
        let local = std::fs::canonicalize(dir.join("local/a.bin")).unwrap();
        assert_eq!(
            a.refs,
            vec![ChunkRef {
                key: vec![0, 0],
                url: format!("file://{}", local.display()),
                offset: 4,
                length: 8,
            }]
        );
        assert_eq!(a.inline, vec![(vec![0, 1], b"xyz".to_vec())]);
        assert_eq!(a.metadata["fill_value"], json!(-1));
        assert_eq!(a.metadata["codecs"][1]["name"], json!("numcodecs.zlib"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Latitude and longitude are recognised by any CF unit spelling or
    /// `standard_name`, not only by name (BRAN2023's `yt_ocean` is in
    /// `degrees_N`).
    #[test]
    fn cf_axes_by_units_or_standard_name() {
        let axis = |name: &str, attrs: Value| {
            ArrayRow::parse(&json!({
                "array": name, "kind": "coord", "storage": "affine",
                "dims": [name], "shape": [3], "chunks": [3], "dtype": "<f8",
                "attrs": attrs, "affine": {"start": 0.0, "step": 1.0},
            }))
            .unwrap()
        };
        for u in ["degrees_north", "degree_north", "degrees_N", "degree_N"] {
            assert!(is_lat(&axis("yt_ocean", json!({"units": u}))), "{u}");
            assert!(!is_lon(&axis("yt_ocean", json!({"units": u}))), "{u}");
        }
        for u in ["degrees_east", "degree_east", "degrees_E", "degree_E"] {
            assert!(is_lon(&axis("xt_ocean", json!({"units": u}))), "{u}");
        }
        assert!(is_lat(&axis("y", json!({"standard_name": "latitude"}))));
        assert!(is_lon(&axis("x", json!({"standard_name": "longitude"}))));
        assert!(!is_lat(&axis("y", json!({"units": "m"}))));
        assert!(is_lat(&axis("lat", json!({}))));
    }
}
