//! Kerchunk Parquet references (fsspec's `LazyReferenceMapper` layout) as
//! virtual arrays.
//!
//! A kerchunk Parquet store is a directory holding `.zmetadata` (Zarr v2
//! `.zgroup`, `.zarray` and `.zattrs` documents under `metadata`, and
//! `record_size`) and, per array, `refs.<i>.parq` files of `record_size`
//! rows each: one row per chunk in C order over the chunk grid, with
//! `path`, `offset`, `size` and `raw`. A row with a path is a byte range of
//! that source, a row with `raw` holds the chunk's encoded bytes, a row
//! with neither is a chunk never written.
//!
//! The arrays are read into the same form as an mdim-refs table
//! ([`crate::table`]): Zarr v2 codecs become v3, `_ARRAY_DIMENSIONS`
//! names the dims, and each dimension coordinate is read (through the
//! references, so inline or not) and kept as affine `{start, step}` when
//! its steps agree to 1e-6 of a step, else as inline values, so a regular
//! latitude and longitude give the data GeoZarr `spatial:` attributes as
//! the table does. Attributes that R writes as one-element lists are
//! unwrapped.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BinaryArray, Int64Array};
use arrow_schema::DataType as ArrowType;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Map, Value};
use wxy_source::{Error, Result};

use crate::table::{build, referenced, resolve_url, ArrayRow, Inline};
use crate::{ChunkRef, RefsStore, VirtualArray};

/// Read the kerchunk Parquet store at `location` (a URL, a GDAL
/// `/vsis3/` or `/vsicurl/` path, or a local directory) as virtual arrays, one per array at `/<name>`. Source paths
/// are taken as [`crate::from_refs_table_at`] takes them: GDAL `/vsicurl/`
/// and `/vsis3/` paths as their URLs, relative paths against the store.
pub async fn from_kerchunk_at(location: &str) -> Result<Vec<VirtualArray>> {
    let location = &crate::location::vsi_to_url(location);
    let dir = if location.contains("://") {
        format!("{}/", location.trim_end_matches('/'))
    } else {
        let p = std::fs::canonicalize(location).map_err(|e| Error::Fetch(e.to_string()))?;
        format!("file://{}/", p.display())
    };
    let base = url::Url::parse(&dir).map_err(|e| Error::Fetch(format!("{dir}: {e}")))?;
    let zmeta: Value = serde_json::from_slice(&get(&format!("{dir}.zmetadata")).await?)
        .map_err(|e| Error::Decode(format!(".zmetadata: {e}")))?;
    let record = zmeta["record_size"]
        .as_u64()
        .filter(|r| *r > 0)
        .ok_or_else(|| Error::Decode(".zmetadata: no record_size".into()))?;
    let meta = zmeta["metadata"]
        .as_object()
        .ok_or_else(|| Error::Decode(".zmetadata: no metadata".into()))?;
    let doc = |k: &str| -> Result<Value> {
        match meta.get(k) {
            // some writers hold each document as a JSON string
            Some(Value::String(s)) => {
                serde_json::from_str(s).map_err(|e| Error::Decode(format!("{k}: {e}")))
            }
            Some(v) => Ok(v.clone()),
            None => Ok(json!({})),
        }
    };

    let mut names: Vec<String> = meta
        .keys()
        .filter_map(|k| k.strip_suffix("/.zarray"))
        .map(String::from)
        .collect();
    names.sort();
    let mut rows = Vec::new();
    let mut chunks: HashMap<String, (Vec<ChunkRef>, Inline)> = HashMap::new();
    let url = |p: &str| resolve_url(p, &[], &base);
    for name in &names {
        let za = doc(&format!("{name}/.zarray"))?;
        let attrs = doc(&format!("{name}/.zattrs"))?;
        let row = ArrayRow::parse(&array_row(name, &za, &attrs)?)?;
        let mut got = read_chunks(&dir, &row, record, &url).await?;
        if row.codec.is_null() {
            // an uncompressed inline chunk can hold more than the chunk
            // (an index that kept a whole coordinate for a shorter array):
            // keep the chunk's own bytes
            let n = row.chunks.iter().product::<u64>() * row.dtype.dtype.size().unwrap_or(1) as u64;
            for (_, b) in got.1.iter_mut() {
                b.truncate(n as usize);
            }
        }
        chunks.insert(name.clone(), got);
        rows.push(row);
    }

    // Dimension coordinates: read their values, keep them as the table
    // keeps coordinates.
    let used: Vec<String> = rows
        .iter()
        .filter(|r| r.dims.len() > 1)
        .flat_map(|r| r.dims.clone())
        .collect();
    for row in rows.iter_mut() {
        let float = matches!(
            row.dtype.dtype,
            wxy_domain::DataType::F32 | wxy_domain::DataType::F64
        );
        if row.dims.len() != 1 || row.dims[0] != row.name || !used.contains(&row.name) || !float {
            continue;
        }
        let (refs, inline) = chunks[&row.name].clone();
        let one = referenced(row, &HashMap::new(), refs, inline)?;
        let store = Arc::new(RefsStore::new(&[one], None)?);
        let v = wxy_zarr::read_values(store, &format!("/{}", row.name)).await?;
        row.chunks = row.shape.clone();
        row.fill = None;
        row.codec = Value::Null;
        match regular(&v) {
            Some(a) => {
                row.storage = "affine".into();
                row.affine = Some(a);
            }
            None => {
                row.storage = "inline".into();
                row.values = Some(v);
            }
        }
    }

    build(&rows, |row| {
        chunks
            .get(&row.name)
            .cloned()
            .ok_or_else(|| Error::Decode(format!("{}: no references read", row.name)))
    })
}

/// The whole object at `url`.
async fn get(url: &str) -> Result<Bytes> {
    let (base, path) = crate::location::split(url)?;
    let store = crate::location::store(&base)?;
    object_store::ObjectStoreExt::get(&store, &path)
        .await
        .map_err(|e| Error::Fetch(format!("{url}: {e}")))?
        .bytes()
        .await
        .map_err(|e| Error::Fetch(format!("{url}: {e}")))
}

/// `(start, step)` when the values step evenly (to 1e-6 of a step).
fn regular(v: &[f64]) -> Option<(f64, f64)> {
    if v.len() < 3 {
        return None;
    }
    let step = (v[v.len() - 1] - v[0]) / (v.len() - 1) as f64;
    let tol = step.abs() * 1e-6;
    v.windows(2)
        .all(|w| ((w[1] - w[0]) - step).abs() <= tol)
        .then_some((v[0], step))
}

/// One-element lists as their element (R writes scalars so).
fn unwrap_attrs(attrs: &Value) -> Map<String, Value> {
    attrs
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| *k != "_ARRAY_DIMENSIONS")
        .map(|(k, v)| {
            let v = match v {
                Value::Array(a) if a.len() == 1 => a[0].clone(),
                _ => v.clone(),
            };
            (k.clone(), v)
        })
        .collect()
}

/// A `.zarray` and its `.zattrs` as a row of the mdim-refs `arrays` table.
fn array_row(name: &str, za: &Value, attrs: &Value) -> Result<Value> {
    if za["order"].as_str().unwrap_or("C") != "C" {
        return Err(Error::Unsupported(format!("{name}: Fortran order")));
    }
    let shape = za["shape"].clone();
    let n = shape.as_array().map_or(0, Vec::len);
    let dims = match &attrs["_ARRAY_DIMENSIONS"] {
        Value::Array(d) if d.len() == n => Value::Array(d.clone()),
        _ => json!((0..n).map(|i| format!("dim_{i}")).collect::<Vec<_>>()),
    };
    let fill = match &za["fill_value"] {
        Value::Number(x) => json!(x.to_string()),
        Value::String(s) => json!(s),
        _ => Value::Null,
    };
    let codec = if za["compressor"].is_null() && za["filters"].is_null() {
        Value::Null
    } else {
        json!({"compressor": za["compressor"], "filters": za["filters"]})
    };
    Ok(json!({
        "array": name,
        "storage": "referenced",
        "dims": dims,
        "shape": shape,
        "chunks": za["chunks"],
        "dtype": za["dtype"],
        "fill": fill,
        "codec": codec,
        "attrs": unwrap_attrs(attrs),
        "affine": null,
        "values": null,
    }))
}

/// The chunks of `row` from its `refs.<i>.parq` files.
async fn read_chunks(
    dir: &str,
    row: &ArrayRow,
    record: u64,
    url: &dyn Fn(&str) -> Result<String>,
) -> Result<(Vec<ChunkRef>, Inline)> {
    let counts: Vec<u64> = row
        .shape
        .iter()
        .zip(&row.chunks)
        .map(|(s, c)| s.div_ceil(*c))
        .collect();
    let total: u64 = counts.iter().product();
    let mut refs = Vec::new();
    let mut inline = Vec::new();
    let mut urls: HashMap<String, String> = HashMap::new();
    for file in 0..total.div_ceil(record) {
        let label = format!("{dir}{}/refs.{file}.parq", row.name);
        let bytes = get(&label).await?;
        let pq = |e: parquet::errors::ParquetError| Error::Decode(format!("{label}: {e}"));
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(pq)?
            .build()
            .map_err(pq)?;
        let mut i = file * record;
        for batch in reader {
            let batch = batch.map_err(|e| Error::Decode(format!("{label}: {e}")))?;
            let col = |name: &str, t: &ArrowType| -> Result<Option<arrow_array::ArrayRef>> {
                batch
                    .column_by_name(name)
                    .map(|c| arrow_cast::cast(c, t))
                    .transpose()
                    .map_err(|e| Error::Decode(format!("{label}: `{name}`: {e}")))
            };
            let path = col("path", &ArrowType::Utf8)?;
            let path = path
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
            let offset = col("offset", &ArrowType::Int64)?;
            let offset = offset
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let size = col("size", &ArrowType::Int64)?;
            let size = size
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let raw = col("raw", &ArrowType::Binary)?;
            let raw = raw
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());
            for r in 0..batch.num_rows() {
                if i >= total {
                    break;
                }
                let key = unravel(i, &counts);
                i += 1;
                match (
                    path.filter(|p| !p.is_null(r)),
                    raw.filter(|d| !d.is_null(r)),
                ) {
                    (Some(p), _) => {
                        let (Some(o), Some(s)) = (offset, size) else {
                            return Err(Error::Decode(format!("{label}: no offset or size")));
                        };
                        let length = s.value(r) as u64;
                        if length == 0 {
                            return Err(Error::Unsupported(format!(
                                "{label}: a whole-file reference (size 0)"
                            )));
                        }
                        let p = p.value(r);
                        let u = match urls.get(p) {
                            Some(u) => u.clone(),
                            None => {
                                let u = url(p)?;
                                urls.insert(p.to_string(), u.clone());
                                u
                            }
                        };
                        refs.push(ChunkRef {
                            key,
                            source: p.to_string(),
                            url: u,
                            offset: o.value(r) as u64,
                            length,
                        });
                    }
                    (None, Some(d)) => inline.push((key, d.value(r).to_vec())),
                    (None, None) => {}
                }
            }
        }
    }
    Ok((refs, inline))
}

/// The chunk key of position `i` in C order over `counts`.
fn unravel(mut i: u64, counts: &[u64]) -> Vec<u64> {
    let mut key = vec![0; counts.len()];
    for (k, c) in key.iter_mut().zip(counts).rev() {
        *k = i % c;
        i /= c;
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;

    #[test]
    fn keys_in_c_order() {
        assert_eq!(unravel(0, &[2, 3]), vec![0, 0]);
        assert_eq!(unravel(4, &[2, 3]), vec![1, 1]);
        assert_eq!(regular(&[1.0, 2.0, 3.0]), Some((1.0, 1.0)));
        assert_eq!(regular(&[1.0, 2.0, 4.0]), None);
    }

    /// A 3 x 4 array in 3 x 2 chunks over an inline y and x: one chunk
    /// referenced (through a /vsicurl/ path), one absent; records of one
    /// row, so two files.
    #[test]
    fn store_reads_as_a_table_would() {
        let dir = std::env::temp_dir().join(format!("wxy-kerchunk-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for a in ["v", "y", "x"] {
            std::fs::create_dir_all(dir.join(a)).unwrap();
        }
        let zarray = |shape: &[u64], chunks: &[u64], dtype: &str| {
            json!({"zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": dtype,
                   "compressor": null, "filters": null, "fill_value": null, "order": "C"})
        };
        let zmeta = json!({"record_size": 1, "metadata": {
            ".zgroup": {"zarr_format": 2},
            "v/.zarray": zarray(&[3, 4], &[3, 2], "<i2"),
            "v/.zattrs": {"_ARRAY_DIMENSIONS": ["y", "x"], "units": ["K"]},
            "y/.zarray": zarray(&[3], &[3], "<f8").to_string(),
            "y/.zattrs": {"_ARRAY_DIMENSIONS": ["y"], "units": ["degrees_north"]},
            "x/.zarray": zarray(&[4], &[4], "<f8"),
            "x/.zattrs": {"_ARRAY_DIMENSIONS": ["x"], "units": ["degrees_east"]},
        }});
        std::fs::write(dir.join(".zmetadata"), zmeta.to_string()).unwrap();
        let write = |file: &str, path: Option<&str>, raw: Option<Vec<u8>>| {
            let cols: Vec<(&str, ArrayRef)> = vec![
                ("path", Arc::new(StringArray::from(vec![path]))),
                ("offset", Arc::new(Int64Array::from(vec![Some(8)]))),
                ("size", Arc::new(Int64Array::from(vec![Some(16)]))),
                ("raw", Arc::new(BinaryArray::from(vec![raw.as_deref()]))),
            ];
            let batch = RecordBatch::try_from_iter(cols).unwrap();
            let f = std::fs::File::create(dir.join(file)).unwrap();
            let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        };
        let f64s = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        write("v/refs.0.parq", Some("/vsicurl/https://h.org/a.nc"), None);
        write("v/refs.1.parq", None, None);
        write("y/refs.0.parq", None, Some(f64s(&[-10.0, 0.0, 10.0])));
        write("x/refs.0.parq", None, Some(f64s(&[0.5, 1.5, 2.5, 3.5])));

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let arrays = rt
            .block_on(from_kerchunk_at(dir.to_str().unwrap()))
            .unwrap();
        let v = arrays.iter().find(|a| a.path == "/v").unwrap();
        assert_eq!(
            v.refs,
            vec![ChunkRef {
                key: vec![0, 0],
                source: "/vsicurl/https://h.org/a.nc".into(),
                url: "https://h.org/a.nc".into(),
                offset: 8,
                length: 16,
            }]
        );
        assert_eq!(v.metadata["attributes"]["units"], json!("K"));
        assert_eq!(v.metadata["dimension_names"], json!(["y", "x"]));
        // y and x step evenly, so the data carries a GeoZarr transform
        // (corner of the first cell) and EPSG:4326
        assert_eq!(
            v.metadata["attributes"]["spatial:transform"],
            json!([1.0, 0.0, 0.0, 0.0, 10.0, -15.0])
        );
        let x = arrays.iter().find(|a| a.path == "/x").unwrap();
        assert_eq!(x.inline.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
