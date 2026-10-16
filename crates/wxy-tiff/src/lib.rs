//! # wxy-tiff
//!
//! GeoTIFF and Cloud Optimized GeoTIFF for wxylib, through
//! [async-tiff](https://github.com/developmentseed/async-tiff). A thin
//! adapter (design doc 03, decision A4): file metadata becomes a
//! [`Collection`], and each resolution level becomes a [`ChunkSource`] whose
//! chunks are the file's tiles.
//!
//! The collection has one domain per level, with dims `(band, y, x)`, an
//! affine mapping on `(y, x)` from the GeoTIFF tags, and `band` unmapped.
//! Overview levels are related to level 0 by
//! [`RelationKind::Overview`]. There is one field, `"data"`, on level 0.
//!
//! Stage S2 reads tiled TIFFs, chunky or planar. Stripped TIFFs are reported
//! as unsupported.

use std::sync::Arc;

use async_tiff::decoder::DecoderRegistry;
use async_tiff::metadata::cache::ReadaheadMetadataCache;
use async_tiff::metadata::TiffMetadataReader;
use async_tiff::reader::{AsyncFileReader, ObjectReader};
use async_tiff::tags::PlanarConfiguration;
use async_tiff::{ImageFileDirectory, Tile, TypedArray, TIFF};
use async_trait::async_trait;
use wxy_domain::{
    Affine, Collection, DataType, Dim, Domain, Field, Frame, Mapping, Positions, Relation,
    RelationKind,
};
use wxy_source::{Chunk, ChunkGrid, ChunkKey, ChunkSource, Encoded, Error, Result, Values};

/// An opened TIFF: its collection and one chunk source per level.
pub struct TiffDataset {
    pub collection: Collection,
    /// Level 0 first, then overviews from finest to coarsest. Index `k`
    /// matches `collection.domains[k]`.
    pub levels: Vec<IfdSource>,
}

/// One TIFF image (a resolution level) as a chunk source.
pub struct IfdSource {
    tiff: Arc<TIFF>,
    ifd: usize,
    reader: Arc<dyn AsyncFileReader>,
    registry: Arc<DecoderRegistry>,
    grid: ChunkGrid,
    dtype: DataType,
}

impl IfdSource {
    fn ifd(&self) -> &ImageFileDirectory {
        &self.tiff.ifds()[self.ifd]
    }
}

/// Open a local file.
#[cfg(not(target_arch = "wasm32"))]
pub async fn open_path(path: impl AsRef<std::path::Path>) -> Result<TiffDataset> {
    let path = std::fs::canonicalize(path.as_ref()).map_err(|e| Error::Fetch(e.to_string()))?;
    let store = Arc::new(object_store::local::LocalFileSystem::new());
    let location = object_store::path::Path::from_filesystem_path(&path)
        .map_err(|e| Error::Fetch(e.to_string()))?;
    open(Arc::new(ObjectReader::new(store, location))).await
}

/// Open from any object store location.
pub async fn open_store(
    store: Arc<dyn object_store::ObjectStore>,
    location: object_store::path::Path,
) -> Result<TiffDataset> {
    open(Arc::new(ObjectReader::new(store, location))).await
}

/// Open from an async-tiff reader.
pub async fn open(reader: Arc<dyn AsyncFileReader>) -> Result<TiffDataset> {
    let cache = ReadaheadMetadataCache::new(reader.clone());
    let mut meta = TiffMetadataReader::try_open(&cache)
        .await
        .map_err(|e| Error::Fetch(e.to_string()))?;
    let ifds = meta
        .read_all_ifds(&cache)
        .await
        .map_err(|e| Error::Fetch(e.to_string()))?;
    let tiff = Arc::new(TIFF::new(ifds, meta.endianness()));
    build(tiff, reader)
}

fn build(tiff: Arc<TIFF>, reader: Arc<dyn AsyncFileReader>) -> Result<TiffDataset> {
    let ifds = tiff.ifds();
    let first = ifds
        .first()
        .ok_or_else(|| Error::Decode("TIFF has no images".into()))?;

    // Level 0, then reduced-resolution images that are not masks.
    let mut level_ifds = vec![0usize];
    for (k, ifd) in ifds.iter().enumerate().skip(1) {
        let t = ifd.new_subfile_type().unwrap_or(0);
        if t & 1 == 1 && t & 4 == 0 {
            level_ifds.push(k);
        }
    }

    let affine0 = geotransform(first);
    let frame = frame(first);
    let registry = Arc::new(DecoderRegistry::default());
    let bands = first.samples_per_pixel() as u64;
    let (w0, h0) = (first.image_width() as f64, first.image_height() as f64);

    let mut collection = Collection::new("tiff");
    let mut levels = Vec::new();
    for (lvl, &k) in level_ifds.iter().enumerate() {
        let ifd = &ifds[k];
        let (w, h) = (ifd.image_width() as u64, ifd.image_height() as u64);
        let (tw, th) = match (ifd.tile_width(), ifd.tile_height()) {
            (Some(tw), Some(th)) => (tw as u64, th as u64),
            _ => {
                return Err(Error::Unsupported(
                    "stripped TIFF (S2 reads tiled TIFFs)".into(),
                ))
            }
        };
        let dtype = dtype(ifd)?;

        let mut domain = Domain::new(vec![
            Dim::new("band", bands),
            Dim::new("y", h),
            Dim::new("x", w),
        ]);
        if let Some(a) = affine0 {
            // Overviews keep the extent; pixel size scales by the size ratio.
            let (fx, fy) = (w0 / w as f64, h0 / h as f64);
            let g = a.0;
            let a = Affine([g[0], g[1] * fx, g[2] * fy, g[3], g[4] * fx, g[5] * fy]);
            let m = domain.add_mapping(Mapping::new(
                vec![1, 2],
                Positions::Affine(a),
                frame.clone(),
            ));
            domain.set_primary(m);
        }
        let d = collection.add_domain(domain);
        if lvl > 0 {
            collection.related.push(Relation {
                kind: RelationKind::Overview {
                    factor: vec![w0 / w as f64, h0 / h as f64],
                },
                from: 0,
                to: d,
            });
        }
        levels.push(IfdSource {
            tiff: tiff.clone(),
            ifd: k,
            reader: reader.clone(),
            registry: registry.clone(),
            grid: ChunkGrid::new(vec![bands, h, w], vec![bands, th, tw]),
            dtype,
        });
    }

    let mut field = Field::new("data", levels[0].dtype, 0);
    field.values.nodata = first
        .gdal_nodata()
        .and_then(|s| s.trim().parse::<f64>().ok());
    collection.add_field(field);
    if let Some(e) = first.geo_key_directory().and_then(|g| g.epsg_code()) {
        collection
            .attrs
            .insert("geotiff:epsg".into(), e.to_string());
    }
    Ok(TiffDataset { collection, levels })
}

/// The affine transform from ModelTransformation, or ModelPixelScale plus
/// ModelTiepoint, in GDAL order. PixelIsPoint rasters are shifted by half a
/// pixel, as GDAL does by default.
fn geotransform(ifd: &ImageFileDirectory) -> Option<Affine> {
    let mut gt = if let Some(m) = ifd.model_transformation().filter(|m| m.len() >= 16) {
        [m[3], m[0], m[1], m[7], m[4], m[5]]
    } else {
        let s = ifd.model_pixel_scale()?;
        let t = ifd.model_tiepoint()?;
        if s.len() < 2 || t.len() < 6 {
            return None;
        }
        [
            t[3] - t[0] * s[0],
            s[0],
            0.0,
            t[4] + t[1] * s[1],
            0.0,
            -s[1],
        ]
    };
    let point = ifd
        .geo_key_directory()
        .and_then(|g| g.raster_type)
        .is_some_and(|r| r == 2);
    if point {
        gt[0] -= 0.5 * gt[1] + 0.5 * gt[2];
        gt[3] -= 0.5 * gt[4] + 0.5 * gt[5];
    }
    Some(Affine(gt))
}

/// The frame from GeoKeys. User-defined CRSs (32767) are not interpreted in
/// S2 and are reported as `Frame::Other`.
fn frame(ifd: &ImageFileDirectory) -> Frame {
    match ifd.geo_key_directory().and_then(|g| g.epsg_code()) {
        Some(32767) => Frame::Other("geotiff:user-defined".into()),
        Some(code) => Frame::crs(format!("EPSG:{code}")),
        None if ifd.geo_key_directory().is_some() => Frame::Other("geotiff:unknown".into()),
        None => Frame::Index,
    }
}

fn dtype(ifd: &ImageFileDirectory) -> Result<DataType> {
    use async_tiff::DataType as T;
    Ok(match tiff_dtype(ifd) {
        Some(T::Bool) => DataType::Bool,
        Some(T::UInt8) | None => DataType::U8,
        Some(T::UInt16) => DataType::U16,
        Some(T::UInt32) => DataType::U32,
        Some(T::UInt64) => DataType::U64,
        Some(T::Int8) => DataType::I8,
        Some(T::Int16) => DataType::I16,
        Some(T::Int32) => DataType::I32,
        Some(T::Int64) => DataType::I64,
        Some(T::Float32) => DataType::F32,
        Some(T::Float64) => DataType::F64,
    })
}

fn tiff_dtype(ifd: &ImageFileDirectory) -> Option<async_tiff::DataType> {
    use async_tiff::tags::SampleFormat as S;
    use async_tiff::DataType as T;
    let bits = *ifd.bits_per_sample().first()?;
    let fmt = ifd.sample_format().first().copied().unwrap_or(S::Uint);
    Some(match (fmt, bits) {
        (S::Uint, 1) => T::Bool,
        (S::Uint, 8) => T::UInt8,
        (S::Uint, 16) => T::UInt16,
        (S::Uint, 32) => T::UInt32,
        (S::Uint, 64) => T::UInt64,
        (S::Int, 8) => T::Int8,
        (S::Int, 16) => T::Int16,
        (S::Int, 32) => T::Int32,
        (S::Int, 64) => T::Int64,
        (S::Float, 32) => T::Float32,
        (S::Float, 64) => T::Float64,
        _ => return None,
    })
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl ChunkSource for IfdSource {
    fn grid(&self) -> &ChunkGrid {
        &self.grid
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    async fn fetch(&self, keys: &[ChunkKey]) -> Result<Vec<Encoded>> {
        let xy: Vec<(usize, usize)> = keys
            .iter()
            .map(|k| (k.0[2] as usize, k.0[1] as usize))
            .collect();
        let tiles = self
            .ifd()
            .fetch_tiles(&xy, self.reader.as_ref())
            .await
            .map_err(|e| Error::Fetch(e.to_string()))?;
        Ok(keys
            .iter()
            .cloned()
            .zip(tiles)
            .map(|(key, tile)| Encoded {
                key,
                payload: Box::new(tile),
            })
            .collect())
    }

    fn decode(&self, encoded: Encoded) -> Result<Chunk> {
        let tile = encoded
            .payload
            .downcast::<Tile>()
            .map_err(|_| Error::Decode("payload is not a TIFF tile".into()))?;
        let array = tile
            .decode(&self.registry)
            .map_err(|e| Error::Decode(e.to_string()))?;
        let (data, shape, _) = array.into_inner();
        let planar = self.ifd().planar_configuration() == PlanarConfiguration::Planar;
        // Stored order: chunky (h, w, planes), planar (planes, h, w).
        let (bands, h, w) = if planar {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[2], shape[0], shape[1])
        };
        let values = to_band_major(data, planar, bands, h, w);
        let window = self.grid.window(&encoded.key);
        Ok(Chunk {
            key: encoded.key,
            window,
            stored: vec![bands as u64, h as u64, w as u64],
            values,
        })
    }
}

/// async-tiff's typed array as `(band, y, x)` values.
fn to_band_major(data: TypedArray, planar: bool, bands: usize, h: usize, w: usize) -> Values {
    fn reorder<T: Copy>(v: Vec<T>, planar: bool, bands: usize, h: usize, w: usize) -> Vec<T> {
        if planar || bands == 1 {
            return v;
        }
        let mut out = Vec::with_capacity(v.len());
        for b in 0..bands {
            for p in 0..h * w {
                out.push(v[p * bands + b]);
            }
        }
        out
    }
    match data {
        TypedArray::Bool(v) => Values::U8(
            reorder(v, planar, bands, h, w)
                .into_iter()
                .map(u8::from)
                .collect(),
        ),
        TypedArray::UInt8(v) => Values::U8(reorder(v, planar, bands, h, w)),
        TypedArray::UInt16(v) => Values::U16(reorder(v, planar, bands, h, w)),
        TypedArray::UInt32(v) => Values::U32(reorder(v, planar, bands, h, w)),
        TypedArray::UInt64(v) => Values::U64(reorder(v, planar, bands, h, w)),
        TypedArray::Int8(v) => Values::I8(reorder(v, planar, bands, h, w)),
        TypedArray::Int16(v) => Values::I16(reorder(v, planar, bands, h, w)),
        TypedArray::Int32(v) => Values::I32(reorder(v, planar, bands, h, w)),
        TypedArray::Int64(v) => Values::I64(reorder(v, planar, bands, h, w)),
        TypedArray::Float32(v) => Values::F32(reorder(v, planar, bands, h, w)),
        TypedArray::Float64(v) => Values::F64(reorder(v, planar, bands, h, w)),
    }
}
