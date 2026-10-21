//! # wxy-predictor
//!
//! TIFF predictors as Zarr codecs for zarrs, so the tiles of a predicted
//! TIFF can be read in place as virtual Zarr chunks (design doc 03, S3b; the
//! GHRSST MUR COGs on source.coop are ZSTD with Predictor=2).
//!
//! [`HorizontalDeltaCodec`] undoes TIFF Predictor=2: each row of a tile
//! stores the first sample, then differences from the previous sample.
//! Decoding is a running sum along the last axis, restarted at every row.
//! As in libtiff, the arithmetic wraps on the unsigned integer of the
//! element's width, whatever the data type. The codec is array to array and
//! sits ahead of `bytes`, so it sees native-endian elements.
//!
//! The Zarr v3 name is `HorizontalDeltaCodec`, the name virtual-tiff
//! registers for zarr-python, with no configuration. The Zarr v2 form that
//! kerchunk stores carry, imagecodecs' `imagecodecs_delta` with `axis: -1`
//! and `dist: 1`, maps onto it in `wxy-refs`.
//!
//! The crate is geo-free. Registration is by `inventory`, so depending on it
//! (and naming it, so the linker keeps it: [`register`]) is enough for
//! zarrs to open arrays that use the codec.
//!
//! Not yet: Predictor=3 (floating point, imagecodecs `floatpred`), which
//! byte-shuffles before differencing; REMA v2 and COP30 need it (design m10).

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, Codec, CodecError, CodecMetadataOptions,
    CodecOptions, CodecPluginV3, CodecTraits, CodecTraitsV3, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::Configuration;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// TIFF Predictor=2 (horizontal differencing) as an array-to-array codec.
#[derive(Clone, Debug, Default)]
pub struct HorizontalDeltaCodec;

zarrs_plugin::impl_extension_aliases!(HorizontalDeltaCodec,
  v3: "HorizontalDeltaCodec", ["tiff.horizontal_delta"]
);

inventory::submit! {
    CodecPluginV3::new::<HorizontalDeltaCodec>()
}

/// Call once (a no-op) from a crate that relies on the codec, so the
/// registration is linked in.
pub fn register() {}

impl CodecTraitsV3 for HorizontalDeltaCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        if !metadata.configuration_is_none_or_empty() {
            return Err(PluginCreateError::Other(
                "HorizontalDeltaCodec takes no configuration".into(),
            ));
        }
        Ok(Codec::ArrayToArray(Arc::new(HorizontalDeltaCodec)))
    }
}

impl CodecTraits for HorizontalDeltaCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for HorizontalDeltaCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

/// Element width in bytes, for the fixed-size types TIFF can hold.
fn width(data_type: &DataType) -> Result<usize, CodecError> {
    match data_type.fixed_size() {
        Some(n @ (1 | 2 | 4 | 8)) => Ok(n),
        _ => Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            "HorizontalDeltaCodec".into(),
        )),
    }
}

macro_rules! rows {
    ($bytes:expr, $row:expr, $ty:ty, $decode:expr) => {{
        const N: usize = std::mem::size_of::<$ty>();
        for row in $bytes.chunks_exact_mut($row * N) {
            let mut prev: $ty = 0;
            for e in row.chunks_exact_mut(N) {
                let v = <$ty>::from_ne_bytes(e.try_into().unwrap());
                let out = if $decode {
                    v.wrapping_add(prev)
                } else {
                    v.wrapping_sub(prev)
                };
                prev = if $decode { out } else { v };
                e.copy_from_slice(&out.to_ne_bytes());
            }
        }
    }};
}

/// Undo (`decode`) or apply horizontal differencing in place, rows of `row`
/// elements of `width` bytes.
pub fn horizontal(bytes: &mut [u8], row: usize, width: usize, decode: bool) {
    match width {
        1 => rows!(bytes, row, u8, decode),
        2 => rows!(bytes, row, u16, decode),
        4 => rows!(bytes, row, u32, decode),
        8 => rows!(bytes, row, u64, decode),
        _ => unreachable!("width checked by the codec"),
    }
}

impl HorizontalDeltaCodec {
    fn apply<'a>(
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        decode: bool,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let w = width(data_type)?;
        let row = shape.last().map_or(1, |n| n.get() as usize);
        let mut b = bytes.into_fixed()?.into_owned();
        if b.len() % (row * w) != 0 {
            return Err(CodecError::Other(format!(
                "HorizontalDeltaCodec: {} bytes is not a whole number of {row}-element rows",
                b.len()
            )));
        }
        horizontal(&mut b, row, w, decode);
        Ok(b.into())
    }
}

impl ArrayToArrayCodecTraits for HorizontalDeltaCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        width(decoded_data_type)?;
        Ok(decoded_data_type.clone())
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Self::apply(bytes, shape, data_type, false)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Self::apply(bytes, shape, data_type, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_restart_and_wrap() {
        // two rows of i16: [1, 2, -32768, 32767] and [5, 5, 5, 5]
        let vals: [i16; 8] = [1, 2, -32768, 32767, 5, 5, 5, 5];
        let mut b: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let orig = b.clone();
        horizontal(&mut b, 4, 2, false);
        let diffs: Vec<i16> = b
            .chunks_exact(2)
            .map(|c| i16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(diffs, vec![1, 1, 32766, -1, 5, 0, 0, 0]);
        horizontal(&mut b, 4, 2, true);
        assert_eq!(b, orig);
    }

    #[test]
    fn zarrs_opens_the_codec_by_name() {
        use zarrs::array::ArrayMetadata;
        let meta: ArrayMetadata = serde_json::from_value(serde_json::json!({
            "zarr_format": 3, "node_type": "array", "shape": [2, 4],
            "data_type": "int16",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 4]}},
            "chunk_key_encoding": {"name": "default"},
            "fill_value": 0,
            "codecs": [{"name": "HorizontalDeltaCodec"},
                       {"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "zstd", "configuration": {"level": 0, "checksum": false}}]
        }))
        .unwrap();
        let store = std::sync::Arc::new(zarrs::storage::store::MemoryStore::new());
        let array = zarrs::array::Array::new_with_metadata(store, "/a", meta).unwrap();
        let data: Vec<i16> = vec![1, 2, -32768, 32767, 5, 5, 5, 5];
        array.store_chunk(&[0, 0], &data).unwrap();
        let back: Vec<i16> = array.retrieve_chunk(&[0, 0]).unwrap();
        assert_eq!(back, data);
    }
}
