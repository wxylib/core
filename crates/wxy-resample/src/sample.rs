//! Cell value types and the conversions a kernel ends with.

/// A cell value type: how to read it as `f64` and how to write a kernel's
/// `f64` result back, as GDAL's warp kernel does.
pub trait Sample: Copy + PartialEq + Default + 'static {
    /// The value as `f64`.
    fn to_f64(self) -> f64;
    /// Round without clamping (GDAL's `GWKRoundValueT`): integers to the
    /// nearest (signed: `floor(v + 0.5)`; unsigned: `v + 0.5` truncated),
    /// floats by a cast.
    fn round_from(v: f64) -> Self;
    /// Clamp to the type's range, then round (GDAL's `GWKClampValueT`).
    fn clamp_from(v: f64) -> Self;
    /// The nearest value that is not this one (GDAL's `AvoidNoData`):
    /// integers step down (up from the minimum), floats step towards the
    /// maximum (down from it).
    fn nudge(self) -> Self;
    /// `nodata` as this type: integers only when they hold it exactly,
    /// floats always (rounded), as GDAL's nodata masker compares.
    fn from_nodata(nodata: f64) -> Option<Self>;
    /// Whether this value is `nodata` (NaN matches NaN).
    fn is_nodata(self, nodata: Self) -> bool;
}

macro_rules! int_sample {
    ($t:ty, $signed:expr) => {
        impl Sample for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn round_from(v: f64) -> Self {
                if $signed {
                    (v + 0.5).floor() as $t
                } else {
                    (v + 0.5) as $t
                }
            }
            fn clamp_from(v: f64) -> Self {
                if v < <$t>::MIN as f64 {
                    <$t>::MIN
                } else if v > <$t>::MAX as f64 {
                    <$t>::MAX
                } else {
                    Self::round_from(v)
                }
            }
            fn nudge(self) -> Self {
                if self == <$t>::MIN {
                    <$t>::MIN + 1
                } else {
                    self - 1
                }
            }
            fn from_nodata(nodata: f64) -> Option<Self> {
                let v = nodata as $t;
                (v as f64 == nodata).then_some(v)
            }
            fn is_nodata(self, nodata: Self) -> bool {
                self == nodata
            }
        }
    };
}

int_sample!(u8, false);
int_sample!(i8, true);
int_sample!(u16, false);
int_sample!(i16, true);
int_sample!(u32, false);
int_sample!(i32, true);
int_sample!(u64, false);
int_sample!(i64, true);

macro_rules! float_sample {
    ($t:ty) => {
        impl Sample for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn round_from(v: f64) -> Self {
                v as $t
            }
            fn clamp_from(v: f64) -> Self {
                v as $t
            }
            fn nudge(self) -> Self {
                if self == <$t>::MAX || self == <$t>::INFINITY {
                    self.next_down()
                } else {
                    self.next_up()
                }
            }
            fn from_nodata(nodata: f64) -> Option<Self> {
                Some(nodata as $t)
            }
            fn is_nodata(self, nodata: Self) -> bool {
                self == nodata || (self.is_nan() && nodata.is_nan())
            }
        }
    };
}

float_sample!(f32);
float_sample!(f64);
