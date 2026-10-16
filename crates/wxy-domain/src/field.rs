//! Fields: values on a domain.

/// Element types. One shared vocabulary for every adapter (survey M3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Complex64,
    Complex128,
    String,
    Bytes,
}

impl DataType {
    /// Size in bytes, or `None` for variable-length types.
    pub fn size(&self) -> Option<usize> {
        use DataType::*;
        Some(match self {
            Bool | U8 | I8 => 1,
            U16 | I16 | F16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 | Complex64 => 8,
            Complex128 => 16,
            String | Bytes => return None,
        })
    }
}

/// How stored numbers become values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValueSemantics {
    pub nodata: Option<f64>,
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    pub units: Option<String>,
}

/// Which element of a domain a value lives on (design doc 02, DM5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// The element itself: a grid cell, a DGGS cell, a feature.
    Cell,
    /// Mesh nodes (vertices), or grid corners.
    Node,
    /// Mesh edges, or grid faces in a staggered layout.
    Edge,
    /// Mesh faces.
    Face,
    /// Layered mesh volumes.
    Volume,
}

/// Values on a domain.
///
/// Where the numbers come from is not part of this type: sources live in
/// `wxy-source`, so this crate stays free of I/O dependencies.
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub dtype: DataType,
    pub values: ValueSemantics,
    /// Index into the owning collection's domains.
    pub domain: usize,
    pub location: Location,
    /// Index of a dim that holds vector components (`u`, `v`), if any.
    /// A warp must rotate such fields, not resample them as scalars.
    pub components: Option<usize>,
}

impl Field {
    pub fn new(name: impl Into<String>, dtype: DataType, domain: usize) -> Self {
        Self {
            name: name.into(),
            dtype,
            values: ValueSemantics::default(),
            domain,
            location: Location::Cell,
            components: None,
        }
    }
}
