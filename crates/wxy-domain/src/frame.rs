//! Frames: what the numbers a mapping produces mean.
//!
//! This crate does not interpret frames. A CRS definition travels as an
//! opaque string (EPSG code, PROJ string, WKT or PROJJSON) and a CRS backend
//! elsewhere decides what it means; see mdsumner/rust-proj-capability for
//! the backend question.

/// The frame a mapping's output coordinates are in.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Frame {
    /// Index space only: no frame was given.
    Index,
    /// A coordinate reference system, as an opaque definition.
    Crs(String),
    /// Anything else with a meaning a reader could name (for example
    /// `"scan_angle"` before the satellite height is known, or a model's
    /// own grid coordinates).
    Other(String),
}

impl Frame {
    pub fn crs(def: impl Into<String>) -> Self {
        Frame::Crs(def.into())
    }

    pub fn is_index(&self) -> bool {
        matches!(self, Frame::Index)
    }
}
