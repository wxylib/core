//! Positions: how a position follows from an index. One variant per rung.

use std::fmt;
use std::sync::Arc;

use crate::affine::Affine;

/// The rung of the position ladder (design doc 02, section 3). Each rung
/// stores strictly more than the one before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rung {
    /// R0: index only.
    IndexOnly,
    /// R1: arithmetic from origin and step (affine).
    Regular,
    /// R2: outer product of 1D coordinate vectors.
    Rectilinear,
    /// R3: one position per element, still (i, j) indexed.
    Curvilinear,
    /// R3': a fitted function of the index (GCP polynomial, RPC, TPS).
    Fitted,
    /// R4: implicit mesh; arithmetic from a cell id (DGGS, L3 bins).
    ImplicitMesh,
    /// R5: explicit mesh; stored positions plus connectivity.
    ExplicitMesh,
    /// R6: features; a geometry per element.
    Features,
}

/// A coordinate array: values in C order (last dim fastest) with a shape.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordArray {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

impl CoordArray {
    pub fn new(shape: Vec<usize>, values: Vec<f64>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "CoordArray shape does not match its values"
        );
        Self { shape, values }
    }
}

/// A fitted model from index to frame (R3'): GCP polynomial, RPC, TPS.
pub trait FittedModel: fmt::Debug + Send + Sync {
    fn forward(&self, index: &[f64]) -> Option<Vec<f64>>;
    fn inverse(&self, _position: &[f64]) -> Option<Vec<f64>> {
        None
    }
}

/// A cell scheme (R4): positions computed from a cell id.
pub trait CellScheme: fmt::Debug + Send + Sync {
    /// A short name, such as `"healpix"`, `"h3"` or `"l3bin"`.
    fn name(&self) -> &str;
    /// The centre of a cell, in the scheme's frame.
    fn centre(&self, cell: u64) -> Option<Vec<f64>>;
}

/// Stored elements (R5 vertices, R6 geometries). Kept opaque in S1.
pub trait Elements: fmt::Debug + Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How positions follow from an index, for the dims a mapping covers.
#[derive(Debug, Clone)]
pub enum Positions {
    /// R0. No positions.
    IndexOnly,
    /// R1, two dims `(row, col)` in that order: an affine transform.
    Affine(Affine),
    /// R1, one dim: `origin` is the position at index 0.0 (the outer edge
    /// of the first element), `step` the change per element.
    Regular1 { origin: f64, step: f64 },
    /// R2. One coordinate vector per covered dim.
    Rectilinear(Vec<Vec<f64>>),
    /// R3. One array per output coordinate, each shaped like the covered
    /// dims.
    Curvilinear(Vec<CoordArray>),
    /// R3'.
    Fitted(Arc<dyn FittedModel>),
    /// R4.
    Scheme(Arc<dyn CellScheme>),
    /// R5 and R6.
    Explicit(Arc<dyn Elements>),
}

impl Positions {
    pub fn rung(&self) -> Rung {
        match self {
            Positions::IndexOnly => Rung::IndexOnly,
            Positions::Affine(_) | Positions::Regular1 { .. } => Rung::Regular,
            Positions::Rectilinear(_) => Rung::Rectilinear,
            Positions::Curvilinear(_) => Rung::Curvilinear,
            Positions::Fitted(_) => Rung::Fitted,
            Positions::Scheme(_) => Rung::ImplicitMesh,
            Positions::Explicit(_) => Rung::ExplicitMesh,
        }
    }

    /// Number of values stored to describe the positions. The "six numbers
    /// against 209,824" comparison in regular-in-what.
    pub fn stored_values(&self) -> usize {
        match self {
            Positions::IndexOnly => 0,
            Positions::Affine(_) => 6,
            Positions::Regular1 { .. } => 2,
            Positions::Rectilinear(v) => v.iter().map(Vec::len).sum(),
            Positions::Curvilinear(a) => a.iter().map(|c| c.values.len()).sum(),
            Positions::Fitted(_) | Positions::Scheme(_) | Positions::Explicit(_) => 0,
        }
    }
}
