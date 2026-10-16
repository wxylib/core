//! # wxy-domain
//!
//! One data model for gridded, mesh and feature data, with no I/O and no
//! CRS dependency. This is stage S1 of the wxylib roadmap; the design is in
//! [wxylib/design](https://github.com/wxylib/design), documents 02 and 03.
//!
//! The model separates three things that gridded-data software usually
//! blurs (after *Regular in what?*, hypertidy/regular-in-what):
//!
//! - **index mapping**: how a position follows from an index
//!   ([`Positions`], one variant per rung of the [`Rung`] ladder);
//! - **frame**: what the positions mean ([`Frame`], opaque to this crate);
//! - **storage**: what a producer wrote down. Readers record the stored
//!   form as given, and [`normalise`] turns it into a cheaper, equivalent
//!   mapping when the evidence allows, with a [`normalise::Report`].
//!
//! A [`Domain`] is index space ([`Dim`]s) plus any number of [`Mapping`]s,
//! each covering some of the dims and naming its frame. A [`Field`] is
//! values on a domain. A [`Collection`] groups fields and domains, and
//! covers a classic raster dataset, a multidimensional group, a mesh with
//! its datasets, or a feature layer.
//!
//! ## Index space convention
//!
//! Continuous index coordinates follow GDAL's pixel convention: `0.0` is
//! the outer edge of the first element along a dim and `i + 0.5` is the
//! centre of element `i`. Every mapping evaluates in this space, whatever
//! its [`CellConvention`] says about where stored coordinates sit.
//!
//! ## What is implemented in S1
//!
//! Rungs R0 to R3 (index only, regular, rectilinear, curvilinear) are
//! implemented and evaluate positions. R3' (fitted models), R4 (implicit
//! meshes such as DGGS) and R5/R6 (explicit meshes, features) are present
//! as trait-object variants so the types will not change when they arrive.

pub mod affine;
pub mod collection;
pub mod dim;
pub mod domain;
pub mod field;
pub mod frame;
pub mod mapping;
pub mod normalise;
pub mod positions;

pub use affine::Affine;
pub use collection::{Collection, Relation, RelationKind};
pub use dim::Dim;
pub use domain::Domain;
pub use field::{DataType, Field, Location, ValueSemantics};
pub use frame::Frame;
pub use mapping::{CellConvention, Mapping};
pub use positions::{CoordArray, Positions, Rung};
