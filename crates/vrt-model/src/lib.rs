//! A typed model of virtual dataset documents and a planner over it, with no
//! I/O and no pixel arithmetic (design m25).
//!
//! - [`dsn`]: source names, the grammar the VRT schemas leave as a string
//!   (paths, URLs, `/vsi` chains, driver prefixes, `vrt://`, inline VRT).
//! - [`xml`]: an element tree that reads and writes any VRT family.
//! - [`classic`]: classic raster VRT as typed values; unknown elements kept.
//! - [`vrtcon`]: `vrt://` strings lowered to a classic VRT.
//! - [`plan`]: which source windows a read needs, and where they land.
//!
//! Readers do the I/O: they open the sources the plan names, sample them as
//! [`plan::nearest_indices`] says, and apply the plan's operations.
//! Multidimensional and vector VRT are next (m25 section 7).

pub mod classic;
pub mod dsn;
pub mod plan;
pub mod vrtcon;
pub mod xml;

pub use classic::{Band, Dataset, Rect, Source};
pub use dsn::Dsn;
pub use plan::{
    plan_band, plan_band_leaves, LeafRow, Op, PlanRow, Request, Resolver, SourceWindow,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("XML: {0}")]
    Xml(String),
    #[error("invalid VRT: {0}")]
    Invalid(String),
    #[error("not supported yet: {0}")]
    Unsupported(String),
}
