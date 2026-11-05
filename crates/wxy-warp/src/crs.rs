//! The CRS backend: which projection library turns one CRS's coordinates
//! into another's.
//!
//! rwarp asks one thing of a projection library, a fixed-direction point
//! transform ([`rwarp::CrsTransform`]), in degrees for geographic CRSs.
//! This module builds those transforms, and answers the two other CRS
//! questions the warp asks (does a definition parse; is it geographic in
//! degrees, for GDAL's `CENTER_LONG` wrap), with one of two pure-Rust
//! engines:
//!
//! | build                      | engine                                                  | takes                                                         |
//! | -------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
//! | default                    | [`proj4rs`] with `crs-definitions`, as rwarp uses it     | `EPSG:NNNN`, PROJ strings                                     |
//! | `--features terrio-crs`    | [`terrio_crs`](https://crates.io/crates/terrio-crs)     | also WKT1, WKT2, PROJJSON, `ESRI:`, `OGC:CRS84`, datum steps  |
//!
//! The choice is made when wxy-warp is built, not at run time (no global
//! state, design doc 05 A8). [`CrsBackend`] names both, and
//! [`crs_pair_with`] builds either one when it is compiled in, so tests and
//! comparisons can hold the two side by side in one build.
//!
//! This is a trial: proj4rs stays the default (small, and what rwarp and
//! the wasm builds use); terrio-crs is the full candidate. See
//! `docs/watch/crs-coverage.md` in wxylib/design for the case table.

use crate::{Error, Result};

/// A projection engine wxy-warp can be built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrsBackend {
    /// proj4rs with its EPSG table (`crs-definitions`).
    Proj4rs,
    /// terrio-crs (feature `terrio-crs`).
    TerrioCrs,
}

impl CrsBackend {
    /// The engine this build uses for warp, points and zonal: terrio-crs
    /// when the `terrio-crs` feature is on, else proj4rs.
    pub const fn active() -> Self {
        if cfg!(feature = "terrio-crs") {
            CrsBackend::TerrioCrs
        } else {
            CrsBackend::Proj4rs
        }
    }

    /// The engines compiled into this build.
    pub fn available() -> &'static [CrsBackend] {
        if cfg!(feature = "terrio-crs") {
            &[CrsBackend::Proj4rs, CrsBackend::TerrioCrs]
        } else {
            &[CrsBackend::Proj4rs]
        }
    }

    /// A short name: `proj4rs` or `terrio-crs`.
    pub const fn name(self) -> &'static str {
        match self {
            CrsBackend::Proj4rs => "proj4rs",
            CrsBackend::TerrioCrs => "terrio-crs",
        }
    }

    /// The definitions this engine takes, for error messages.
    const fn takes(self) -> &'static str {
        match self {
            CrsBackend::Proj4rs => "EPSG:NNNN or a PROJ string",
            CrsBackend::TerrioCrs => {
                "EPSG:NNNN, ESRI:NNNN, OGC:CRS84, WKT1, WKT2, PROJJSON or a PROJ string"
            }
        }
    }
}

/// The `(dst -> src, src -> dst)` pair of transforms with this build's
/// engine (see [`CrsBackend::active`]).
pub fn crs_pair(
    src: &str,
    dst: &str,
) -> Result<(Box<dyn rwarp::CrsTransform>, Box<dyn rwarp::CrsTransform>)> {
    crs_pair_with(CrsBackend::active(), src, dst)
}

/// [`crs_pair`] with a named engine, which must be compiled in.
pub fn crs_pair_with(
    backend: CrsBackend,
    src: &str,
    dst: &str,
) -> Result<(Box<dyn rwarp::CrsTransform>, Box<dyn rwarp::CrsTransform>)> {
    match backend {
        CrsBackend::Proj4rs => {
            // Parse each first, for wxy's own message naming what it takes.
            check_crs_with(backend, src)?;
            check_crs_with(backend, dst)?;
            rwarp::crs::crs_pair(src, dst).map_err(Error::Unsupported)
        }
        CrsBackend::TerrioCrs => terrio::pair(src, dst),
    }
}

/// Parse `def` as this build's engine does.
pub fn check_crs(def: &str) -> Result<()> {
    check_crs_with(CrsBackend::active(), def)
}

/// [`check_crs`] with a named engine.
pub fn check_crs_with(backend: CrsBackend, def: &str) -> Result<()> {
    match backend {
        CrsBackend::Proj4rs => proj4rs::proj::Proj::from_user_string(def)
            .map(|_| ())
            .map_err(|e| unsupported(backend, def, e)),
        CrsBackend::TerrioCrs => terrio::parse(def).map(|_| ()),
    }
}

/// Whether `def` is a geographic CRS in degrees, the test GDAL makes
/// before wrapping a source's longitudes around its centre.
pub fn is_geographic_degrees(def: &str) -> Result<bool> {
    is_geographic_degrees_with(CrsBackend::active(), def)
}

/// [`is_geographic_degrees`] with a named engine.
pub fn is_geographic_degrees_with(backend: CrsBackend, def: &str) -> Result<bool> {
    match backend {
        CrsBackend::Proj4rs => proj4rs::proj::Proj::from_user_string(def)
            .map(|p| p.is_latlong())
            .map_err(|e| unsupported(backend, def, e)),
        CrsBackend::TerrioCrs => terrio::parse(def).map(|s| s.is_geographic()),
    }
}

/// WGS 84 longitude and latitude, in degrees, to `def`'s coordinates
/// (`None` outside its domain): where the poles land, for the warp's
/// special points.
pub(crate) fn from_wgs84(def: &str, lon: f64, lat: f64) -> Option<(f64, f64)> {
    let (_, to) = crs_pair("EPSG:4326", def).ok()?;
    to.convert(lon, lat)
}

fn unsupported(backend: CrsBackend, def: &str, e: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!(
        "CRS {def:?}: {e} ({}, {})",
        backend.takes(),
        backend.name()
    ))
}

#[cfg(feature = "terrio-crs")]
mod terrio {
    use super::{unsupported, CrsBackend};
    use crate::Result;
    use terrio_crs::{SpatialRef, Transformer};

    pub(super) fn parse(def: &str) -> Result<SpatialRef> {
        // `parse_literal`: a definition is never read as a file name.
        let s = SpatialRef::parse_literal(def)
            .map_err(|e| unsupported(CrsBackend::TerrioCrs, def, e))?;
        // terrio-crs keeps a WKT it cannot use (`GEOGCRS["x"]`, no datum)
        // and refuses it only when a transform is built: build one to
        // WGS 84 here, so a bad definition is refused before any reading.
        let wgs84 =
            SpatialRef::from_epsg(4326).map_err(|e| unsupported(CrsBackend::TerrioCrs, def, e))?;
        Transformer::new(&s, &wgs84).map_err(|e| unsupported(CrsBackend::TerrioCrs, def, e))?;
        Ok(s)
    }

    /// A terrio-crs transform; `pole` when the source is geographic.
    struct Backend {
        t: Transformer,
        pole: bool,
    }

    impl rwarp::CrsTransform for Backend {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            let got = self.t.transform(x, y).ok().or_else(|| {
                // terrio-crs 0.2.2 refuses a point exactly at a pole
                // (EPSG:4326 (lon, -90) to EPSG:3031), which PROJ maps to
                // the projection's pole. Approach it from two opposite
                // longitudes, 1e-10 degrees short, and take the midpoint:
                // the first-order offsets cancel, so a pole on a cell edge
                // stays on it. Remove once fixed upstream;
                // tests/crs_backends.rs pole_is_a_point.
                if !(self.pole && y.abs() == 90.0) {
                    return None;
                }
                let lat = y.signum() * (90.0 - 1e-10);
                let a = self.t.transform(x, lat).ok()?;
                let b = self.t.transform(x + 180.0, lat).ok()?;
                Some(((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0))
            });
            got.filter(|(x, y)| x.is_finite() && y.is_finite())
        }
    }

    pub(super) fn pair(
        src: &str,
        dst: &str,
    ) -> Result<(Box<dyn rwarp::CrsTransform>, Box<dyn rwarp::CrsTransform>)> {
        let (s, d) = (parse(src)?, parse(dst)?);
        let new = |a: &SpatialRef, b: &SpatialRef| {
            Transformer::new(a, b).map_err(|e| {
                crate::Error::Unsupported(format!(
                    "terrio-crs: no transform {src:?} to {dst:?}: {e}"
                ))
            })
        };
        Ok((
            Box::new(Backend {
                t: new(&d, &s)?,
                pole: d.is_geographic(),
            }),
            Box::new(Backend {
                t: new(&s, &d)?,
                pole: s.is_geographic(),
            }),
        ))
    }
}

#[cfg(not(feature = "terrio-crs"))]
mod terrio {
    use crate::{Error, Result};

    const OFF: &str = "wxy-warp was built without the terrio-crs feature";

    /// Never built: [`parse`] always refuses.
    pub(super) struct SpatialRef;

    impl SpatialRef {
        pub(super) fn is_geographic(&self) -> bool {
            false
        }
    }

    pub(super) fn parse(_def: &str) -> Result<SpatialRef> {
        Err(Error::Unsupported(OFF.into()))
    }

    pub(super) fn pair(
        _src: &str,
        _dst: &str,
    ) -> Result<(Box<dyn rwarp::CrsTransform>, Box<dyn rwarp::CrsTransform>)> {
        Err(Error::Unsupported(OFF.into()))
    }
}
