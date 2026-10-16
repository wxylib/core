//! # wxy-geozarr
//!
//! The GeoZarr [`spatial`](https://github.com/zarr-conventions/spatial) and
//! [`proj`](https://github.com/zarr-conventions/proj) conventions, v0.1, as
//! plain Rust over `serde_json` attribute maps. It depends on nothing else
//! in wxylib, so any Zarr reader or writer can use it (zarrs, icechunk, or
//! hand-written metadata).
//!
//! - [`Spatial`] and [`Proj`] parse from and write to attributes.
//! - [`inherit`] applies the conventions' group-to-array inheritance.
//! - [`declare`] adds the `zarr_conventions` registration entries.
//! - [`Spatial::geotransform`] and [`Spatial::from_geotransform`] convert to
//!   and from GDAL's coefficient order and pixel-corner registration.

use serde_json::{json, Map, Value};

pub type Attributes = Map<String, Value>;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    #[error("{0}: {1}")]
    Invalid(&'static str, String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A registered convention, as listed in `zarr_conventions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Convention {
    pub name: &'static str,
    pub uuid: &'static str,
    pub schema_url: &'static str,
    pub spec_url: &'static str,
    pub description: &'static str,
}

pub const SPATIAL: Convention = Convention {
    name: "spatial",
    uuid: "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    schema_url:
        "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
    spec_url: "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    description: "Spatial coordinate information",
};

pub const PROJ: Convention = Convention {
    name: "proj",
    uuid: "f17cb550-5864-4468-aeb7-f3180cfb622f",
    schema_url:
        "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    spec_url: "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    description: "Coordinate reference system information for geospatial data",
};

impl Convention {
    pub fn to_json(&self) -> Value {
        json!({
            "schema_url": self.schema_url,
            "spec_url": self.spec_url,
            "uuid": self.uuid,
            "name": self.name,
            "description": self.description,
        })
    }
}

/// Add `conventions` to `attrs["zarr_conventions"]`, skipping any already
/// listed (matched by uuid).
pub fn declare(attrs: &mut Attributes, conventions: &[Convention]) {
    let list = attrs
        .entry("zarr_conventions")
        .or_insert_with(|| Value::Array(Vec::new()));
    if !list.is_array() {
        *list = Value::Array(Vec::new());
    }
    let list = list.as_array_mut().expect("array");
    for c in conventions {
        let present = list
            .iter()
            .any(|v| v.get("uuid").and_then(Value::as_str) == Some(c.uuid));
        if !present {
            list.push(c.to_json());
        }
    }
}

/// Whether `attrs` declares `convention` (by uuid, or by name).
pub fn declares(attrs: &Attributes, convention: &Convention) -> bool {
    attrs
        .get("zarr_conventions")
        .and_then(Value::as_array)
        .is_some_and(|l| {
            l.iter().any(|v| {
                v.get("uuid").and_then(Value::as_str) == Some(convention.uuid)
                    || v.get("name").and_then(Value::as_str) == Some(convention.name)
            })
        })
}

/// A group's attributes overlaid by a child array's: the inheritance rule
/// for `spatial:` and `proj:` (array values take precedence).
pub fn inherit(group: &Attributes, array: &Attributes) -> Attributes {
    let mut out = group.clone();
    for (k, v) in array {
        out.insert(k.clone(), v.clone());
    }
    out
}

/// Grid cell registration (`spatial:registration`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Registration {
    /// The transform places cell corners (GeoTIFF PixelIsArea). Default.
    #[default]
    Pixel,
    /// The transform places cell centres (GeoTIFF PixelIsPoint).
    Node,
}

/// The `spatial:` properties.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Spatial {
    /// `[y_dim, x_dim]`, ordered by axis role.
    pub dimensions: Option<[String; 2]>,
    /// `"affine"` when absent.
    pub transform_type: Option<String>,
    /// `[a, b, c, d, e, f]`: `x = a*col + b*row + c`, `y = d*col + e*row + f`.
    pub transform: Option<[f64; 6]>,
    /// `[height, width]`.
    pub shape: Option<[u64; 2]>,
    /// `[xmin, ymin, xmax, ymax]`.
    pub bbox: Option<[f64; 4]>,
    pub registration: Registration,
}

fn numbers<const N: usize>(v: &Value, what: &'static str) -> Result<[f64; N]> {
    let a = v
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or_else(|| Error::Invalid(what, format!("needs {N} numbers")))?;
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x
            .as_f64()
            .ok_or_else(|| Error::Invalid(what, format!("{x} is not a number")))?;
    }
    Ok(out)
}

impl Spatial {
    /// Parse `spatial:` properties. `None` when there are none.
    pub fn from_attrs(attrs: &Attributes) -> Result<Option<Self>> {
        if !attrs.keys().any(|k| k.starts_with("spatial:")) {
            return Ok(None);
        }
        let mut s = Spatial::default();
        if let Some(v) = attrs.get("spatial:dimensions") {
            let d: Vec<String> = v
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let [y, x]: [String; 2] = d
                .try_into()
                .map_err(|_| Error::Invalid("spatial:dimensions", "needs two names".into()))?;
            s.dimensions = Some([y, x]);
        }
        if let Some(v) = attrs.get("spatial:transform_type") {
            s.transform_type = v.as_str().map(str::to_string);
        }
        if let Some(v) = attrs.get("spatial:transform") {
            s.transform = Some(numbers::<6>(v, "spatial:transform")?);
        }
        if let Some(v) = attrs.get("spatial:bbox") {
            s.bbox = Some(numbers::<4>(v, "spatial:bbox")?);
        }
        if let Some(v) = attrs.get("spatial:shape") {
            let [h, w] = numbers::<2>(v, "spatial:shape")?;
            s.shape = Some([h as u64, w as u64]);
        }
        s.registration = match attrs.get("spatial:registration").and_then(Value::as_str) {
            None | Some("pixel") => Registration::Pixel,
            Some("node") => Registration::Node,
            Some(o) => return Err(Error::Invalid("spatial:registration", o.into())),
        };
        Ok(Some(s))
    }

    /// Write the properties that are set into `attrs`, and declare the
    /// convention.
    pub fn to_attrs(&self, attrs: &mut Attributes) {
        declare(attrs, &[SPATIAL]);
        if let Some(d) = &self.dimensions {
            attrs.insert("spatial:dimensions".into(), json!(d));
        }
        if let Some(t) = &self.transform_type {
            attrs.insert("spatial:transform_type".into(), json!(t));
        }
        if let Some(t) = &self.transform {
            attrs.insert("spatial:transform".into(), json!(t));
        }
        if let Some(s) = &self.shape {
            attrs.insert("spatial:shape".into(), json!(s));
        }
        if let Some(b) = &self.bbox {
            attrs.insert("spatial:bbox".into(), json!(b));
        }
        if self.registration == Registration::Node {
            attrs.insert("spatial:registration".into(), json!("node"));
        }
    }

    /// Whether the transform is affine (the default type).
    pub fn is_affine(&self) -> bool {
        self.transform_type.as_deref().unwrap_or("affine") == "affine"
    }

    /// The transform as a GDAL geotransform `[c, a, b, f, d, e]` placing
    /// cell corners, so node-registered grids are shifted by half a cell.
    /// `None` without an affine transform.
    pub fn geotransform(&self) -> Option<[f64; 6]> {
        let t = self.transform.filter(|_| self.is_affine())?;
        let mut gt = [t[2], t[0], t[1], t[5], t[3], t[4]];
        if self.registration == Registration::Node {
            gt[0] -= 0.5 * (gt[1] + gt[2]);
            gt[3] -= 0.5 * (gt[4] + gt[5]);
        }
        Some(gt)
    }

    /// Pixel-registered properties from a GDAL geotransform.
    pub fn from_geotransform(gt: [f64; 6], dimensions: [&str; 2], shape: [u64; 2]) -> Self {
        Spatial {
            dimensions: Some(dimensions.map(str::to_string)),
            transform: Some([gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]]),
            shape: Some(shape),
            ..Default::default()
        }
    }
}

/// The `proj:` properties. At least one should be set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Proj {
    /// Authority and code, such as `EPSG:4326`.
    pub code: Option<String>,
    pub wkt2: Option<String>,
    pub projjson: Option<Value>,
}

impl Proj {
    pub fn from_attrs(attrs: &Attributes) -> Option<Self> {
        let p = Proj {
            code: attrs
                .get("proj:code")
                .and_then(Value::as_str)
                .map(str::to_string),
            wkt2: attrs
                .get("proj:wkt2")
                .and_then(Value::as_str)
                .map(str::to_string),
            projjson: attrs
                .get("proj:projjson")
                .filter(|v| v.is_object())
                .cloned(),
        };
        (p.code.is_some() || p.wkt2.is_some() || p.projjson.is_some()).then_some(p)
    }

    pub fn to_attrs(&self, attrs: &mut Attributes) {
        declare(attrs, &[PROJ]);
        if let Some(c) = &self.code {
            attrs.insert("proj:code".into(), json!(c));
        }
        if let Some(w) = &self.wkt2 {
            attrs.insert("proj:wkt2".into(), json!(w));
        }
        if let Some(p) = &self.projjson {
            attrs.insert("proj:projjson".into(), p.clone());
        }
    }

    /// The `proj:` property a CRS string belongs in: `AUTH:CODE` is a code,
    /// a JSON object is PROJJSON, and a bracketed keyword form is WKT2.
    /// Anything else (a PROJ string, say) has no `proj:` property.
    pub fn from_definition(def: &str) -> Option<Self> {
        let def = def.trim();
        if def.starts_with('{') {
            let v: Value = serde_json::from_str(def).ok()?;
            return v.is_object().then(|| Proj {
                projjson: Some(v),
                ..Default::default()
            });
        }
        if let Some((auth, code)) = def.split_once(':') {
            let word = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric());
            if word(auth) && word(code) {
                return Some(Proj {
                    code: Some(def.to_string()),
                    ..Default::default()
                });
            }
        }
        let keyword = def.split('[').next().unwrap_or("");
        if def.ends_with(']')
            && !keyword.is_empty()
            && keyword.chars().all(|c| c.is_ascii_uppercase() || c == '_')
        {
            return Some(Proj {
                wkt2: Some(def.to_string()),
                ..Default::default()
            });
        }
        None
    }

    /// One string for the CRS: the code, else WKT2, else PROJJSON text.
    pub fn definition(&self) -> Option<String> {
        self.code
            .clone()
            .or_else(|| self.wkt2.clone())
            .or_else(|| self.projjson.as_ref().map(Value::to_string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(v: Value) -> Attributes {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn spec_example_round_trips() {
        let a = attrs(json!({
            "proj:code": "EPSG:26711",
            "spatial:dimensions": ["Y", "X"],
            "spatial:transform": [60.0, 0.0, 440720.0, 0.0, -60.0, 3750120.0]
        }));
        let s = Spatial::from_attrs(&a).unwrap().unwrap();
        assert_eq!(
            s.geotransform().unwrap(),
            [440720.0, 60.0, 0.0, 3750120.0, 0.0, -60.0]
        );
        assert_eq!(
            Proj::from_attrs(&a).unwrap().definition().unwrap(),
            "EPSG:26711"
        );

        let mut out = Attributes::new();
        s.to_attrs(&mut out);
        Proj::from_attrs(&a).unwrap().to_attrs(&mut out);
        assert!(declares(&out, &SPATIAL) && declares(&out, &PROJ));
        assert_eq!(Spatial::from_attrs(&out).unwrap().unwrap(), s);
    }

    #[test]
    fn node_registration_moves_to_corner() {
        let a = attrs(json!({
            "spatial:transform": [10.0, 0.0, 5.0, 0.0, -10.0, 95.0],
            "spatial:registration": "node"
        }));
        let gt = Spatial::from_attrs(&a)
            .unwrap()
            .unwrap()
            .geotransform()
            .unwrap();
        assert_eq!(gt, [0.0, 10.0, 0.0, 100.0, 0.0, -10.0]);
    }

    #[test]
    fn array_overrides_group() {
        let g = attrs(json!({"proj:code": "EPSG:4326", "spatial:shape": [1, 2]}));
        let a = attrs(json!({"spatial:shape": [3, 4]}));
        let m = inherit(&g, &a);
        assert_eq!(m["proj:code"], "EPSG:4326");
        assert_eq!(
            Spatial::from_attrs(&m).unwrap().unwrap().shape,
            Some([3, 4])
        );
    }

    #[test]
    fn declare_is_idempotent() {
        let mut a = Attributes::new();
        declare(&mut a, &[SPATIAL, PROJ]);
        declare(&mut a, &[SPATIAL]);
        assert_eq!(a["zarr_conventions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn unknown_transform_type_gives_no_geotransform() {
        let a = attrs(json!({"spatial:transform_type": "rpc", "spatial:transform": [1,0,0,0,1,0]}));
        assert!(Spatial::from_attrs(&a)
            .unwrap()
            .unwrap()
            .geotransform()
            .is_none());
    }

    #[test]
    fn proj_from_definition() {
        let code = Proj::from_definition("EPSG:3031").unwrap();
        assert_eq!(code.code.as_deref(), Some("EPSG:3031"));
        let wkt = Proj::from_definition("GEOGCRS[\"WGS 84\"]").unwrap();
        assert!(wkt.wkt2.is_some());
        let pj = Proj::from_definition(r#"{"type": "GeographicCRS"}"#).unwrap();
        assert!(pj.projjson.is_some());
        assert!(Proj::from_definition("+proj=stere +lat_0=-90").is_none());
    }
}
