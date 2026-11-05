//! Classic raster VRT (`<VRTDataset>` with `<VRTRasterBand>`s).
//!
//! The interpreted parts are those a planner needs: the grid, the bands, and
//! for each source its name, band, windows and per-source operations. Every
//! other element and attribute is kept in `extra` and written back, so a
//! document survives a read and write; [`Dataset::unknown`] lists what was
//! not interpreted, so a reader can say what it is ignoring.

use crate::dsn::Dsn;
use crate::xml::{self, Element, Node};
use crate::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub x_size: usize,
    pub y_size: usize,
    /// `subClass` (for example `VRTWarpedDataset`); `None` for a plain VRT.
    pub sub_class: Option<String>,
    /// The `<SRS>` element as written (text and attributes).
    pub srs: Option<Element>,
    pub geotransform: Option<[f64; 6]>,
    pub bands: Vec<Band>,
    pub extra_attrs: Vec<(String, String)>,
    pub extra: Vec<Element>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    pub data_type: String,
    pub band: Option<usize>,
    pub sub_class: Option<String>,
    /// `<NoDataValue>` text, as written.
    pub nodata: Option<String>,
    pub sources: Vec<Source>,
    pub extra_attrs: Vec<(String, String)>,
    pub extra: Vec<Element>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x_off: f64,
    pub y_off: f64,
    pub x_size: f64,
    pub y_size: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    /// Element name: `SimpleSource`, `ComplexSource`, `AveragedSource`,
    /// `KernelFilteredSource`, `NoDataFromMaskSource`.
    pub kind: String,
    pub filename: Dsn,
    /// `relativeToVRT` (or its old spelling) is 1.
    pub relative: bool,
    /// Attributes of `<SourceFilename>` other than `relativeToVRT`.
    pub filename_attrs: Vec<(String, String)>,
    /// `<SourceBand>` text: a number, or `mask,N`.
    pub source_band: String,
    pub properties: Option<Element>,
    pub src_rect: Option<Rect>,
    pub dst_rect: Option<Rect>,
    /// ComplexSource `<NODATA>` text.
    pub nodata: Option<String>,
    pub scale_offset: Option<f64>,
    pub scale_ratio: Option<f64>,
    /// `<LUT>` pairs (input, output).
    pub lut: Option<Vec<(f64, f64)>>,
    /// The `resampling` attribute.
    pub resampling: Option<String>,
    pub extra_attrs: Vec<(String, String)>,
    pub extra: Vec<Element>,
}

const SOURCE_KINDS: &[&str] = &[
    "SimpleSource",
    "ComplexSource",
    "AveragedSource",
    "KernelFilteredSource",
    "NoDataFromMaskSource",
];

fn num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, Error> {
    s.trim()
        .parse::<T>()
        .map_err(|_| Error::Invalid(format!("{what}: {s:?}")))
}

fn rect(e: &Element) -> Result<Rect, Error> {
    let g = |k: &str| -> Result<f64, Error> {
        num(
            e.attr(k)
                .ok_or_else(|| Error::Invalid(format!("{} without {k}", e.name)))?,
            k,
        )
    };
    Ok(Rect {
        x_off: g("xOff")?,
        y_off: g("yOff")?,
        x_size: g("xSize")?,
        y_size: g("ySize")?,
    })
}

fn rect_el(name: &str, r: &Rect) -> Element {
    let mut e = Element::new(name);
    e.set_attr("xOff", fmt_num(r.x_off));
    e.set_attr("yOff", fmt_num(r.y_off));
    e.set_attr("xSize", fmt_num(r.x_size));
    e.set_attr("ySize", fmt_num(r.y_size));
    e
}

/// Shortest decimal that reads back to the same f64.
pub(crate) fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn split_attrs(e: &Element, known: &[&str]) -> Vec<(String, String)> {
    e.attrs
        .iter()
        .filter(|(k, _)| !known.contains(&k.as_str()))
        .cloned()
        .collect()
}

impl Dataset {
    /// Parse a classic raster VRT document.
    pub fn from_xml(text: &str) -> Result<Dataset, Error> {
        Dataset::from_element(&xml::parse(text)?)
    }

    pub fn from_element(root: &Element) -> Result<Dataset, Error> {
        if root.name != "VRTDataset" {
            return Err(Error::Invalid(format!(
                "root is <{}>, not <VRTDataset>",
                root.name
            )));
        }
        if root.elements().any(|e| e.name == "Group") {
            return Err(Error::Unsupported(
                "multidimensional VRT (a <Group>) is not a classic VRT".into(),
            ));
        }
        let x_size = num(root.attr("rasterXSize").unwrap_or("0"), "rasterXSize")?;
        let y_size = num(root.attr("rasterYSize").unwrap_or("0"), "rasterYSize")?;
        let mut ds = Dataset {
            x_size,
            y_size,
            sub_class: root.attr("subClass").map(str::to_string),
            srs: None,
            geotransform: None,
            bands: vec![],
            extra_attrs: split_attrs(root, &["rasterXSize", "rasterYSize", "subClass"]),
            extra: vec![],
        };
        for e in root.elements() {
            match e.name.as_str() {
                "SRS" => ds.srs = Some(e.clone()),
                "GeoTransform" => {
                    let v: Vec<f64> = e
                        .text()
                        .split(',')
                        .map(|s| num(s, "GeoTransform"))
                        .collect::<Result<_, _>>()?;
                    if v.len() != 6 {
                        return Err(Error::Invalid("GeoTransform needs 6 values".into()));
                    }
                    ds.geotransform = Some([v[0], v[1], v[2], v[3], v[4], v[5]]);
                }
                "VRTRasterBand" => ds.bands.push(Band::from_element(e)?),
                _ => ds.extra.push(e.clone()),
            }
        }
        Ok(ds)
    }

    pub fn to_element(&self) -> Element {
        let mut r = Element::new("VRTDataset");
        r.set_attr("rasterXSize", self.x_size.to_string());
        r.set_attr("rasterYSize", self.y_size.to_string());
        if let Some(s) = &self.sub_class {
            r.set_attr("subClass", s.clone());
        }
        r.attrs.extend(self.extra_attrs.iter().cloned());
        if let Some(s) = &self.srs {
            r.push(s.clone());
        }
        if let Some(g) = &self.geotransform {
            let t: Vec<String> = g.iter().map(|v| format!("{v:e}")).collect();
            r.push(Element::with_text("GeoTransform", t.join(", ")));
        }
        for e in &self.extra {
            r.push(e.clone());
        }
        for b in &self.bands {
            r.push(b.to_element());
        }
        r
    }

    pub fn to_xml(&self) -> String {
        xml::write(&self.to_element())
    }

    /// Element names this model did not interpret, as `parent/child` paths.
    pub fn unknown(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .extra
            .iter()
            .map(|e| format!("VRTDataset/{}", e.name))
            .collect();
        for b in &self.bands {
            out.extend(b.extra.iter().map(|e| format!("VRTRasterBand/{}", e.name)));
            for s in &b.sources {
                out.extend(s.extra.iter().map(|e| format!("{}/{}", s.kind, e.name)));
            }
        }
        out
    }
}

impl Band {
    fn from_element(e: &Element) -> Result<Band, Error> {
        let mut b = Band {
            data_type: e.attr("dataType").unwrap_or("Byte").to_string(),
            band: e.attr("band").map(|s| num(s, "band")).transpose()?,
            sub_class: e.attr("subClass").map(str::to_string),
            nodata: None,
            sources: vec![],
            extra_attrs: split_attrs(e, &["dataType", "band", "subClass"]),
            extra: vec![],
        };
        for c in e.elements() {
            if c.name == "NoDataValue" {
                b.nodata = Some(c.text());
            } else if SOURCE_KINDS.contains(&c.name.as_str()) {
                b.sources.push(Source::from_element(c)?);
            } else {
                b.extra.push(c.clone());
            }
        }
        Ok(b)
    }

    fn to_element(&self) -> Element {
        let mut e = Element::new("VRTRasterBand");
        e.set_attr("dataType", self.data_type.clone());
        if let Some(n) = self.band {
            e.set_attr("band", n.to_string());
        }
        if let Some(s) = &self.sub_class {
            e.set_attr("subClass", s.clone());
        }
        e.attrs.extend(self.extra_attrs.iter().cloned());
        if let Some(n) = &self.nodata {
            e.push(Element::with_text("NoDataValue", n.clone()));
        }
        for x in &self.extra {
            e.push(x.clone());
        }
        for s in &self.sources {
            e.push(s.to_element());
        }
        e
    }

    /// `<NoDataValue>` as a number (`nan` and `inf` included).
    pub fn nodata_value(&self) -> Option<f64> {
        self.nodata.as_deref().and_then(parse_float)
    }
}

pub(crate) fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    match t.to_ascii_lowercase().as_str() {
        "nan" => Some(f64::NAN),
        "inf" | "+inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        _ => t.parse().ok(),
    }
}

impl Source {
    fn from_element(e: &Element) -> Result<Source, Error> {
        let fe = e
            .child("SourceFilename")
            .ok_or_else(|| Error::Invalid(format!("<{}> without <SourceFilename>", e.name)))?;
        let rel = fe
            .attr_any(&["relativeToVRT", "relativetoVRT"])
            .unwrap_or("0");
        let mut s = Source {
            kind: e.name.clone(),
            filename: Dsn::parse(&fe.text()),
            relative: matches!(
                rel.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            filename_attrs: split_attrs(fe, &["relativeToVRT", "relativetoVRT"]),
            source_band: "1".into(),
            properties: None,
            src_rect: None,
            dst_rect: None,
            nodata: None,
            scale_offset: None,
            scale_ratio: None,
            lut: None,
            resampling: e.attr("resampling").map(str::to_string),
            extra_attrs: split_attrs(e, &["resampling"]),
            extra: vec![],
        };
        for c in e.elements() {
            match c.name.as_str() {
                "SourceFilename" => {}
                "SourceBand" => s.source_band = c.text(),
                "SourceProperties" => s.properties = Some(c.clone()),
                "SrcRect" => s.src_rect = Some(rect(c)?),
                "DstRect" => s.dst_rect = Some(rect(c)?),
                "NODATA" => s.nodata = Some(c.text()),
                "ScaleOffset" => s.scale_offset = Some(num(&c.text(), "ScaleOffset")?),
                "ScaleRatio" => s.scale_ratio = Some(num(&c.text(), "ScaleRatio")?),
                "LUT" => {
                    let pairs = c
                        .text()
                        .split(',')
                        .map(|p| {
                            let (a, b) = p
                                .split_once(':')
                                .ok_or_else(|| Error::Invalid(format!("LUT entry {p:?}")))?;
                            Ok((num(a, "LUT")?, num(b, "LUT")?))
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    s.lut = Some(pairs);
                }
                _ => s.extra.push(c.clone()),
            }
        }
        Ok(s)
    }

    fn to_element(&self) -> Element {
        let mut e = Element::new(&self.kind);
        if let Some(r) = &self.resampling {
            e.set_attr("resampling", r.clone());
        }
        e.attrs.extend(self.extra_attrs.iter().cloned());
        let mut f = Element::with_text("SourceFilename", self.filename.to_string());
        f.set_attr("relativeToVRT", if self.relative { "1" } else { "0" });
        f.attrs.extend(self.filename_attrs.iter().cloned());
        e.push(f);
        e.push(Element::with_text("SourceBand", self.source_band.clone()));
        if let Some(p) = &self.properties {
            e.push(p.clone());
        }
        if let Some(r) = &self.src_rect {
            e.push(rect_el("SrcRect", r));
        }
        if let Some(r) = &self.dst_rect {
            e.push(rect_el("DstRect", r));
        }
        if let Some(n) = &self.nodata {
            e.push(Element::with_text("NODATA", n.clone()));
        }
        if let Some(v) = self.scale_offset {
            e.push(Element::with_text("ScaleOffset", format!("{v}")));
        }
        if let Some(v) = self.scale_ratio {
            e.push(Element::with_text("ScaleRatio", format!("{v}")));
        }
        if let Some(l) = &self.lut {
            let t: Vec<String> = l.iter().map(|(a, b)| format!("{a}:{b}")).collect();
            e.push(Element::with_text("LUT", t.join(",")));
        }
        for x in &self.extra {
            e.children.push(Node::Element(x.clone()));
        }
        e
    }

    /// The source band number, and whether it is the band's mask (`mask,N`).
    pub fn band_number(&self) -> Result<(usize, bool), Error> {
        let t = self.source_band.trim();
        match t.strip_prefix("mask,") {
            Some(n) => Ok((num(n, "SourceBand")?, true)),
            None => Ok((num(t, "SourceBand")?, false)),
        }
    }

    /// `<NODATA>` as a number.
    pub fn nodata_value(&self) -> Option<f64> {
        self.nodata.as_deref().and_then(parse_float)
    }

    /// Source size from `<SourceProperties>`, when written.
    pub fn declared_size(&self) -> Option<(usize, usize)> {
        let p = self.properties.as_ref()?;
        Some((
            p.attr("RasterXSize")?.parse().ok()?,
            p.attr("RasterYSize")?.parse().ok()?,
        ))
    }
}
