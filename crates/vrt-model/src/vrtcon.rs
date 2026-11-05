//! `vrt://` strings for classic rasters, lowered to a [`Dataset`].
//!
//! In the reference library a `vrt://` string is a translate call written as
//! a URI: each key is a translate flag, and the result is the VRT that call
//! would write. This module does the same lowering, given a description of
//! the source (its size, bands and grid) that the caller reads from the
//! source's header. Keys not yet lowered are an error that names them.

use crate::classic::{Band, Dataset, Rect, Source};
use crate::dsn::Dsn;
use crate::xml::Element;
use crate::Error;

/// What the lowering needs to know about the source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInfo {
    pub x_size: usize,
    pub y_size: usize,
    /// Data type name of each band (`Byte`, `Int16`, `Float32`, ...).
    pub band_types: Vec<String>,
    pub geotransform: Option<[f64; 6]>,
    /// The source's SRS, as text the VRT `<SRS>` element can carry.
    pub srs: Option<String>,
    /// Each band's nodata, as text.
    pub nodata: Vec<Option<String>>,
}

/// Keys lowered so far, in the reference library's terms.
pub const LOWERED: &[&str] = &[
    "bands", "srcwin", "outsize", "a_nodata", "a_srs", "a_ullr", "a_gt",
];

fn nums(v: &str, n: usize, key: &str) -> Result<Vec<f64>, Error> {
    let out: Vec<f64> = v
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| Error::Invalid(format!("{key}={v}")))
        })
        .collect::<Result<_, _>>()?;
    if out.len() != n {
        return Err(Error::Invalid(format!("{key} needs {n} values: {v}")));
    }
    Ok(out)
}

/// Lower `vrt://` options over a source into the dataset a translate to VRT
/// would write. `source` is written into each `<SourceFilename>` as given.
pub fn lower(
    source: &Dsn,
    options: &[(String, String)],
    info: &SourceInfo,
) -> Result<Dataset, Error> {
    let unknown: Vec<&str> = options
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !LOWERED.contains(k))
        .collect();
    if !unknown.is_empty() {
        return Err(Error::Unsupported(format!(
            "vrt:// keys not lowered yet: {}",
            unknown.join(", ")
        )));
    }
    let get = |k: &str| {
        options
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.as_str())
    };

    let bands: Vec<usize> = match get("bands") {
        Some(v) => v
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| Error::Invalid(format!("bands={v}")))
            })
            .collect::<Result<_, _>>()?,
        None => (1..=info.band_types.len()).collect(),
    };
    for &b in &bands {
        if b == 0 || b > info.band_types.len() {
            return Err(Error::Invalid(format!(
                "band {b} of {}",
                info.band_types.len()
            )));
        }
    }
    let win = match get("srcwin") {
        Some(v) => {
            let w = nums(v, 4, "srcwin")?;
            Rect {
                x_off: w[0],
                y_off: w[1],
                x_size: w[2],
                y_size: w[3],
            }
        }
        None => Rect {
            x_off: 0.0,
            y_off: 0.0,
            x_size: info.x_size as f64,
            y_size: info.y_size as f64,
        },
    };
    let (ox, oy) = match get("outsize") {
        Some(v) => {
            let parts: Vec<&str> = v.split(',').map(str::trim).collect();
            if parts.len() != 2 {
                return Err(Error::Invalid(format!("outsize={v}")));
            }
            let one = |p: &str, full: f64| -> Result<f64, Error> {
                match p.strip_suffix('%') {
                    Some(pct) => {
                        let f: f64 = pct
                            .parse()
                            .map_err(|_| Error::Invalid(format!("outsize={v}")))?;
                        Ok((f / 100.0 * full).round())
                    }
                    None => p
                        .parse::<f64>()
                        .map_err(|_| Error::Invalid(format!("outsize={v}"))),
                }
            };
            (one(parts[0], win.x_size)?, one(parts[1], win.y_size)?)
        }
        None => (win.x_size, win.y_size),
    };

    let mut gt = info.geotransform;
    if let Some(g) = gt.as_mut() {
        let (x0, y0) = (
            g[0] + win.x_off * g[1] + win.y_off * g[2],
            g[3] + win.x_off * g[4] + win.y_off * g[5],
        );
        let (fx, fy) = (win.x_size / ox, win.y_size / oy);
        *g = [x0, g[1] * fx, g[2] * fy, y0, g[4] * fx, g[5] * fy];
    }
    if let Some(v) = get("a_ullr") {
        let u = nums(v, 4, "a_ullr")?;
        gt = Some([u[0], (u[2] - u[0]) / ox, 0.0, u[1], 0.0, (u[3] - u[1]) / oy]);
    }
    if let Some(v) = get("a_gt") {
        let u = nums(v, 6, "a_gt")?;
        gt = Some([u[0], u[1], u[2], u[3], u[4], u[5]]);
    }
    let srs = get("a_srs")
        .map(str::to_string)
        .or_else(|| info.srs.clone())
        .map(|s| Element::with_text("SRS", s));

    let mut out_bands = vec![];
    for (i, &b) in bands.iter().enumerate() {
        let nodata = get("a_nodata")
            .map(str::to_string)
            .or_else(|| info.nodata.get(b - 1).cloned().flatten());
        let mut props = Element::new("SourceProperties");
        props.set_attr("RasterXSize", info.x_size.to_string());
        props.set_attr("RasterYSize", info.y_size.to_string());
        props.set_attr("DataType", info.band_types[b - 1].clone());
        out_bands.push(Band {
            data_type: info.band_types[b - 1].clone(),
            band: Some(i + 1),
            sub_class: None,
            nodata,
            sources: vec![Source {
                kind: "SimpleSource".into(),
                filename: source.clone(),
                relative: false,
                filename_attrs: vec![],
                source_band: b.to_string(),
                properties: Some(props),
                src_rect: Some(win),
                dst_rect: Some(Rect {
                    x_off: 0.0,
                    y_off: 0.0,
                    x_size: ox,
                    y_size: oy,
                }),
                nodata: None,
                scale_offset: None,
                scale_ratio: None,
                lut: None,
                resampling: None,
                extra_attrs: vec![],
                extra: vec![],
            }],
            extra_attrs: vec![],
            extra: vec![],
        });
    }
    Ok(Dataset {
        x_size: ox as usize,
        y_size: oy as usize,
        sub_class: None,
        srs,
        geotransform: gt,
        bands: out_bands,
        extra_attrs: vec![],
        extra: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> SourceInfo {
        SourceInfo {
            x_size: 30,
            y_size: 20,
            band_types: vec!["Byte".into(), "Int16".into()],
            geotransform: Some([100.0, 1.0, 0.0, -10.0, 0.0, -1.0]),
            srs: Some("EPSG:4326".into()),
            nodata: vec![None, Some("-1".into())],
        }
    }

    #[test]
    fn srcwin_outsize_bands() {
        let d = Dsn::parse("vrt://a.tif?bands=2&srcwin=4,2,20,10&outsize=50%,50%");
        let Dsn::VrtString { source, options } = d else {
            panic!()
        };
        let ds = lower(&source, &options, &info()).unwrap();
        assert_eq!((ds.x_size, ds.y_size), (10, 5));
        assert_eq!(
            ds.geotransform.unwrap(),
            [104.0, 2.0, 0.0, -12.0, 0.0, -2.0]
        );
        assert_eq!(ds.bands.len(), 1);
        assert_eq!(ds.bands[0].data_type, "Int16");
        assert_eq!(ds.bands[0].nodata.as_deref(), Some("-1"));
        let back = Dataset::from_xml(&ds.to_xml()).unwrap();
        assert_eq!(back, ds);
    }

    #[test]
    fn unknown_keys_are_named() {
        let e = lower(
            &Dsn::parse("a.tif"),
            &[("tr".into(), "1,1".into())],
            &info(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("tr"));
    }
}
