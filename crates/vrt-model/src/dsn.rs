//! Source names: what a `SourceFilename` or `SrcDataSource` can hold.
//!
//! The VRT schemas type a source name as a plain string. Its grammar lives in
//! the reference library's open code: plain paths and URLs, virtual file
//! system chains (`/vsicurl/`, `/vsizip/{...}`), driver prefixes with a
//! subdataset (`NETCDF:"file.nc":sst`), `vrt://` strings, nested VRT files
//! and inline VRT documents. [`Dsn::parse`] reads all of them into one typed
//! value and [`Dsn::to_string`] writes the same string back, so strings built
//! by tools such as the R package dsn round-trip (design m25 section 4.4).

use std::fmt;

/// A parsed source name.
#[derive(Debug, Clone, PartialEq)]
pub enum Dsn {
    /// A local or relative path.
    Path(String),
    /// A URL with a scheme (`https://`, `s3://`, `file://`, ...).
    Url(String),
    /// A virtual file system handler over a path or another source name.
    /// `/vsizip/{/vsicurl/https://h/a.zip}/b.tif` has handler `vsizip`,
    /// `inner` the braced name, and `path` `b.tif`.
    Vsi {
        handler: String,
        inner: Option<Box<Dsn>>,
        path: String,
    },
    /// A driver prefix: `NETCDF:"file.nc":sst`. `quoted` records whether the
    /// source was written in double quotes.
    Driver {
        prefix: String,
        source: Box<Dsn>,
        quoted: bool,
        subdataset: Option<String>,
    },
    /// A `vrt://` string: a source name with translate options.
    VrtString {
        source: Box<Dsn>,
        options: Vec<(String, String)>,
    },
    /// An inline VRT document (a string starting with `<VRTDataset`).
    Inline(String),
}

const ARCHIVES: &[&str] = &[
    "vsizip",
    "vsitar",
    "vsi7z",
    "vsirar",
    "vsigzip",
    "vsisubfile",
    "vsicrypt",
];

impl Dsn {
    pub fn parse(s: &str) -> Dsn {
        let t = s.trim();
        if t.starts_with("<VRTDataset") || t.starts_with("<OGRVRTDataSource") {
            return Dsn::Inline(t.to_string());
        }
        if let Some(rest) = s.strip_prefix("vrt://") {
            // Split at the last '?', so the source may itself be a vrt://
            // string with options (the reference library splits at the
            // first '?', which cannot nest). The two agree unless the
            // source name itself contains a '?'.
            let (src, query) = match rest.rfind('?') {
                Some(i) => (&rest[..i], &rest[i + 1..]),
                None => (rest, ""),
            };
            let options = query
                .split('&')
                .filter(|kv| !kv.is_empty())
                .map(|kv| match kv.split_once('=') {
                    Some((k, v)) => (k.to_string(), v.to_string()),
                    None => (kv.to_string(), String::new()),
                })
                .collect();
            return Dsn::VrtString {
                source: Box::new(Dsn::parse(src)),
                options,
            };
        }
        if let Some(rest) = s.strip_prefix("/vsi") {
            if let Some(slash) = rest.find('/') {
                let handler = format!("vsi{}", &rest[..slash]);
                let after = &rest[slash + 1..];
                if after.starts_with('{') {
                    if let Some(close) = matching_brace(after) {
                        let inner = Dsn::parse(&after[1..close]);
                        let path = after[close + 1..].trim_start_matches('/').to_string();
                        return Dsn::Vsi {
                            handler,
                            inner: Some(Box::new(inner)),
                            path,
                        };
                    }
                }
                return Dsn::Vsi {
                    handler,
                    inner: None,
                    path: after.to_string(),
                };
            }
        }
        if s.contains("://") && !driver_prefix(s) {
            return Dsn::Url(s.to_string());
        }
        if driver_prefix(s) {
            let colon = s.find(':').unwrap();
            let prefix = s[..colon].to_string();
            let rest = &s[colon + 1..];
            if let Some(q) = rest.strip_prefix('"') {
                if let Some(end) = q.find('"') {
                    let source = Dsn::parse(&q[..end]);
                    let tail = &q[end + 1..];
                    let subdataset = tail.strip_prefix(':').map(str::to_string);
                    return Dsn::Driver {
                        prefix,
                        source: Box::new(source),
                        quoted: true,
                        subdataset,
                    };
                }
            }
            let (src, sub) = match rest.find(':') {
                Some(i) => (&rest[..i], Some(rest[i + 1..].to_string())),
                None => (rest, None),
            };
            return Dsn::Driver {
                prefix,
                source: Box::new(Dsn::parse(src)),
                quoted: false,
                subdataset: sub,
            };
        }
        Dsn::Path(s.to_string())
    }

    /// The source at the bottom of the chain: the file or URL bytes come from.
    pub fn leaf(&self) -> &Dsn {
        match self {
            Dsn::Vsi { inner: Some(i), .. } => i.leaf(),
            Dsn::Driver { source, .. } | Dsn::VrtString { source, .. } => source.leaf(),
            d => d,
        }
    }

    /// True when the name points at another VRT (a nested VRT file, a
    /// `vrt://` string or an inline document): the planner recurses into it.
    pub fn is_virtual(&self) -> bool {
        match self {
            Dsn::VrtString { .. } | Dsn::Inline(_) => true,
            Dsn::Path(p) | Dsn::Url(p) => p.to_ascii_lowercase().ends_with(".vrt"),
            Dsn::Vsi {
                inner: None, path, ..
            } => path.to_ascii_lowercase().ends_with(".vrt"),
            _ => false,
        }
    }

    /// For a relative path, the path joined to `base` (a directory or URL
    /// prefix, without trailing separator), as `relativeToVRT="1"` does.
    pub fn relative_to(&self, base: &str) -> Dsn {
        match self {
            Dsn::Path(p) if !p.starts_with('/') && !base.is_empty() => {
                Dsn::parse(&format!("{base}/{p}"))
            }
            Dsn::VrtString { source, options } => Dsn::VrtString {
                source: Box::new(source.relative_to(base)),
                options: options.clone(),
            },
            d => d.clone(),
        }
    }

    /// Is this an archive handler (the path names a member inside it)?
    pub fn is_archive(&self) -> bool {
        matches!(self, Dsn::Vsi { handler, .. } if ARCHIVES.contains(&handler.as_str()))
    }
}

impl fmt::Display for Dsn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dsn::Path(s) | Dsn::Url(s) | Dsn::Inline(s) => f.write_str(s),
            Dsn::Vsi {
                handler,
                inner: None,
                path,
            } => write!(f, "/{handler}/{path}"),
            Dsn::Vsi {
                handler,
                inner: Some(i),
                path,
            } => {
                if path.is_empty() {
                    write!(f, "/{handler}/{{{i}}}")
                } else {
                    write!(f, "/{handler}/{{{i}}}/{path}")
                }
            }
            Dsn::Driver {
                prefix,
                source,
                quoted,
                subdataset,
            } => {
                if *quoted {
                    write!(f, "{prefix}:\"{source}\"")?;
                } else {
                    write!(f, "{prefix}:{source}")?;
                }
                if let Some(s) = subdataset {
                    write!(f, ":{s}")?;
                }
                Ok(())
            }
            Dsn::VrtString { source, options } => {
                write!(f, "vrt://{source}")?;
                for (i, (k, v)) in options.iter().enumerate() {
                    f.write_str(if i == 0 { "?" } else { "&" })?;
                    if v.is_empty() {
                        f.write_str(k)?;
                    } else {
                        write!(f, "{k}={v}")?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// `NAME:` at the start, at least two characters (so `C:/x` is a path), and
/// not a URL scheme (`https://`).
fn driver_prefix(s: &str) -> bool {
    let Some(colon) = s.find(':') else {
        return false;
    };
    let p = &s[..colon];
    p.len() >= 2
        && p.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s[colon..].starts_with("://")
}

fn matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(s: &str) -> Dsn {
        let d = Dsn::parse(s);
        assert_eq!(d.to_string(), s, "round trip of {s}");
        d
    }

    #[test]
    fn forms_round_trip() {
        // Strings as the R package dsn writes them: vrtcon(), sds(),
        // vsicurl(), vsizip().
        rt("vrt://myfile.nc?a_ullr=0,90,360,-90&bands=1,2,1");
        rt("NETCDF:\"file.nc\":sst");
        rt("/vsicurl/https://host/a.tif");
        rt("/vsizip/{/vsicurl/https://host/a.zip}/b.tif");
        rt("/vsizip//vsicurl/https://host/a.zip/b.tif");
        rt("ZARR:\"/vsicurl/https://h/x.zarr\":/sst");
        rt("HDF5:f.h5://grp/var");
        rt("tile_01.tif");
        rt("https://host/a.tif?sig=abc");
        rt("C:/data/a.tif");
    }

    #[test]
    fn typed_values() {
        match rt("NETCDF:\"/vsis3/b/k.nc\":sst") {
            Dsn::Driver {
                prefix,
                source,
                subdataset,
                ..
            } => {
                assert_eq!(prefix, "NETCDF");
                assert_eq!(
                    *source,
                    Dsn::Vsi {
                        handler: "vsis3".into(),
                        inner: None,
                        path: "b/k.nc".into()
                    }
                );
                assert_eq!(subdataset.as_deref(), Some("sst"));
            }
            d => panic!("{d:?}"),
        }
        let z = rt("/vsizip/{/vsicurl/https://host/a.zip}/b.tif");
        assert!(z.is_archive());
        assert_eq!(
            z.leaf(),
            &Dsn::Vsi {
                handler: "vsicurl".into(),
                inner: None,
                path: "https://host/a.zip".into()
            }
        );
        assert!(matches!(Dsn::parse("C:/data/a.tif"), Dsn::Path(_)));
    }

    #[test]
    fn vrt_strings_nest() {
        let d = rt("vrt://vrt://a.tif?bands=1?srcwin=0,0,10,10");
        match d {
            Dsn::VrtString { source, options } => {
                assert_eq!(
                    options,
                    vec![("srcwin".to_string(), "0,0,10,10".to_string())]
                );
                assert!(matches!(*source, Dsn::VrtString { .. }));
            }
            d => panic!("{d:?}"),
        }
        assert!(Dsn::parse("mosaic.vrt").is_virtual());
        assert!(Dsn::parse("<VRTDataset rasterXSize=\"1\"/>").is_virtual());
    }
}
