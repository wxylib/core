//! Source names as URLs a client can fetch.
//!
//! A VRT names its sources the reference library's way: `/vsicurl/https://...`,
//! `/vsis3/bucket/key`, `/vsigs/bucket/key`. A browser or an `object_store`
//! client wants a plain `https://` URL. [`fetch_url`] maps one to the other
//! where the mapping is exact, and says why it is not where it is not
//! (archives, driver subdatasets, nested VRTs, local paths). This is the
//! HTTP row of design m29's name mapping.
//!
//! Object store hosts follow the reference library's configuration options
//! (`AWS_S3_ENDPOINT`, `AWS_VIRTUAL_HOSTING`, `AWS_REGION`, `AWS_HTTPS`,
//! `AZURE_STORAGE_ACCOUNT`), held in [`UrlConfig`] rather than read from the
//! environment, so the mapping stays free of I/O and global state.

use std::fmt;

use crate::dsn::Dsn;

/// Hosts for object store names. The default is public AWS S3 with virtual
/// hosting, Google Cloud Storage, and no Azure account.
#[derive(Debug, Clone, PartialEq)]
pub struct UrlConfig {
    /// `AWS_S3_ENDPOINT`: a host (with optional port), or a URL with scheme.
    pub s3_endpoint: Option<String>,
    /// `AWS_VIRTUAL_HOSTING`: bucket in the host name. `None` means the
    /// reference library's default, yes, also for a custom endpoint (most
    /// non-AWS endpoints need `Some(false)`). A bucket name with a dot is
    /// always addressed by path, as the reference library does.
    pub s3_virtual_hosting: Option<bool>,
    /// `AWS_REGION`: adds the region to the AWS host name.
    pub s3_region: Option<String>,
    /// `AWS_HTTPS`: false gives `http://` for an endpoint without a scheme.
    pub https: bool,
    /// Host for `/vsigs/`.
    pub gs_host: String,
    /// `AZURE_STORAGE_ACCOUNT`, for `/vsiaz/` and `/vsiadls/`.
    pub az_account: Option<String>,
}

impl Default for UrlConfig {
    fn default() -> Self {
        UrlConfig {
            s3_endpoint: None,
            s3_virtual_hosting: None,
            s3_region: None,
            https: true,
            gs_host: "storage.googleapis.com".into(),
            az_account: None,
        }
    }
}

/// Why a source name has no single fetchable URL.
#[derive(Debug, Clone, PartialEq)]
pub enum NotFetchable {
    /// A local or relative path. Resolve relative names against the VRT's
    /// URL first ([`Dsn::relative_to`]).
    Local(String),
    /// A member of an archive or a sub-range (`/vsizip/`, `/vsisubfile/`, ...).
    Archive(String),
    /// A driver subdataset (`NETCDF:"f.nc":sst`). `file` is the URL of the
    /// file that holds it, when that maps.
    Subdataset { name: String, file: Option<String> },
    /// A nested VRT (`vrt://` string or inline document).
    Virtual(String),
    /// A handler or scheme with no URL mapping here, or one that needs
    /// configuration that is absent.
    Unmapped(String),
}

impl fmt::Display for NotFetchable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotFetchable::Local(s) => write!(f, "local path {s}"),
            NotFetchable::Archive(s) => write!(f, "archive member or range {s}"),
            NotFetchable::Subdataset { name, file } => match file {
                Some(u) => write!(f, "subdataset {name} of {u}"),
                None => write!(f, "subdataset {name}"),
            },
            NotFetchable::Virtual(s) => write!(f, "virtual dataset {s}"),
            NotFetchable::Unmapped(s) => write!(f, "no URL mapping for {s}"),
        }
    }
}

/// Percent-encode an object key, keeping `/` and RFC 3986 unreserved bytes.
fn encode_key(k: &str) -> String {
    let mut out = String::with_capacity(k.len());
    for b in k.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn split_bucket(path: &str) -> Option<(&str, &str)> {
    let p = path.trim_start_matches('/');
    let (b, k) = p.split_once('/')?;
    if b.is_empty() || k.is_empty() {
        None
    } else {
        Some((b, k))
    }
}

fn s3_url(bucket: &str, key: &str, cfg: &UrlConfig) -> String {
    let key = encode_key(key);
    match &cfg.s3_endpoint {
        Some(ep) => {
            let (scheme, host) = match ep.split_once("://") {
                Some((s, h)) => (s.to_string(), h.trim_end_matches('/').to_string()),
                None => (
                    if cfg.https { "https" } else { "http" }.to_string(),
                    ep.trim_end_matches('/').to_string(),
                ),
            };
            if cfg.s3_virtual_hosting.unwrap_or(true) && !bucket.contains('.') {
                format!("{scheme}://{bucket}.{host}/{key}")
            } else {
                format!("{scheme}://{host}/{bucket}/{key}")
            }
        }
        None => {
            let host = match &cfg.s3_region {
                Some(r) => format!("s3.{r}.amazonaws.com"),
                None => "s3.amazonaws.com".to_string(),
            };
            if cfg.s3_virtual_hosting.unwrap_or(true) && !bucket.contains('.') {
                format!("https://{bucket}.{host}/{key}")
            } else {
                format!("https://{host}/{bucket}/{key}")
            }
        }
    }
}

fn object_url(handler: &str, path: &str, cfg: &UrlConfig) -> Result<String, NotFetchable> {
    let unmapped = || NotFetchable::Unmapped(format!("/{handler}/{path}"));
    match handler {
        "vsis3" | "vsis3_streaming" => {
            let (b, k) = split_bucket(path).ok_or_else(unmapped)?;
            Ok(s3_url(b, k, cfg))
        }
        "vsigs" | "vsigs_streaming" => {
            let (b, k) = split_bucket(path).ok_or_else(unmapped)?;
            Ok(format!("https://{}/{b}/{}", cfg.gs_host, encode_key(k)))
        }
        "vsiaz" | "vsiaz_streaming" | "vsiadls" => {
            let (c, k) = split_bucket(path).ok_or_else(unmapped)?;
            let a = cfg.az_account.as_deref().ok_or_else(unmapped)?;
            let host = if handler == "vsiadls" { "dfs" } else { "blob" };
            Ok(format!(
                "https://{a}.{host}.core.windows.net/{c}/{}",
                encode_key(k)
            ))
        }
        _ => Err(unmapped()),
    }
}

/// The plain URL a client fetches for `dsn`, or why there is none.
///
/// - `https://` and `http://` URLs, and `/vsicurl/` over one, map to that URL.
/// - `/vsis3/`, `/vsigs/`, `/vsiaz/`, `/vsiadls/` (and their `_streaming`
///   forms) and `s3://`, `gs://` URLs map to the object store's HTTP URL.
/// - Archive members, ranges, driver subdatasets, nested VRTs, local paths
///   and other handlers do not map, and the error says which they are.
pub fn fetch_url(dsn: &Dsn, cfg: &UrlConfig) -> Result<String, NotFetchable> {
    match dsn {
        Dsn::Url(u) => {
            let lower = u.to_ascii_lowercase();
            if lower.starts_with("https://") || lower.starts_with("http://") {
                Ok(u.clone())
            } else if let Some(p) = u.strip_prefix("s3://") {
                object_url("vsis3", p, cfg)
            } else if let Some(p) = u.strip_prefix("gs://") {
                object_url("vsigs", p, cfg)
            } else if lower.starts_with("file://") {
                Err(NotFetchable::Local(u.clone()))
            } else {
                Err(NotFetchable::Unmapped(u.clone()))
            }
        }
        Dsn::Path(p) => Err(NotFetchable::Local(p.clone())),
        Dsn::Vsi {
            handler,
            inner: None,
            path,
        } if handler == "vsicurl" || handler == "vsicurl_streaming" => {
            let lower = path.to_ascii_lowercase();
            if lower.starts_with("https://") || lower.starts_with("http://") {
                Ok(path.clone())
            } else {
                // `/vsicurl?url=...` and option forms are not split by the
                // name parser yet.
                Err(NotFetchable::Unmapped(dsn.to_string()))
            }
        }
        Dsn::Vsi { .. } if dsn.is_archive() => Err(NotFetchable::Archive(dsn.to_string())),
        Dsn::Vsi {
            handler,
            inner: None,
            path,
        } => object_url(handler, path, cfg),
        Dsn::Vsi { .. } => Err(NotFetchable::Unmapped(dsn.to_string())),
        Dsn::Driver { source, .. } => Err(NotFetchable::Subdataset {
            name: dsn.to_string(),
            file: fetch_url(source, cfg).ok(),
        }),
        Dsn::VrtString { .. } => Err(NotFetchable::Virtual(dsn.to_string())),
        Dsn::Inline(_) => Err(NotFetchable::Virtual("inline VRT".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Result<String, NotFetchable> {
        fetch_url(&Dsn::parse(s), &UrlConfig::default())
    }

    #[test]
    fn http_forms() {
        let u = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n45w120/USGS_13_n45w120.tif";
        assert_eq!(url(&format!("/vsicurl/{u}")).unwrap(), u);
        assert_eq!(url(u).unwrap(), u);
        assert_eq!(
            url("/vsicurl_streaming/http://h/a.tif").unwrap(),
            "http://h/a.tif"
        );
    }

    #[test]
    fn object_stores() {
        assert_eq!(
            url("/vsis3/sentinel-cogs/a b/c.tif").unwrap(),
            "https://sentinel-cogs.s3.amazonaws.com/a%20b/c.tif"
        );
        assert_eq!(
            url("s3://my.bucket/k.tif").unwrap(),
            "https://s3.amazonaws.com/my.bucket/k.tif"
        );
        assert_eq!(
            url("/vsigs/gcp-public-data-landsat/x.TIF").unwrap(),
            "https://storage.googleapis.com/gcp-public-data-landsat/x.TIF"
        );
        let cfg = UrlConfig {
            s3_endpoint: Some("transfer.data.aad.gov.au".into()),
            s3_virtual_hosting: Some(false),
            ..Default::default()
        };
        assert_eq!(
            fetch_url(&Dsn::parse("/vsis3/aadc-transfer/e2e/mdim.vrt"), &cfg).unwrap(),
            "https://transfer.data.aad.gov.au/aadc-transfer/e2e/mdim.vrt"
        );
        let cfg = UrlConfig {
            s3_region: Some("us-west-2".into()),
            ..Default::default()
        };
        assert_eq!(
            fetch_url(&Dsn::parse("/vsis3/b/k.tif"), &cfg).unwrap(),
            "https://b.s3.us-west-2.amazonaws.com/k.tif"
        );
        assert!(matches!(
            url("/vsiaz/c/k.tif"),
            Err(NotFetchable::Unmapped(_))
        ));
        let cfg = UrlConfig {
            az_account: Some("acct".into()),
            ..Default::default()
        };
        assert_eq!(
            fetch_url(&Dsn::parse("/vsiaz/c/k.tif"), &cfg).unwrap(),
            "https://acct.blob.core.windows.net/c/k.tif"
        );
    }

    #[test]
    fn not_fetchable() {
        assert!(matches!(url("a.tif"), Err(NotFetchable::Local(_))));
        assert!(matches!(url("/data/a.tif"), Err(NotFetchable::Local(_))));
        assert!(matches!(
            url("/vsizip/{/vsicurl/https://h/a.zip}/b.tif"),
            Err(NotFetchable::Archive(_))
        ));
        assert_eq!(
            url("NETCDF:\"/vsicurl/https://h/f.nc\":sst"),
            Err(NotFetchable::Subdataset {
                name: "NETCDF:\"/vsicurl/https://h/f.nc\":sst".into(),
                file: Some("https://h/f.nc".into())
            })
        );
        assert!(matches!(
            url("vrt:///vsicurl/https://h/a.tif?bands=1"),
            Err(NotFetchable::Virtual(_))
        ));
        assert!(matches!(
            url("/vsimem/a.tif"),
            Err(NotFetchable::Unmapped(_))
        ));
    }

    #[test]
    fn relative_then_fetch() {
        let d = Dsn::parse("n45w120/a.tif").relative_to("https://h/TIFF/current");
        assert_eq!(
            fetch_url(&d, &UrlConfig::default()).unwrap(),
            "https://h/TIFF/current/n45w120/a.tif"
        );
        let d = Dsn::parse("n45w120/a.tif").relative_to("/vsis3/b/TIFF");
        assert_eq!(
            fetch_url(&d, &UrlConfig::default()).unwrap(),
            "https://b.s3.amazonaws.com/TIFF/n45w120/a.tif"
        );
    }
}
