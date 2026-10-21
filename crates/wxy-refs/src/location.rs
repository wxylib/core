//! Object stores for the URLs a table and its references name.
//!
//! `file://`, `http(s)://` and `s3://bucket/key`. An S3 store takes its
//! credentials, region, endpoint and addressing style from the `AWS_*`
//! environment as `object_store` reads it (`AWS_ENDPOINT`,
//! `AWS_VIRTUAL_HOSTED_STYLE_REQUEST`, ...), and also from the names GDAL's
//! `/vsis3/` reads, so one environment serves both: `AWS_S3_ENDPOINT` (a
//! host, with `AWS_HTTPS=NO` for plain HTTP) and `AWS_VIRTUAL_HOSTING`.
//! `AWS_NO_SIGN_REQUEST=YES` reads a public bucket unsigned.

use std::sync::Arc;

use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use wxy_source::{Error, Result};

/// The part of `url` that names its store (scheme and authority, or
/// `file://`), and its location in that store.
pub fn split(url: &str) -> Result<(String, ObjectPath)> {
    let bad = |e: &dyn std::fmt::Display| Error::Fetch(format!("{url}: {e}"));
    let u = url::Url::parse(url).map_err(|e| bad(&e))?;
    match u.scheme() {
        "file" => {
            let p = u.to_file_path().map_err(|_| bad(&"not a local path"))?;
            Ok((
                "file://".to_string(),
                ObjectPath::from_filesystem_path(&p).map_err(|e| bad(&e))?,
            ))
        }
        "http" | "https" | "s3" => Ok((
            u[..url::Position::BeforePath].to_string(),
            ObjectPath::from_url_path(u.path()).map_err(|e| bad(&e))?,
        )),
        s => Err(Error::Unsupported(format!("location scheme {s}: {url}"))),
    }
}

/// A store for `base`, as [`split`] returns it.
pub fn store(base: &str) -> Result<Arc<dyn ObjectStore>> {
    let bad = |e: &dyn std::fmt::Display| Error::Fetch(format!("{base}: {e}"));
    if base == "file://" {
        return Ok(Arc::new(object_store::local::LocalFileSystem::new()));
    }
    let u = url::Url::parse(base).map_err(|e| bad(&e))?;
    Ok(match u.scheme() {
        "s3" => Arc::new(
            s3_from_env(u.host_str().ok_or_else(|| bad(&"no bucket"))?)
                .build()
                .map_err(|e| bad(&e))?,
        ),
        _ => Arc::new(
            object_store::http::HttpBuilder::new()
                .with_url(base)
                .build()
                .map_err(|e| bad(&e))?,
        ),
    })
}

fn yes(v: &str) -> bool {
    matches!(v.to_ascii_uppercase().as_str(), "YES" | "TRUE" | "ON" | "1")
}

/// An S3 builder for `bucket` from the environment: `object_store`'s own
/// `AWS_*` names, then GDAL's where `object_store` has none set.
pub fn s3_from_env(bucket: &str) -> object_store::aws::AmazonS3Builder {
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let mut b = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);
    if var("AWS_ENDPOINT").is_none() && var("AWS_ENDPOINT_URL").is_none() {
        if let Some(host) = var("AWS_S3_ENDPOINT") {
            let http = var("AWS_HTTPS").is_some_and(|v| !yes(&v));
            let endpoint = if host.contains("://") {
                host
            } else {
                format!("{}://{host}", if http { "http" } else { "https" })
            };
            b = b
                .with_allow_http(endpoint.starts_with("http://"))
                .with_endpoint(endpoint);
        }
    }
    if var("AWS_VIRTUAL_HOSTED_STYLE_REQUEST").is_none() {
        if let Some(v) = var("AWS_VIRTUAL_HOSTING") {
            b = b.with_virtual_hosted_style_request(yes(&v));
        }
    }
    if var("AWS_NO_SIGN_REQUEST").is_some_and(|v| yes(&v)) {
        b = b.with_skip_signature(true);
    }
    b
}
