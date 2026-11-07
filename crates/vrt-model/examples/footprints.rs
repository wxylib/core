//! List the sources of a classic VRT that meet a box, with the URL to fetch.
//!
//! cargo run -p vrt-model --example footprints -- <file.vrt> [xmin xmax ymin ymax]
//!
//! Relative source names resolve against the VRT file's directory; give the
//! VRT's URL as WXY_VRT_BASE to resolve them against that instead.

use vrt_model::{
    fetch_url, footprints, sources_in_bounds, Bounds, Dataset, Dsn, Resolver, UrlConfig,
};

struct NoIo;
impl Resolver for NoIo {
    fn size(&self, _: &Dsn) -> Option<(usize, usize)> {
        None
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = a
        .first()
        .expect("usage: footprints <file.vrt> [xmin xmax ymin ymax]");
    let ds = Dataset::from_xml(&std::fs::read_to_string(path).unwrap()).unwrap();
    let base = std::env::var("WXY_VRT_BASE").unwrap_or_else(|_| vrt_model::plan::parent_of(path));
    let all = footprints(&ds, 1, &base, &NoIo).unwrap();
    println!("{} sources on band 1", all.len());
    if a.len() < 5 {
        return;
    }
    let v: Vec<f64> = a[1..5].iter().map(|s| s.parse().unwrap()).collect();
    let q = Bounds::new(v[0], v[1], v[2], v[3]);
    let hit = sources_in_bounds(&ds, 1, &q, &base, &NoIo).unwrap();
    println!("{} meet {:?}", hit.len(), q);
    let cfg = UrlConfig::default();
    for f in hit {
        match fetch_url(&f.dsn, &cfg) {
            Ok(u) => println!("{u}"),
            Err(e) => println!("# {}: {e}", f.dsn),
        }
    }
}
