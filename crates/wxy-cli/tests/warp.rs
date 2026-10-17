//! `wxy warp` end to end: the GeoZarr it writes holds what GDAL's warper
//! writes for the same source and target grid (wxylib/fixtures, warp/),
//! from a GeoTIFF source and from the same data as a Zarr source. Cases
//! are those the wxy-warp oracle test gates on.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;
use wxy_source::{read_window, Window};

fn fixtures() -> PathBuf {
    std::env::var("WXY_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures"))
}

fn decode(bytes: &[u8], dtype: &str) -> Vec<f64> {
    match dtype {
        "int16" => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        "uint16" => bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        o => panic!("dtype {o}"),
    }
}

#[tokio::test]
async fn warp_writes_what_gdal_writes() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("warp/manifest.json")) else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let runs = [
        ("utm-to-4326", "cubic", "cog/utm-u16-lzw-pred2-3band.tif"),
        (
            "utm-to-4326-ovr",
            "bilinear",
            "cog/utm-u16-lzw-pred2-3band.tif",
        ),
        ("world-to-3031", "near", "cog/world-int16-deflate.tif"),
        ("world-to-3031", "near", "zarr/world-int16-spatial.zarr"),
    ];
    for (k, (name, method, source)) in runs.iter().enumerate() {
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == *name)
            .unwrap();
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let (w, h) = (
            case["width"].as_u64().unwrap(),
            case["height"].as_u64().unwrap(),
        );
        let extent = [
            gt[0],
            gt[3] + gt[5] * h as f64,
            gt[0] + gt[1] * w as f64,
            gt[3],
        ];
        let out = tmp.join(format!("cli-warp-{k}.zarr"));
        let _ = std::fs::remove_dir_all(&out);
        let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("warp")
            .arg(root.join(source))
            .arg(&out)
            .args([
                "--crs",
                case["crs"].as_str().unwrap(),
                "-r",
                method,
                "--extent",
            ])
            .args(extent.map(|v| v.to_string()))
            .args(["--size", &w.to_string(), &h.to_string()])
            .arg("--no-wrap")
            .status()
            .unwrap();
        assert!(status.success(), "{name} {method} from {source}");

        let ds = wxy_zarr::open_path(&out, "/data").await.unwrap();
        let shape = ds.collection.domains[0].shape();
        let got = read_window(&ds.source, &Window::new(vec![0; shape.len()], shape))
            .await
            .unwrap();
        let file = case["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["resampling"] == *method)
            .unwrap()["file"]
            .as_str()
            .unwrap();
        let want = decode(
            &std::fs::read(root.join("warp").join(file)).unwrap(),
            case["dtype"].as_str().unwrap(),
        );
        assert_eq!(got.values.to_f64(), want, "{name} {method} from {source}");
    }
}

#[test]
fn crs_the_writer_cannot_record_is_refused_before_reading() {
    let run = |crs: &str| {
        Command::new(env!("CARGO_BIN_EXE_wxy"))
            .args(["warp", "no-such-source.tif", "unused.zarr", "--crs", crs])
            .args(["--extent", "0", "0", "1", "1", "--size", "1", "1"])
            .output()
            .unwrap()
    };
    for crs in [
        "+proj=stere +lat_0=-90 +lat_ts=-71",
        "OGC:CRS84",
        "GEOGCRS[\"x\"]",
    ] {
        let out = run(crs);
        assert!(!out.status.success());
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("CRS") || err.contains("crs"), "{crs}: {err}");
        assert!(
            !err.contains("no-such-source"),
            "{crs} read the source: {err}"
        );
    }
}

/// Stage S4b end to end: a time x depth x lat x lon Zarr store, chunked
/// two time steps deep, streams through `wxy warp` batch by batch, and the
/// store it writes keeps the leading dims and holds the in-memory warp.
#[tokio::test]
async fn a_4d_store_streams_through() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let (src, out) = (tmp.join("cli-4d-src.zarr"), tmp.join("cli-4d-out.zarr"));
    for p in [&src, &out] {
        let _ = std::fs::remove_dir_all(p);
    }
    let (nt, nz, ny, nx) = (3u64, 2u64, 90u64, 180u64);
    let mut v = Vec::new();
    for t in 0..nt {
        for z in 0..nz {
            for r in 0..ny {
                for c in 0..nx {
                    let (lon, lat) = (-179.0 + 2.0 * c as f64, 89.0 - 2.0 * r as f64);
                    let x =
                        100.0 * (lon.to_radians() * (1 + t) as f64).sin() * lat.to_radians().cos();
                    v.push((x + 10.0 * z as f64) as f32);
                }
            }
        }
    }
    let values = wxy_source::Values::F32(v);
    let grid = wxy_zarr::GridArray {
        dims: ["time", "depth", "lat", "lon"].map(String::from).to_vec(),
        shape: vec![nt, nz, ny, nx],
        chunk: vec![2, 1, 45, 90],
        geotransform: [-180.0, 2.0, 0.0, 90.0, 0.0, -2.0],
        crs: Some("EPSG:4326".into()),
        nodata: None,
    };
    wxy_zarr::write_path(&src, "/sst", &values, &grid)
        .await
        .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
        .arg("warp")
        .arg(&src)
        .arg(&out)
        .args([
            "--source-array",
            "/sst",
            "--crs",
            "EPSG:3031",
            "-r",
            "bilinear",
        ])
        .args([
            "--extent", "-5e6", "-5e6", "5e6", "5e6", "--size", "60", "60",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let got = wxy_zarr::open_path(&out, "/data").await.unwrap();
    let d = &got.collection.domains[0];
    let names: Vec<&str> = d.dims.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["time", "depth", "y", "x"]);
    let shape = d.shape();
    assert_eq!(shape, vec![nt, nz, 60, 60]);
    let got = read_window(&got.source, &Window::new(vec![0; 4], shape))
        .await
        .unwrap();

    let ds = wxy_zarr::open_path(&src, "/sst").await.unwrap();
    let dst = wxy_warp::GridSpec::from_extent("EPSG:3031", [-5e6, 5e6, -5e6, 5e6], [60, 60]);
    let opts = wxy_warp::WarpOptions {
        resampling: wxy_warp::ResampleAlg::Bilinear,
        ..Default::default()
    };
    let want = wxy_warp::warp(&ds.source, &ds.collection.domains[0], &dst, &opts, None)
        .await
        .unwrap();
    assert_eq!(got.values, want.values);
}

/// A one-chunk Zarr v3 array written by hand (little-endian bytes, no
/// compression), so the tests below control every attribute.
fn raw_array(
    root: &std::path::Path,
    name: &str,
    dims: &[&str],
    shape: &[usize],
    data: &[u8],
    dtype: &str,
    attrs: Value,
) {
    let dir = root.join(name);
    std::fs::create_dir_all(dir.join("c")).unwrap();
    let meta = serde_json::json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": dims,
        "attributes": attrs,
    });
    std::fs::write(dir.join("zarr.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    let mut chunk = dir.join("c");
    for _ in 1..shape.len() {
        chunk = chunk.join("0");
    }
    std::fs::create_dir_all(&chunk).unwrap();
    std::fs::write(chunk.join("0"), data).unwrap();
}

fn raw_group(root: &std::path::Path) {
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("zarr.json"),
        br#"{"zarr_format":3,"node_type":"group","attributes":{}}"#,
    )
    .unwrap();
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn warp_cli(src: &std::path::Path, out: &std::path::Path, args: &[&str]) {
    let _ = std::fs::remove_dir_all(out);
    let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
        .arg("warp")
        .arg(src)
        .arg(out)
        .args(["--source-array", "/v"])
        .args(args)
        .status()
        .unwrap();
    assert!(status.success(), "{args:?}");
}

async fn read_all(path: &std::path::Path) -> Vec<f64> {
    let ds = wxy_zarr::open_path(path, "/data").await.unwrap();
    let shape = ds.collection.domains[0].shape();
    read_window(&ds.source, &Window::new(vec![0; shape.len()], shape))
        .await
        .unwrap()
        .values
        .to_f64()
}

/// Stage S5b end to end: the fixtures' curvilinear swath as a CF Zarr
/// store (2D `lon` and `lat` beside the data, named by `coordinates`)
/// warps through `wxy warp` to what GDAL's warper writes from the same
/// arrays through its geolocation transformer; with `--bounds`, placed by
/// its CF cell bounds, to what GDAL writes from the corner arrays.
#[tokio::test]
async fn a_cf_curvilinear_store_warps_as_gdal_does() {
    let root = fixtures();
    let Ok(file) = std::fs::File::open(root.join("curvilinear/manifest.json")) else {
        eprintln!("skipping: no fixtures at {}", root.display());
        return;
    };
    let manifest: Value = serde_json::from_reader(file).unwrap();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let dir = root.join("curvilinear");
    let src = tmp.join("cli-curvilinear-src.zarr");
    raw_group(&src);
    let (ny, nx) = (120, 160);
    let base = "swath-120x160";
    let read = |s: &str| std::fs::read(dir.join(format!("{base}-{s}.bin"))).unwrap();
    raw_array(
        &src,
        "v",
        &["y", "x"],
        &[ny, nx],
        &read("values"),
        "float32",
        serde_json::json!({"coordinates": "lat lon"}),
    );
    raw_array(
        &src,
        "lon",
        &["y", "x"],
        &[ny, nx],
        &read("lon"),
        "float64",
        serde_json::json!({"units": "degrees_east", "bounds": "lon_bnds"}),
    );
    raw_array(
        &src,
        "lat",
        &["y", "x"],
        &[ny, nx],
        &read("lat"),
        "float64",
        serde_json::json!({"units": "degrees_north", "bounds": "lat_bnds"}),
    );
    // CF cell bounds from the fixtures' corner arrays: vertices c(j,i),
    // c(j,i+1), c(j+1,i+1), c(j+1,i)
    for (name, file) in [("lon_bnds", "edges-lon"), ("lat_bnds", "edges-lat")] {
        let c: Vec<f64> = read(file)
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut v = Vec::with_capacity(ny * nx * 4);
        for j in 0..ny {
            for i in 0..nx {
                for (dj, di) in [(0, 0), (0, 1), (1, 1), (1, 0)] {
                    v.push(c[(j + dj) * (nx + 1) + i + di]);
                }
            }
        }
        raw_array(
            &src,
            name,
            &["y", "x", "nv"],
            &[ny, nx, 4],
            &f64_bytes(&v),
            "float64",
            serde_json::json!({}),
        );
    }

    let runs = [
        ("swath-to-4326", "bilinear"),
        ("swath-to-3031", "cubic"),
        ("swath-to-4326-coarse", "near"),
        ("swath-edges-to-4326", "bilinear"),
        ("swath-edges-to-3031", "near"),
    ];
    for (k, (name, method)) in runs.iter().enumerate() {
        let case = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == *name)
            .unwrap();
        let gt: Vec<f64> = case["geotransform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let (w, h) = (
            case["width"].as_u64().unwrap(),
            case["height"].as_u64().unwrap(),
        );
        let extent = [
            gt[0],
            gt[3] + gt[5] * h as f64,
            gt[0] + gt[1] * w as f64,
            gt[3],
        ];
        let extent = extent.map(|v| v.to_string());
        let (ws, hs) = (w.to_string(), h.to_string());
        let mut args = vec![
            "--crs",
            case["crs"].as_str().unwrap(),
            "-r",
            method,
            "--extent",
        ];
        args.extend(extent.iter().map(String::as_str));
        args.extend(["--size", &ws, &hs, "--no-wrap"]);
        if case["cell"] == "edge" {
            args.push("--bounds");
        }
        let out = tmp.join(format!("cli-curvilinear-{k}.zarr"));
        warp_cli(&src, &out, &args);
        let file = case["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["resampling"] == *method)
            .unwrap()["file"]
            .as_str()
            .unwrap();
        let want: Vec<f64> = std::fs::read(dir.join(file))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();
        assert_eq!(read_all(&out).await, want, "{name} {method}");
    }
}

/// 1D CF coordinates (xarray's layout): evenly spaced axes warp as the
/// affine grid they normalise to, uneven ones as the same axes spelled out
/// as 2D arrays (GDAL's route for irregular axes).
#[tokio::test]
async fn one_d_cf_coordinates_warp() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let (ny, nx) = (40usize, 60usize);
    let lat_regular: Vec<f64> = (0..ny).map(|j| -30.5 - j as f64).collect();
    let lat_uneven: Vec<f64> = (0..ny)
        .map(|j| -30.0 - j as f64 - 0.01 * (j * j) as f64)
        .collect();
    let lon: Vec<f64> = (0..nx).map(|i| 120.5 + i as f64).collect();
    let mut v = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            v.push(((i as f32) * 0.3).sin() * 100.0 + j as f32);
        }
    }
    let values: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let args = [
        "--crs",
        "EPSG:3577",
        "-r",
        "bilinear",
        "--extent",
        "-2e6",
        "-4e6",
        "2e6",
        "-2e6",
        "--size",
        "80",
        "40",
    ];

    let store = |name: &str, lat: &[f64], two_d: bool| {
        let p = tmp.join(name);
        raw_group(&p);
        let none = serde_json::json!({});
        if two_d {
            raw_array(
                &p,
                "v",
                &["y", "x"],
                &[ny, nx],
                &values,
                "float32",
                serde_json::json!({"coordinates": "lon lat"}),
            );
            let lon2: Vec<f64> = (0..ny).flat_map(|_| lon.clone()).collect();
            let lat2: Vec<f64> = lat
                .iter()
                .flat_map(|&y| std::iter::repeat_n(y, nx))
                .collect();
            raw_array(
                &p,
                "lon",
                &["y", "x"],
                &[ny, nx],
                &f64_bytes(&lon2),
                "float64",
                serde_json::json!({"standard_name": "longitude"}),
            );
            raw_array(
                &p,
                "lat",
                &["y", "x"],
                &[ny, nx],
                &f64_bytes(&lat2),
                "float64",
                serde_json::json!({"standard_name": "latitude"}),
            );
        } else {
            raw_array(
                &p,
                "v",
                &["lat", "lon"],
                &[ny, nx],
                &values,
                "float32",
                none.clone(),
            );
            raw_array(
                &p,
                "lon",
                &["lon"],
                &[nx],
                &f64_bytes(&lon),
                "float64",
                serde_json::json!({"units": "degrees_east"}),
            );
            raw_array(
                &p,
                "lat",
                &["lat"],
                &[ny],
                &f64_bytes(lat),
                "float64",
                serde_json::json!({"units": "degrees_north"}),
            );
        }
        p
    };

    // regular: the same as the affine store GridArray writes
    let src = store("cli-1d-regular.zarr", &lat_regular, false);
    let affine = tmp.join("cli-1d-affine.zarr");
    let _ = std::fs::remove_dir_all(&affine);
    let grid = wxy_zarr::GridArray {
        dims: ["y", "x"].map(String::from).to_vec(),
        shape: vec![ny as u64, nx as u64],
        chunk: vec![ny as u64, nx as u64],
        geotransform: [120.0, 1.0, 0.0, -30.0, 0.0, -1.0],
        crs: Some("EPSG:4326".into()),
        nodata: None,
    };
    wxy_zarr::write_path(&affine, "/v", &wxy_source::Values::F32(v.clone()), &grid)
        .await
        .unwrap();
    let (a, b) = (tmp.join("cli-1d-a.zarr"), tmp.join("cli-1d-b.zarr"));
    warp_cli(&src, &a, &args);
    warp_cli(&affine, &b, &args);
    let got = read_all(&a).await;
    assert!(got.iter().any(|&x| x != 0.0));
    assert_eq!(got, read_all(&b).await);

    // uneven: the same as 2D arrays
    let src = store("cli-1d-uneven.zarr", &lat_uneven, false);
    let src2 = store("cli-2d-uneven.zarr", &lat_uneven, true);
    warp_cli(&src, &a, &args);
    warp_cli(&src2, &b, &args);
    let got = read_all(&a).await;
    assert!(got.iter().any(|&x| x != 0.0));
    assert_eq!(got, read_all(&b).await);
}

/// A store as xarray writes it (fixtures cf/): an OISST-style month with
/// packed int16, float32 1D lat and lon, latitude ascending and longitude
/// 0 to 360. `wxy warp` writes every day as GDAL's warper writes it from a
/// north-up GeoTIFF of that day.
#[tokio::test]
async fn an_xarray_store_warps_as_gdal_does() {
    let dir = fixtures().join("cf");
    let Ok(file) = std::fs::File::open(dir.join("manifest.json")) else {
        eprintln!("skipping: no cf fixtures at {}", dir.display());
        return;
    };
    let m: Value = serde_json::from_reader(file).unwrap();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let gt: Vec<f64> = m["geotransform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let (w, h) = (m["width"].as_u64().unwrap(), m["height"].as_u64().unwrap());
    let extent = [
        gt[0],
        gt[3] + gt[5] * h as f64,
        gt[0] + gt[1] * w as f64,
        gt[3],
    ]
    .map(|v| v.to_string());
    for (k, r) in m["results"].as_array().unwrap().iter().enumerate() {
        let method = r["resampling"].as_str().unwrap();
        let out = tmp.join(format!("cli-cf-{k}.zarr"));
        let _ = std::fs::remove_dir_all(&out);
        let status = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("warp")
            .arg(dir.join(m["store"].as_str().unwrap()))
            .arg(&out)
            .args(["--source-array", m["array"].as_str().unwrap()])
            .args([
                "--crs",
                m["crs"].as_str().unwrap(),
                "-r",
                method,
                "--extent",
            ])
            .args(&extent)
            .args(["--size", &w.to_string(), &h.to_string()])
            .status()
            .unwrap();
        assert!(status.success(), "{method}");
        let ds = wxy_zarr::open_path(&out, "/data").await.unwrap();
        let shape = ds.collection.domains[0].shape();
        assert_eq!(shape, vec![m["days"].as_u64().unwrap(), 1, h, w]);
        let got = read_window(&ds.source, &Window::new(vec![0; 4], shape))
            .await
            .unwrap();
        let want = decode(
            &std::fs::read(dir.join(r["file"].as_str().unwrap())).unwrap(),
            "int16",
        );
        assert_eq!(got.values.to_f64(), want, "{method}");

        // the leading coordinates copied as they were; x and y as CF
        // 1D coordinates at cell centres
        let store = dir.join(m["store"].as_str().unwrap());
        for name in ["time", "zlev"] {
            assert_eq!(
                std::fs::read(out.join(name).join("zarr.json")).unwrap(),
                std::fs::read(store.join(name).join("zarr.json")).unwrap(),
                "{name}"
            );
        }
        let xs = wxy_zarr::open_path(&out, "/x").await.unwrap();
        let xs = read_window(&xs.source, &Window::new(vec![0], vec![w]))
            .await
            .unwrap()
            .values
            .to_f64();
        assert_eq!(xs[0], gt[0] + 0.5 * gt[1]);
        assert_eq!(xs[w as usize - 1], gt[0] + (w as f64 - 0.5) * gt[1]);
    }
}

/// `wxy points`: x,y lines in, one CSV row per point and band out, the
/// values GDAL reads at those points (fixtures points/).
#[test]
fn points_reads_what_gdal_reads() {
    use std::io::Write;
    let root = fixtures();
    let dir = root.join("points");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no point fixtures at {}", dir.display());
        return;
    }
    // nearest from lon/lat; bilinear in the file's own CRS, where the
    // positions are GDAL's to the bit
    for (name, crs, method, file) in [
        (
            "utm-u16-lzw-pred2-3band-4326",
            "EPSG:4326",
            "near",
            "values",
        ),
        (
            "utm-u16-lzw-pred2-3band-native",
            "EPSG:32755",
            "bilinear",
            "bilinear",
        ),
    ] {
        let f64s = |s: &str| -> Vec<f64> {
            std::fs::read(dir.join(format!("{name}-{s}.bin")))
                .unwrap()
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let (x, y, want) = (f64s("x"), f64s("y"), f64s(file));
        let mut child = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("points")
            .arg(root.join("cog/utm-u16-lzw-pred2-3band.tif"))
            .args(["--crs", crs, "--no-wrap", "-r", method])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        {
            let mut stdin = child.stdin.take().unwrap();
            writeln!(stdin, "x,y").unwrap();
            for (a, b) in x.iter().zip(&y) {
                writeln!(stdin, "{a},{b}").unwrap();
            }
        }
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("x,y,band,value"));
        let n = x.len();
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), n * 3);
        let found = std::fs::read(dir.join(format!("{name}-found.bin"))).unwrap();
        for (r, row) in rows.iter().enumerate() {
            let (i, band) = (r / 3, r % 3);
            let f: Vec<&str> = row.split(',').collect();
            assert_eq!(f[0].parse::<f64>().unwrap(), x[i]);
            assert_eq!(f[2], band.to_string());
            let w = want[band * n + i];
            let missing = if method == "near" {
                found[i] == 0
            } else {
                w.is_nan()
            };
            if missing {
                assert_eq!(f[3], "", "{row}");
            } else {
                assert_eq!(f[3].parse::<f64>().unwrap(), w, "{name} {row}");
            }
        }
    }
}

/// `wxy zonal`: GeoJSON polygons in, one CSV row per feature and band out,
/// over the cells GDAL's rasterizer burns for each polygon (fixtures
/// polygons/), with and without --all-touched.
#[tokio::test]
async fn zonal_summarises_the_cells_gdal_burns() {
    let root = fixtures();
    let dir = root.join("polygons");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no polygon fixtures at {}", dir.display());
        return;
    }
    let name = "utm-u16-lzw-pred2-3band-native";
    let read = |s: &str| std::fs::read(dir.join(format!("{name}-{s}"))).unwrap();
    let coords: Vec<f64> = read("coords.bin")
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let rings: Vec<Vec<usize>> = serde_json::from_slice(&read("rings.json")).unwrap();
    let cells: Value = serde_json::from_slice(&read("cells.json")).unwrap();
    // GeoJSON with every coordinate written to round trip
    let mut at = 0;
    let mut features = Vec::new();
    for (k, p) in rings.iter().enumerate() {
        let mut rs = Vec::new();
        for &n in p {
            let pts: Vec<String> = (0..n)
                .map(|i| format!("[{},{}]", coords[2 * (at + i)], coords[2 * (at + i) + 1]))
                .collect();
            rs.push(format!("[{}]", pts.join(",")));
            at += n;
        }
        features.push(format!(
            r#"{{"type":"Feature","id":"p{k}","geometry":{{"type":"Polygon","coordinates":[{}]}}}}"#,
            rs.join(",")
        ));
    }
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let gj = tmp.join("zonal.geojson");
    std::fs::write(
        &gj,
        format!(
            r#"{{"type":"FeatureCollection","features":[{}]}}"#,
            features.join(",")
        ),
    )
    .unwrap();

    let tif = root.join("cog/utm-u16-lzw-pred2-3band.tif");
    let ds = wxy_tiff::open_path(&tif).await.unwrap();
    let shape = ds.collection.domains[0].shape();
    let (h, w) = (shape[1], shape[2]);
    let all = read_window(&ds.levels[0], &Window::new(vec![0; 3], shape.clone()))
        .await
        .unwrap()
        .values
        .to_f64();
    for (mode, flag) in [("centre", None), ("all_touched", Some("--all-touched"))] {
        let out = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("zonal")
            .arg(&tif)
            .arg(&gj)
            .args(["--crs", "EPSG:32755"])
            .args(flag)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("feature,band,cells,count,sum,mean,min,max")
        );
        let rows: Vec<Vec<String>> = lines
            .map(|l| l.split(',').map(String::from).collect())
            .collect();
        assert_eq!(rows.len(), rings.len() * 3);
        for (k, runs) in cells[mode].as_array().unwrap().iter().enumerate() {
            let idx: Vec<u64> = runs
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| {
                    let (s, n) = (r[0].as_u64().unwrap(), r[1].as_u64().unwrap());
                    s..s + n
                })
                .collect();
            for b in 0..3u64 {
                let row = &rows[k * 3 + b as usize];
                assert_eq!(row[0], format!("p{k}"));
                assert_eq!(row[1], b.to_string());
                assert_eq!(row[2], idx.len().to_string(), "{mode} p{k}");
                let v: Vec<f64> = idx.iter().map(|&i| all[(b * h * w + i) as usize]).collect();
                assert_eq!(row[3], v.len().to_string(), "{mode} p{k}");
                if v.is_empty() {
                    continue;
                }
                let sum: f64 = v.iter().sum();
                assert_eq!(row[4].parse::<f64>().unwrap(), sum, "{mode} p{k} band {b}");
                let min = v.iter().copied().fold(f64::INFINITY, f64::min);
                let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(row[6].parse::<f64>().unwrap(), min);
                assert_eq!(row[7].parse::<f64>().unwrap(), max);
            }
        }
    }

    // --coverage: polygons inside the grid count their own area, in cells
    let out = Command::new(env!("CARGO_BIN_EXE_wxy"))
        .arg("zonal")
        .arg(&tif)
        .arg(&gj)
        .args(["--crs", "EPSG:32755", "--coverage"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let rows: Vec<Vec<String>> = text
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(String::from).collect())
        .collect();
    let gt = match &ds.collection.domains[0].primary_for(2).unwrap().positions {
        wxy_domain::Positions::Affine(a) => a.0,
        _ => unreachable!(),
    };
    let mut at = 0;
    let mut checked = 0;
    for (k, p) in rings.iter().enumerate() {
        let ring = |n: usize, at: usize| -> Vec<[f64; 2]> {
            (0..n)
                .map(|i| [coords[2 * (at + i)], coords[2 * (at + i) + 1]])
                .collect()
        };
        let rs: Vec<Vec<[f64; 2]>> = p
            .iter()
            .map(|&n| {
                let r = ring(n, at);
                at += n;
                r
            })
            .collect();
        let inside = rs.iter().flatten().all(|v| {
            let (c, r) = ((v[0] - gt[0]) / gt[1], (v[1] - gt[3]) / gt[5]);
            c >= 0.0 && c <= w as f64 && r >= 0.0 && r <= h as f64
        });
        if !inside {
            continue;
        }
        let shoelace = |r: &Vec<[f64; 2]>| {
            r.windows(2)
                .map(|s| s[0][0] * s[1][1] - s[1][0] * s[0][1])
                .sum::<f64>()
                .abs()
                / 2.0
        };
        let area = shoelace(&rs[0]) - rs[1..].iter().map(shoelace).sum::<f64>();
        let want = area / (gt[1] * gt[5]).abs();
        let got: f64 = rows[k * 3][3].parse().unwrap();
        assert!(
            (got - want).abs() < 1e-4 * want.max(1.0),
            "p{k}: {got} of {want}"
        );
        checked += 1;
    }
    assert!(checked > 0);

    // --dry-run: the plan as JSON, and no statistics
    let out = Command::new(env!("CARGO_BIN_EXE_wxy"))
        .arg("zonal")
        .arg(&tif)
        .arg(&gj)
        .args(["--crs", "EPSG:32755", "--coverage", "--dry-run"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let plan: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(plan["operation"], "zonal");
    assert_eq!(plan["cells"], "coverage");
    assert_eq!(plan["zones"], rings.len());
    let keys = plan["fetch"]["keys"].as_array().unwrap();
    assert_eq!(plan["fetch"]["chunks"], keys.len());
    assert!(!keys.is_empty());
    assert!(plan["fetch"]["stored_bytes"].as_u64().unwrap() > 0);
    assert_eq!(plan["source"]["shape"], serde_json::json!(shape));
}

/// m05 pilot C1: `--cache DIR` keeps the bytes a warp reads; a second run
/// reads nothing from the file and writes the same output; a dry run says
/// what the cache holds; and the cached rows, joined to the file's
/// reference table on (path, offset, size), give back the plan's chunks.
#[tokio::test]
async fn a_cached_warp_reads_nothing_the_second_time() {
    let src = fixtures().join("cog/world-int16-deflate.tif");
    if !src.exists() {
        eprintln!("skipping: no fixture at {}", src.display());
        return;
    }
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cached-warp");
    let _ = std::fs::remove_dir_all(&tmp);
    let cache = tmp.join("cache");
    let wxy = |out: &str, extra: &[&str]| {
        let o = Command::new(env!("CARGO_BIN_EXE_wxy"))
            .arg("warp")
            .arg(&src)
            .arg(tmp.join(out))
            .args([
                "--crs",
                "EPSG:3031",
                "--extent",
                "-4e6",
                "-4e6",
                "4e6",
                "4e6",
            ])
            .args(["--size", "145", "145", "-r", "bilinear", "--cache"])
            .arg(&cache)
            .args(extra)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        o
    };
    let dry = |label: &str| -> Value {
        let plan: Value = serde_json::from_slice(&wxy("dry.zarr", &["--dry-run"]).stdout).unwrap();
        eprintln!("{label}: {}", plan["fetch"]["cache"]);
        plan
    };
    let held = |p: &Value| p["fetch"]["cache"]["held"].as_u64().unwrap();
    let missing = |p: &Value| p["fetch"]["cache"]["missing"].as_u64().unwrap();

    let before = dry("before");
    let n = before["fetch"]["chunks"].as_u64().unwrap();
    assert!(n > 1);
    assert_eq!((held(&before), missing(&before)), (0, n));

    let first = String::from_utf8(wxy("a.zarr", &[]).stderr).unwrap();
    assert!(!first.contains(" 0 from the source"), "{first}");
    let after = dry("after");
    assert_eq!((held(&after), missing(&after)), (n, 0));
    assert_eq!(
        after["fetch"]["cache"]["held_bytes"],
        after["fetch"]["stored_bytes"]
    );

    let second = String::from_utf8(wxy("b.zarr", &[]).stderr).unwrap();
    assert!(second.contains(", 0 from the source (0 bytes)"), "{second}");
    fn files(dir: &std::path::Path, out: &mut Vec<(PathBuf, Vec<u8>)>, root: &std::path::Path) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                files(&p, out, root);
            } else {
                out.push((
                    p.strip_prefix(root).unwrap().into(),
                    std::fs::read(&p).unwrap(),
                ));
            }
        }
    }
    let all = |d: &str| {
        let mut v = Vec::new();
        files(&tmp.join(d), &mut v, &tmp.join(d));
        v
    };
    let (a, b) = (all("a.zarr"), all("b.zarr"));
    assert!(a.len() > 1);
    assert_eq!(a, b, "the cached run wrote a different store");

    // the join: cached rows to the reference table, on (path, offset, size)
    let refs = wxy_virtual::scan_tiff(&src).await.unwrap().refs;
    let mut keys = std::collections::BTreeSet::new();
    for f in wxy_cache::fragments(&cache).unwrap() {
        for row in wxy_cache::read_fragment(&f).unwrap() {
            if let Some(r) = refs
                .iter()
                .find(|r| r.url == row.source && r.offset == row.offset && r.length == row.size)
            {
                keys.insert(r.key.clone());
            }
        }
    }
    let planned: std::collections::BTreeSet<Vec<u64>> = after["fetch"]["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| serde_json::from_value(k.clone()).unwrap())
        .collect();
    assert_eq!(keys, planned);
}
