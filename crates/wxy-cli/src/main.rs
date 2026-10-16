//! `wxy`: wxylib from the shell.
//!
//! ```sh
//! wxy warp in.tif out.zarr --crs EPSG:3031 --extent -4e6 -4e6 4e6 4e6 --size 145 145
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use wxy_source::ChunkSource;
use wxy_warp::{GridSpec, ResampleAlg, WarpOptions};

#[derive(Parser)]
#[command(name = "wxy", version, about = "wxylib from the shell")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Warp a GeoTIFF or Zarr array onto a target grid, written as GeoZarr.
    Warp(Warp),
}

#[derive(Parser)]
struct Warp {
    /// Source: a GeoTIFF file, or a Zarr store directory.
    source: PathBuf,
    /// Output Zarr store directory, created if missing.
    output: PathBuf,
    /// Target CRS as an EPSG:NNNN code, such as EPSG:3031. (The CRS backend
    /// also reads PROJ strings, but GeoZarr's proj: has no place to record
    /// one; WKT2 and PROJJSON wait on a PROJ-backed CRS.)
    #[arg(long)]
    crs: String,
    /// Target extent: xmin ymin xmax ymax, in target CRS units.
    #[arg(long, required = true, num_args = 4, value_names = ["XMIN", "YMIN", "XMAX", "YMAX"], allow_negative_numbers = true)]
    extent: Vec<f64>,
    /// Target size in cells: ncol nrow.
    #[arg(long, required = true, num_args = 2, value_names = ["NCOL", "NROW"])]
    size: Vec<usize>,
    /// Resampling method.
    #[arg(long, short, value_enum, default_value_t = Resampling::Near)]
    resampling: Resampling,
    /// Maximum error of the approximate transformer, in pixels; 0 is exact.
    #[arg(long, default_value_t = 0.125)]
    error_threshold: f64,
    /// Nodata, overriding the source's.
    #[arg(long, allow_negative_numbers = true)]
    nodata: Option<f64>,
    /// Array to read when the source is Zarr.
    #[arg(long, default_value = "/data")]
    source_array: String,
    /// Array to write in the output store.
    #[arg(long, default_value = "/data")]
    array: String,
    /// Leave target cells beyond a 360 degree lon/lat source's edge empty,
    /// as GDAL's warper does, instead of wrapping longitude and reading pole rows.
    #[arg(long)]
    no_wrap: bool,
    /// Replace the output array if it exists.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Resampling {
    Near,
    Bilinear,
    Cubic,
}

impl From<Resampling> for ResampleAlg {
    fn from(r: Resampling) -> Self {
        match r {
            Resampling::Near => ResampleAlg::NearestNeighbour,
            Resampling::Bilinear => ResampleAlg::Bilinear,
            Resampling::Cubic => ResampleAlg::Cubic,
        }
    }
}

type Failure = Box<dyn std::error::Error>;

/// An opened source: chunks, index-space domain and nodata.
struct Opened {
    source: Box<dyn ChunkSource>,
    domain: wxy_domain::Domain,
    nodata: Option<f64>,
}

async fn open(path: &Path, array: &str) -> Result<Opened, Failure> {
    if path.is_dir() {
        let ds = wxy_zarr::open_path(path, array).await?;
        let mut c = ds.collection;
        Ok(Opened {
            source: Box::new(ds.source),
            nodata: c.fields[0].values.nodata,
            domain: c.domains.swap_remove(0),
        })
    } else {
        let ds = wxy_tiff::open_path(path).await?;
        let mut c = ds.collection;
        let mut levels = ds.levels;
        Ok(Opened {
            source: Box::new(levels.swap_remove(0)),
            nodata: c.fields[0].values.nodata,
            domain: c.domains.swap_remove(0),
        })
    }
}

async fn warp(w: Warp) -> Result<(), Failure> {
    let existing = w
        .output
        .join(w.array.trim_start_matches('/'))
        .join("zarr.json");
    if existing.exists() && !w.overwrite {
        return Err(format!("{} exists (use --overwrite)", existing.display()).into());
    }
    // Check the CRS before reading anything: the warp must understand it
    // and the GeoZarr writer must be able to record it.
    wxy_warp::check_crs(&w.crs)?;
    if wxy_zarr::write::Proj::from_definition(&w.crs).is_none() {
        return Err(format!(
            "--crs {:?} cannot be recorded as GeoZarr proj: (use EPSG:NNNN)",
            w.crs
        )
        .into());
    }
    let src = open(&w.source, &w.source_array).await?;
    let (e, s) = (&w.extent, &w.size);
    if e[0] >= e[2] || e[1] >= e[3] {
        return Err("--extent must be xmin ymin xmax ymax with min < max".into());
    }
    if s[0] == 0 || s[1] == 0 {
        return Err("--size must be positive".into());
    }
    let dst = GridSpec::from_extent(&w.crs, [e[0], e[2], e[1], e[3]], [s[0], s[1]]);
    let opts = WarpOptions {
        resampling: w.resampling.into(),
        error_threshold: w.error_threshold,
        nodata: w.nodata,
        wrap_globe: !w.no_wrap,
    };
    let out = wxy_warp::warp(&*src.source, &src.domain, &dst, &opts, src.nodata).await?;

    let mut grid = wxy_zarr::GridArray::new(out.window.size.clone(), dst.geotransform);
    grid.crs = Some(w.crs);
    grid.nodata = w.nodata.or(src.nodata);
    wxy_zarr::write_path(&w.output, &w.array, &out.values, &grid).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Warp(w) => warp(w).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wxy: {e}");
            ExitCode::FAILURE
        }
    }
}
