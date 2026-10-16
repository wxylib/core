//! Affine transforms in GDAL geotransform order.

/// `[x0, dx, rx, y0, ry, dy]`, as GDAL and vaster order them:
///
/// ```text
/// x = x0 + col * dx + row * rx
/// y = y0 + col * ry + row * dy
/// ```
///
/// `(col, row)` are continuous index coordinates (0 is the outer edge of
/// the first cell). Rotation and shear terms (`rx`, `ry`) are allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine(pub [f64; 6]);

impl Affine {
    /// From an extent `[xmin, xmax, ymin, ymax]` and dimension
    /// `[ncol, nrow]`, north up (row 0 at `ymax`).
    pub fn from_extent(extent: [f64; 4], dim: [usize; 2]) -> Self {
        Affine(vaster::extent_dim_to_gt(&extent, &dim))
    }

    /// The extent `[xmin, xmax, ymin, ymax]` covered by `dim` cells, for an
    /// axis-aligned transform.
    pub fn extent(&self, dim: [usize; 2]) -> [f64; 4] {
        vaster::gt_to_extent(&self.0, &dim)
    }

    pub fn forward(&self, col: f64, row: f64) -> (f64, f64) {
        let g = &self.0;
        (g[0] + col * g[1] + row * g[2], g[3] + col * g[4] + row * g[5])
    }

    pub fn inverse(&self) -> Option<Affine> {
        vaster::inv_geotransform(&self.0).map(Affine)
    }

    pub fn is_axis_aligned(&self) -> bool {
        self.0[2] == 0.0 && self.0[4] == 0.0
    }
}
