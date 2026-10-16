//! Mappings: positions for some of a domain's dims, in a named frame.

use crate::frame::Frame;
use crate::positions::{Positions, Rung};

/// Where stored coordinates sit relative to the elements they describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellConvention {
    /// Coordinates are element centres (CF default, GeoTIFF PixelIsPoint
    /// read as centres). `n` values per axis.
    Centre,
    /// Coordinates are element edges. `n + 1` values per axis.
    Edge,
}

/// Positions for a subset of a domain's dims, and the frame they are in.
#[derive(Debug, Clone)]
pub struct Mapping {
    /// Indices into the domain's dims, in the order `positions` expects.
    /// For [`Positions::Affine`] this is `[row_dim, col_dim]`.
    pub dims: Vec<usize>,
    pub positions: Positions,
    pub frame: Frame,
    pub cell: CellConvention,
    /// Where this mapping came from: `"stored"` for what a reader found, or
    /// the name of the step that derived it (such as `"normalise"`).
    pub provenance: String,
}

impl Mapping {
    pub fn new(dims: Vec<usize>, positions: Positions, frame: Frame) -> Self {
        Self {
            dims,
            positions,
            frame,
            cell: CellConvention::Centre,
            provenance: "stored".into(),
        }
    }

    pub fn with_cell(mut self, cell: CellConvention) -> Self {
        self.cell = cell;
        self
    }

    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = provenance.into();
        self
    }

    pub fn rung(&self) -> Rung {
        self.positions.rung()
    }

    /// The position at a continuous index, one value per covered dim in
    /// `self.dims` order (see the crate docs for the index convention).
    /// Returns one value per output coordinate, or `None` where the mapping
    /// cannot place the index (R0, outside a curvilinear grid, or a rung
    /// not evaluated in S1).
    pub fn position(&self, index: &[f64]) -> Option<Vec<f64>> {
        match &self.positions {
            Positions::IndexOnly => None,
            Positions::Affine(a) => {
                let (x, y) = a.forward(index[1], index[0]);
                Some(vec![x, y])
            }
            Positions::Regular1 { origin, step } => Some(vec![origin + index[0] * step]),
            Positions::Rectilinear(axes) => axes
                .iter()
                .zip(index)
                .map(|(c, &i)| interp_axis(c, i, self.cell))
                .collect(),
            Positions::Curvilinear(arrays) => {
                let shape = &arrays.first()?.shape;
                arrays
                    .iter()
                    .map(|a| interp_nd(&a.values, shape, index))
                    .collect()
            }
            Positions::Fitted(m) => m.forward(index),
            Positions::Scheme(_) | Positions::Explicit(_) => None,
        }
    }
}

/// Linear interpolation along one coordinate vector at continuous index
/// `i`, extrapolating from the end steps.
fn interp_axis(c: &[f64], i: f64, cell: CellConvention) -> Option<f64> {
    if c.is_empty() || !i.is_finite() {
        return None;
    }
    // Position of index `i` in "sample space", where sample k is c[k].
    let s = match cell {
        CellConvention::Centre => i - 0.5,
        CellConvention::Edge => i,
    };
    if c.len() == 1 {
        return Some(c[0]);
    }
    let k = (s.floor() as isize).clamp(0, c.len() as isize - 2) as usize;
    let t = s - k as f64;
    Some(c[k] + t * (c[k + 1] - c[k]))
}

/// Multilinear interpolation in a centre-registered array at a continuous
/// index. Returns `None` outside the array's sample hull.
fn interp_nd(values: &[f64], shape: &[usize], index: &[f64]) -> Option<f64> {
    let nd = shape.len();
    if index.len() != nd {
        return None;
    }
    let mut base = vec![0usize; nd];
    let mut frac = vec![0f64; nd];
    for d in 0..nd {
        let s = index[d] - 0.5;
        if !(s >= 0.0 && s <= (shape[d] - 1) as f64) {
            return None;
        }
        let k = (s.floor() as usize).min(shape[d].saturating_sub(2));
        base[d] = k;
        frac[d] = s - k as f64;
    }
    let mut strides = vec![1usize; nd];
    for d in (0..nd.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    let mut acc = 0.0;
    for corner in 0..(1usize << nd) {
        let mut w = 1.0;
        let mut off = 0;
        for d in 0..nd {
            let up = (corner >> d) & 1 == 1;
            let k = if up && shape[d] > 1 { base[d] + 1 } else { base[d] };
            w *= if up { frac[d] } else { 1.0 - frac[d] };
            off += k * strides[d];
        }
        if w != 0.0 {
            acc += w * values[off];
        }
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;
    use crate::positions::CoordArray;

    #[test]
    fn affine_centre_of_first_cell() {
        let m = Mapping::new(
            vec![0, 1],
            Positions::Affine(Affine::from_extent([0.0, 360.0, -90.0, 90.0], [360, 180])),
            Frame::crs("EPSG:4326"),
        );
        assert_eq!(m.position(&[0.5, 0.5]).unwrap(), vec![0.5, 89.5]);
    }

    #[test]
    fn rectilinear_centres_interpolate_and_extrapolate() {
        let m = Mapping::new(
            vec![0],
            Positions::Rectilinear(vec![vec![0.5, 1.5, 3.5]]),
            Frame::Index,
        );
        assert_eq!(m.position(&[0.5]).unwrap(), vec![0.5]);
        assert_eq!(m.position(&[2.5]).unwrap(), vec![3.5]);
        // outer edge of the first element extrapolates by the first step
        assert_eq!(m.position(&[0.0]).unwrap(), vec![0.0]);
    }

    #[test]
    fn curvilinear_bilinear() {
        // 2 x 2 centres, x = col, y = 10 * row
        let x = CoordArray::new(vec![2, 2], vec![0.0, 1.0, 0.0, 1.0]);
        let y = CoordArray::new(vec![2, 2], vec![0.0, 0.0, 10.0, 10.0]);
        let m = Mapping::new(vec![0, 1], Positions::Curvilinear(vec![x, y]), Frame::Index);
        assert_eq!(m.position(&[1.0, 1.0]).unwrap(), vec![0.5, 5.0]);
        assert!(m.position(&[0.0, 0.0]).is_none());
    }
}
