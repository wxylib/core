//! Domains: index space plus the mappings that place it.

use crate::dim::Dim;
use crate::mapping::Mapping;
use crate::positions::Rung;

/// Where elements are: index space, and any number of mappings.
///
/// A domain keeps every mapping it was given (design doc 02, gate 2
/// default 2). One may be marked primary per dim; [`Domain::primary_for`]
/// returns it, falling back to the first mapping that covers the dim.
#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub dims: Vec<Dim>,
    pub mappings: Vec<Mapping>,
    /// Indices into `mappings` marked primary. At most one per dim.
    pub primary: Vec<usize>,
}

impl Domain {
    pub fn new(dims: Vec<Dim>) -> Self {
        Self {
            dims,
            ..Default::default()
        }
    }

    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().map(|d| d.size).collect()
    }

    pub fn dim_index(&self, name: &str) -> Option<usize> {
        self.dims.iter().position(|d| d.name == name)
    }

    /// Add a mapping and return its index. Panics if it names a dim the
    /// domain does not have.
    pub fn add_mapping(&mut self, mapping: Mapping) -> usize {
        assert!(
            mapping.dims.iter().all(|&d| d < self.dims.len()),
            "mapping covers a dim outside the domain"
        );
        self.mappings.push(mapping);
        self.mappings.len() - 1
    }

    /// Mark a mapping primary for the dims it covers, replacing any other
    /// primary mapping on those dims.
    pub fn set_primary(&mut self, index: usize) {
        let dims = self.mappings[index].dims.clone();
        let mappings = &self.mappings;
        self.primary
            .retain(|&p| !mappings[p].dims.iter().any(|d| dims.contains(d)));
        self.primary.push(index);
    }

    /// Mappings that cover dim `d`.
    pub fn mappings_for(&self, d: usize) -> impl Iterator<Item = (usize, &Mapping)> {
        self.mappings
            .iter()
            .enumerate()
            .filter(move |(_, m)| m.dims.contains(&d))
    }

    /// The primary mapping for dim `d`.
    pub fn primary_for(&self, d: usize) -> Option<&Mapping> {
        self.primary
            .iter()
            .map(|&p| &self.mappings[p])
            .find(|m| m.dims.contains(&d))
            .or_else(|| self.mappings_for(d).map(|(_, m)| m).next())
    }

    /// The rung of the primary mapping for dim `d` (R0 when unmapped).
    pub fn rung_for(&self, d: usize) -> Rung {
        self.primary_for(d).map_or(Rung::IndexOnly, Mapping::rung)
    }
}
