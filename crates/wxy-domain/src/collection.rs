//! Collections: fields and domains together, with attributes.

use std::collections::BTreeMap;

use crate::domain::Domain;
use crate::field::Field;

/// How two domains in a collection relate.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationKind {
    /// `to` is an overview of `from`, coarser by `factor` along each
    /// mapped dim (not necessarily an integer, as in GDAL).
    Overview {
        factor: Vec<f64>,
    },
    /// `to` holds another stagger location of the same grid (SGRID, the
    /// Arakawa C-grid's u and v points).
    Stagger,
    Other(String),
}

/// A relation from one domain to another, by index.
#[derive(Debug, Clone, PartialEq)]
pub struct Relation {
    pub kind: RelationKind,
    pub from: usize,
    pub to: usize,
}

/// A set of fields and domains with attributes: a classic dataset, a
/// multidimensional group, a mesh with its datasets, a feature layer.
#[derive(Debug, Clone, Default)]
pub struct Collection {
    pub name: String,
    pub domains: Vec<Domain>,
    pub fields: Vec<Field>,
    pub attrs: BTreeMap<String, String>,
    pub children: Vec<Collection>,
    pub related: Vec<Relation>,
}

impl Collection {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn add_domain(&mut self, domain: Domain) -> usize {
        self.domains.push(domain);
        self.domains.len() - 1
    }

    pub fn add_field(&mut self, field: Field) -> usize {
        assert!(
            field.domain < self.domains.len(),
            "field names a missing domain"
        );
        self.fields.push(field);
        self.fields.len() - 1
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn domain_of(&self, field: &Field) -> &Domain {
        &self.domains[field.domain]
    }
}
