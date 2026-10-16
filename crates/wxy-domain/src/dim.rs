//! Dimensions: named axes of index space.

/// A named axis of index space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dim {
    pub name: String,
    pub size: u64,
}

impl Dim {
    pub fn new(name: impl Into<String>, size: u64) -> Self {
        Self { name: name.into(), size }
    }
}
