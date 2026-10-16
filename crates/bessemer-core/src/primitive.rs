//! What a graph node computes.
//!
//! Primitives form a closed enum: equality and hashing (the basis for common-subexpression
//! elimination) are derived, and every per-primitive rule — shape inference here; gradient,
//! batching and fusion rules later — is an exhaustive `match`.

use smallvec::{SmallVec, smallvec};

use crate::shape::Shape;
use crate::{Array, DType, Error, Result};

/// An operation recorded in the graph, with its parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Primitive {
    /// Element-wise sum.
    Add,
    /// Element-wise product.
    Multiply,
    /// Test-only two-output primitive: `(a + b, a * b)`.
    #[cfg(test)]
    AddMultiply,
}

/// The dtype and shape of one output of a primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputSpec {
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
}

impl Primitive {
    /// The operation's name, as used in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Multiply => "multiply",
            #[cfg(test)]
            Self::AddMultiply => "add_multiply",
        }
    }

    /// Validates the inputs and infers the outputs, without computing anything.
    pub(crate) fn output_specs(&self, inputs: &[Array]) -> Result<SmallVec<[OutputSpec; 1]>> {
        match self {
            Self::Add | Self::Multiply => Ok(smallvec![self.same_shape_and_dtype(inputs)?]),
            #[cfg(test)]
            Self::AddMultiply => {
                let spec = self.same_shape_and_dtype(inputs)?;
                Ok(smallvec![spec.clone(), spec])
            }
        }
    }

    fn same_shape_and_dtype(&self, inputs: &[Array]) -> Result<OutputSpec> {
        let [a, b] = inputs else {
            return Err(Error::InvalidArgument(format!(
                "{} takes 2 inputs, got {}",
                self.name(),
                inputs.len()
            )));
        };
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                op: self.name(),
                lhs: a.shape().to_vec(),
                rhs: b.shape().to_vec(),
            });
        }
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                op: self.name(),
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        Ok(OutputSpec {
            dtype: a.dtype(),
            shape: a.shape().into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn hash(p: &Primitive) -> u64 {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    }

    #[test]
    fn equal_primitives_compare_and_hash_equal() {
        assert_eq!(Primitive::Add, Primitive::Add);
        assert_eq!(hash(&Primitive::Add), hash(&Primitive::Add));
        assert_ne!(Primitive::Add, Primitive::Multiply);
    }
}
