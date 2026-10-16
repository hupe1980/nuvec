//! Array operations.

use std::ops::{Add, Mul};

use crate::primitive::Primitive;
use crate::{Array, Result};

/// Element-wise `a + b`. Shapes and dtypes must match (broadcasting and promotion come
/// later). Integers wrap around; for booleans this is logical OR.
pub fn add(a: &Array, b: &Array) -> Result<Array> {
    binary(Primitive::Add, a, b)
}

/// Element-wise `a * b`. Shapes and dtypes must match (broadcasting and promotion come
/// later). Integers wrap around; for booleans this is logical AND.
pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
    binary(Primitive::Multiply, a, b)
}

fn binary(primitive: Primitive, a: &Array, b: &Array) -> Result<Array> {
    let mut outputs = Array::from_op(primitive, vec![a.clone(), b.clone()])?;
    Ok(outputs.swap_remove(0))
}

/// Operators cannot return a `Result`. An invalid expression yields an array whose
/// evaluation (or readback) returns the error, instead of panicking.
fn or_failed(result: Result<Array>, lhs: &Array) -> Array {
    result.unwrap_or_else(|err| Array::failed(err, lhs.dtype(), lhs.shape()))
}

macro_rules! operator {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait<&Array> for &Array {
            type Output = Array;
            fn $method(self, rhs: &Array) -> Array {
                or_failed($op(self, rhs), self)
            }
        }
        impl $trait<Array> for &Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                or_failed($op(self, &rhs), self)
            }
        }
        impl $trait<&Array> for Array {
            type Output = Array;
            fn $method(self, rhs: &Array) -> Array {
                or_failed($op(&self, rhs), &self)
            }
        }
        impl $trait<Array> for Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                or_failed($op(&self, &rhs), &self)
            }
        }
    };
}

operator!(Add, add, add);
operator!(Mul, mul, multiply);
