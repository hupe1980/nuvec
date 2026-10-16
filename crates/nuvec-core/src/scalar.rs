//! Weak-typed scalar operands, as in MLX's Python layer (`python/src/utils.cpp`).
//!
//! A Rust number combined with an array adapts to the array: `&f16_array * 2.0` stays
//! float16, `&int8_array + 1` stays int8 (and `+ 300` is an overflow error), `&bool_array + 1`
//! is int32, `&int_array * 0.5` is float32. Every Rust scalar is weak, including typed
//! literals such as `2.0f32`; use [`Array::scalar`] for a strongly typed scalar.

use crate::cast::Canon;
use crate::{Array, Category, Complex64, DType, Error, Result, bf16, f16};

/// A number whose dtype is decided by the array it is combined with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// A boolean.
    Bool(bool),
    /// An integer (exact for every Rust integer type).
    Int(i128),
    /// A real floating-point number.
    Float(f64),
    /// A complex number.
    Complex(Complex64),
}

/// An operand of a binary operation: an array, or a weak scalar.
#[derive(Debug, Clone)]
pub enum ScalarOrArray {
    /// An array; its dtype is kept.
    Array(Array),
    /// A weak scalar; its dtype follows the other operand.
    Scalar(Scalar),
}

impl From<Array> for ScalarOrArray {
    fn from(a: Array) -> Self {
        Self::Array(a)
    }
}

impl From<&Array> for ScalarOrArray {
    fn from(a: &Array) -> Self {
        Self::Array(a.clone())
    }
}

impl From<Scalar> for ScalarOrArray {
    fn from(s: Scalar) -> Self {
        Self::Scalar(s)
    }
}

/// A Rust number usable as a weak operand of `+ - * /` on the right of an array.
///
/// Implemented for `bool`, every integer type, `f16`, `bf16`, `f32`, `f64` and `Complex64`;
/// a single generic operator impl over this trait lets literals such as `2.0` and `1` keep
/// their usual defaults.
pub trait ScalarOperand: Into<Scalar> {}

macro_rules! from_scalar {
    ($($t:ty => |$x:ident| $e:expr),* $(,)?) => {$(
        impl ScalarOperand for $t {}
        impl From<$t> for Scalar {
            fn from($x: $t) -> Self {
                $e
            }
        }
        impl From<$t> for ScalarOrArray {
            fn from(x: $t) -> Self {
                Self::Scalar(Scalar::from(x))
            }
        }
    )*};
}

from_scalar! {
    bool => |x| Scalar::Bool(x),
    i8 => |x| Scalar::Int(x.into()),
    i16 => |x| Scalar::Int(x.into()),
    i32 => |x| Scalar::Int(x.into()),
    i64 => |x| Scalar::Int(x.into()),
    u8 => |x| Scalar::Int(x.into()),
    u16 => |x| Scalar::Int(x.into()),
    u32 => |x| Scalar::Int(x.into()),
    u64 => |x| Scalar::Int(x.into()),
    f16 => |x| Scalar::Float(x.to_f64()),
    bf16 => |x| Scalar::Float(x.to_f64()),
    f32 => |x| Scalar::Float(x.into()),
    f64 => |x| Scalar::Float(x),
    Complex64 => |x| Scalar::Complex(x),
}

/// The range of an integer dtype.
fn int_range(d: DType) -> (i128, i128) {
    match d {
        DType::U8 => (0, u8::MAX.into()),
        DType::U16 => (0, u16::MAX.into()),
        DType::U32 => (0, u32::MAX.into()),
        DType::U64 => (0, u64::MAX.into()),
        DType::I8 => (i8::MIN.into(), i8::MAX.into()),
        DType::I16 => (i16::MIN.into(), i16::MAX.into()),
        DType::I32 => (i32::MIN.into(), i32::MAX.into()),
        _ => (i64::MIN.into(), i64::MAX.into()),
    }
}

impl Scalar {
    /// The dtype of a scalar on its own: bool, int32 (int64, then uint64, when out of
    /// range), float32 or complex64.
    pub fn default_dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Int(v) if i32::try_from(v).is_ok() => DType::I32,
            Self::Int(v) if i64::try_from(v).is_ok() => DType::I64,
            Self::Int(_) => DType::U64,
            Self::Float(_) => DType::F32,
            Self::Complex(_) => DType::Complex64,
        }
    }

    /// The dtype this scalar takes next to an array of dtype `like` (MLX's `to_array`).
    fn dtype_like(self, like: DType) -> Result<DType> {
        Ok(match self {
            Self::Bool(_) => like,
            Self::Int(v) => match like.category() {
                // bool is always promoted.
                Category::Bool => Self::Int(v).default_dtype(),
                Category::Signed | Category::Unsigned => {
                    let (lo, hi) = int_range(like);
                    if !(lo..=hi).contains(&v) {
                        return Err(Error::InvalidArgument(format!(
                            "converting {v} to {like} would overflow"
                        )));
                    }
                    like
                }
                _ => like,
            },
            Self::Float(_) if like.category() == Category::Float => like,
            Self::Float(_) => DType::F32,
            Self::Complex(_) => DType::Complex64,
        })
    }

    fn canon(self) -> Canon {
        match self {
            Self::Bool(b) => Canon::Int(i128::from(b)),
            Self::Int(v) => Canon::Int(v),
            Self::Float(f) => Canon::Float(f),
            Self::Complex(z) => Canon::Complex(z),
        }
    }

    /// A 0-D array holding this scalar, typed relative to `like` (or by default).
    pub(crate) fn to_array(self, like: Option<DType>) -> Result<Array> {
        let dtype = match like {
            Some(like) => self.dtype_like(like)?,
            None => self.default_dtype(),
        };
        crate::ops::filled(&[], dtype, |_| self.canon())
    }
}

impl ScalarOrArray {
    /// This operand as an array: weak scalars take a dtype relative to `like`.
    pub(crate) fn into_array(self, like: Option<DType>) -> Result<Array> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Scalar(s) => s.to_array(like),
        }
    }

    fn dtype(&self) -> Option<DType> {
        match self {
            Self::Array(a) => Some(a.dtype()),
            Self::Scalar(_) => None,
        }
    }
}

/// Both operands as arrays, following MLX's `to_arrays`: arrays keep their dtype, a weak
/// scalar adapts to the other operand, and two scalars take their default dtypes.
/// An input that has already failed reports its error before any scalar conversion.
pub(crate) fn resolve(a: ScalarOrArray, b: ScalarOrArray) -> Result<(Array, Array)> {
    for x in [&a, &b] {
        if let ScalarOrArray::Array(x) = x
            && let Some(err) = x.error()
        {
            return Err(err);
        }
    }
    let (da, db) = (a.dtype(), b.dtype());
    Ok((a.into_array(db)?, b.into_array(da)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn like(s: impl Into<Scalar>, d: DType) -> Result<DType> {
        s.into().dtype_like(d)
    }

    #[test]
    fn mlx_weak_typing_rules() {
        assert_eq!(like(2.0, DType::F16).unwrap(), DType::F16);
        assert_eq!(like(2.0, DType::I32).unwrap(), DType::F32);
        assert_eq!(like(2.0, DType::F64).unwrap(), DType::F64);
        assert_eq!(like(2, DType::I8).unwrap(), DType::I8);
        assert_eq!(like(2, DType::F16).unwrap(), DType::F16);
        assert_eq!(like(2, DType::Bool).unwrap(), DType::I32);
        assert_eq!(like(true, DType::U16).unwrap(), DType::U16);
        assert_eq!(
            like(Complex64::new(1.0, 1.0), DType::F32).unwrap(),
            DType::Complex64
        );
        assert!(matches!(
            like(300, DType::I8),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            like(-1, DType::U32),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(like(u64::MAX, DType::U64).unwrap(), DType::U64);
    }

    #[test]
    fn defaults() {
        assert_eq!(Scalar::from(1).default_dtype(), DType::I32);
        assert_eq!(Scalar::from(1i64 << 40).default_dtype(), DType::I64);
        assert_eq!(Scalar::from(u64::MAX).default_dtype(), DType::U64);
        assert_eq!(Scalar::from(1.0f32).default_dtype(), DType::F32);
        assert_eq!(Scalar::from(false).default_dtype(), DType::Bool);
    }
}
