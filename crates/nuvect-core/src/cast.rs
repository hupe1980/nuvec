//! Storage types and element conversions.
//!
//! Every dtype maps to a `Pod` storage type ([`with_type!`]). Conversions go through an
//! exact canonical value, so each element is rounded at most once.

use bytemuck::{Pod, Zeroable};

use crate::{Complex64, bf16, f16};

/// Storage type of [`crate::DType::Bool`]: one byte, 0 or 1. A distinct type from `u8`, so
/// conversions *to* bool test for non-zero instead of truncating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
#[repr(transparent)]
pub(crate) struct B8(pub(crate) u8);

/// An exact intermediate value: every integer fits `i128`, every float fits `f64`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Canon {
    Int(i128),
    Float(f64),
    Complex(Complex64),
}

/// A storage type with conversions to and from [`Canon`].
pub(crate) trait Scalar: Pod {
    fn to_canon(self) -> Canon;
    fn from_canon(value: Canon) -> Self;
}

macro_rules! int_scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            fn to_canon(self) -> Canon {
                Canon::Int(self as i128)
            }
            fn from_canon(value: Canon) -> Self {
                match value {
                    Canon::Int(i) => i as $t,      // wraps, like C and NumPy
                    Canon::Float(f) => f as $t,    // saturates; NaN becomes 0
                    Canon::Complex(z) => z.re as $t,
                }
            }
        }
    )*};
}
int_scalar!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! float_scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            fn to_canon(self) -> Canon {
                Canon::Float(self as f64)
            }
            fn from_canon(value: Canon) -> Self {
                match value {
                    Canon::Int(i) => i as $t,
                    Canon::Float(f) => f as $t,
                    Canon::Complex(z) => z.re as $t,
                }
            }
        }
    )*};
}
float_scalar!(f32, f64);

macro_rules! half_scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            fn to_canon(self) -> Canon {
                Canon::Float(self.to_f64())
            }
            fn from_canon(value: Canon) -> Self {
                match value {
                    Canon::Int(i) => <$t>::from_f64(i as f64),
                    Canon::Float(f) => <$t>::from_f64(f),
                    Canon::Complex(z) => <$t>::from_f32(z.re),
                }
            }
        }
    )*};
}
half_scalar!(f16, bf16);

impl Scalar for Complex64 {
    fn to_canon(self) -> Canon {
        Canon::Complex(self)
    }
    fn from_canon(value: Canon) -> Self {
        match value {
            Canon::Int(i) => Complex64::new(i as f32, 0.0),
            Canon::Float(f) => Complex64::new(f as f32, 0.0),
            Canon::Complex(z) => z,
        }
    }
}

impl Scalar for B8 {
    fn to_canon(self) -> Canon {
        Canon::Int(i128::from(self.0 != 0))
    }
    fn from_canon(value: Canon) -> Self {
        B8(u8::from(match value {
            Canon::Int(i) => i != 0,
            Canon::Float(f) => f != 0.0, // NaN is true, as in NumPy
            Canon::Complex(z) => z.re != 0.0 || z.im != 0.0,
        }))
    }
}

/// Converts one element.
pub(crate) fn convert<S: Scalar, D: Scalar>(x: S) -> D {
    D::from_canon(x.to_canon())
}

/// Runs `$body` with `$T` bound to the storage type of `$dtype`.
macro_rules! with_type {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::DType::Bool => {
                type $T = $crate::cast::B8;
                $body
            }
            $crate::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::DType::F16 => {
                type $T = $crate::f16;
                $body
            }
            $crate::DType::BF16 => {
                type $T = $crate::bf16;
                $body
            }
            $crate::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::DType::Complex64 => {
                type $T = $crate::Complex64;
                $body
            }
        }
    };
}
pub(crate) use with_type;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_int_saturates_and_nan_is_zero() {
        assert_eq!(convert::<f32, u8>(300.7), 255);
        assert_eq!(convert::<f64, i8>(-1e9), -128);
        assert_eq!(convert::<f32, i32>(f32::NAN), 0);
        assert_eq!(convert::<f32, i32>(-2.9), -2);
    }

    #[test]
    fn int_to_int_wraps() {
        assert_eq!(convert::<i32, u8>(257), 1);
        assert_eq!(convert::<u64, i64>(u64::MAX), -1);
    }

    #[test]
    fn bool_semantics() {
        assert_eq!(convert::<u8, B8>(2), B8(1));
        assert_eq!(convert::<f32, B8>(f32::NAN), B8(1));
        assert_eq!(convert::<f32, B8>(-0.0), B8(0));
        assert_eq!(convert::<B8, f32>(B8(1)), 1.0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
    fn complex_and_half() {
        assert_eq!(convert::<Complex64, f32>(Complex64::new(1.5, 9.0)), 1.5);
        assert_eq!(convert::<i32, Complex64>(3), Complex64::new(3.0, 0.0));
        assert_eq!(convert::<f64, f16>(1.0 + 1.0 / 4096.0), f16::from_f32(1.0));
        assert_eq!(convert::<bf16, f32>(bf16::from_f32(0.5)), 0.5);
    }
}
