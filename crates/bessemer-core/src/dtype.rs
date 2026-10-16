//! Element types.

use std::fmt;
use std::ops::{Add, Mul};

use bytemuck::{Pod, Zeroable};
pub use half::{bf16, f16};

/// The element type of an array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum DType {
    /// Boolean, stored as one byte (0 or 1).
    Bool,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// IEEE 754 half precision.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 double precision.
    F64,
    /// Complex number with `f32` real and imaginary parts.
    Complex64,
}

/// The broad kind of an element type, used for type promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// [`DType::Bool`].
    Bool,
    /// Unsigned integers.
    Unsigned,
    /// Signed integers.
    Signed,
    /// Floating point.
    Float,
    /// Complex.
    Complex,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 | Self::Complex64 => 8,
        }
    }

    /// The broad kind of this type.
    pub const fn category(self) -> Category {
        match self {
            Self::Bool => Category::Bool,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => Category::Unsigned,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => Category::Signed,
            Self::F16 | Self::BF16 | Self::F32 | Self::F64 => Category::Float,
            Self::Complex64 => Category::Complex,
        }
    }

    /// The conventional lowercase name, as in MLX and NumPy.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Complex64 => "complex64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A complex number with `f32` parts, the element type of [`DType::Complex64`].
#[derive(Debug, Clone, Copy, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Complex64 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex64 {
    /// Creates a complex number from its parts.
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// A Rust type that can be stored in an array. Implemented for the element types of
/// every [`DType`]; sealed.
pub trait Element: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The array element type this Rust type maps to.
    const DTYPE: DType;
}

pub(crate) mod sealed {
    /// Conversions between host values and array bytes.
    pub trait Sealed: Sized {
        /// Writes `src` into `dst`, which is exactly `src.len() * size` bytes.
        fn write_bytes(src: &[Self], dst: &mut [u8]);
        /// Reads values from `src`, whose length is a multiple of the element size.
        fn read_bytes(src: &[u8]) -> Vec<Self>;
    }
}

macro_rules! pod_element {
    ($($t:ty => $d:ident),* $(,)?) => {$(
        impl sealed::Sealed for $t {
            fn write_bytes(src: &[Self], dst: &mut [u8]) {
                dst.copy_from_slice(bytemuck::cast_slice(src));
            }
            fn read_bytes(src: &[u8]) -> Vec<Self> {
                bytemuck::pod_collect_to_vec(src)
            }
        }
        impl Element for $t {
            const DTYPE: DType = DType::$d;
        }
    )*};
}

pod_element! {
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    f16 => F16, bf16 => BF16, f32 => F32, f64 => F64,
    Complex64 => Complex64,
}

impl sealed::Sealed for bool {
    fn write_bytes(src: &[Self], dst: &mut [u8]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = u8::from(s);
        }
    }
    fn read_bytes(src: &[u8]) -> Vec<Self> {
        src.iter().map(|&b| b != 0).collect()
    }
}

impl Element for bool {
    const DTYPE: DType = DType::Bool;
}

#[cfg(test)]
mod tests {
    use super::sealed::Sealed;
    use super::*;

    #[test]
    fn sizes_match_rust_types() {
        assert_eq!(DType::Bool.size(), 1);
        assert_eq!(DType::F16.size(), size_of::<f16>());
        assert_eq!(DType::BF16.size(), size_of::<bf16>());
        assert_eq!(DType::Complex64.size(), size_of::<Complex64>());
        assert_eq!(DType::I64.size(), size_of::<i64>());
    }

    #[test]
    fn categories() {
        assert_eq!(DType::U32.category(), Category::Unsigned);
        assert_eq!(DType::I8.category(), Category::Signed);
        assert_eq!(DType::BF16.category(), Category::Float);
        assert_eq!(DType::Complex64.category(), Category::Complex);
        assert_eq!(DType::Bool.category(), Category::Bool);
    }

    #[test]
    fn bool_round_trips_through_bytes() {
        let src = [true, false, true];
        let mut bytes = [0u8; 3];
        bool::write_bytes(&src, &mut bytes);
        assert_eq!(bytes, [1, 0, 1]);
        assert_eq!(bool::read_bytes(&bytes), src);
    }

    #[test]
    fn complex_arithmetic() {
        let (a, b) = (Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0));
        assert_eq!(a + b, Complex64::new(4.0, 1.0));
        assert_eq!(a * b, Complex64::new(5.0, 5.0));
    }
}
