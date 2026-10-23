//! Creating arrays from host data: the [`array!`](crate::array!) macro, `From`/`FromIterator`
//! conversions and [`Array::from_slice`].

use std::panic::Location;

use crate::axes::{self, IntoShape};
use crate::buffer::Buffer;
use crate::shape;
use crate::{Array, Complex64, DType, Element, Error, Result, bf16, f16};

/// An array of `data` in `shape` (row-major).
pub(crate) fn from_data<T: Element>(data: &[T], shape: &[usize]) -> Result<Array> {
    let expected = shape::checked_size(shape)?;
    if data.len() != expected {
        return Err(Error::size_mismatch("from_slice", expected, data.len()));
    }
    let mut buffer = Buffer::try_alloc(shape::checked_nbytes(shape, T::DTYPE.size())?)?;
    T::write_bytes(data, buffer.bytes_mut());
    Array::from_buffer(buffer, T::DTYPE, shape)
}

impl Array {
    /// Runs the constructor `f` of the public operation `op` on the requested shape
    /// (`spec`, validated), returning its array; on error, a failed array of `dtype` with
    /// the requested shape if it is valid, else 0-D.
    pub(crate) fn constructor(
        op: &'static str,
        at: &'static Location<'static>,
        dtype: DType,
        spec: &[i64],
        f: impl FnOnce(&[usize]) -> Result<Array>,
    ) -> Array {
        match axes::dims(spec) {
            Ok(shape) => {
                f(&shape).unwrap_or_else(|e| Array::failed(e.locate(op, at), dtype, &shape))
            }
            Err(e) => Array::failed(e.locate(op, at), dtype, &[]),
        }
    }

    /// An array holding `data` (row-major) in `shape`, of `T`'s dtype.
    ///
    /// A length that does not match the shape gives a failed array
    /// ([`crate::OpErrorKind::SizeMismatch`]) with the requested shape.
    ///
    /// ```
    /// use nuvec_core::{Array, DType};
    ///
    /// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    /// assert_eq!((a.shape(), a.dtype()), (&[2, 3][..], DType::F32));
    /// assert!(Array::from_slice(&[1u8, 2], [3]).is_failed());
    /// ```
    #[track_caller]
    pub fn from_slice<T: Element>(data: &[T], shape: impl IntoShape) -> Array {
        let spec = shape.shape_spec();
        Self::constructor("from_slice", Location::caller(), T::DTYPE, &spec, |shape| {
            from_data(data, shape)
        })
    }
}

/// A 0-D array holding `value`, of `T`'s dtype (a *strong* scalar: unlike a Rust number used
/// as an operand, it does not adapt to the other operand's dtype).
impl<T: Element> From<T> for Array {
    #[track_caller]
    fn from(value: T) -> Self {
        Array::from_slice(&[value], ())
    }
}

/// A 1-D array of the elements, of `T`'s dtype.
impl<T: Element, const N: usize> From<[T; N]> for Array {
    #[track_caller]
    fn from(values: [T; N]) -> Self {
        Array::from_slice(&values, [N])
    }
}

/// A 1-D array of the elements, of `T`'s dtype.
impl<T: Element> From<Vec<T>> for Array {
    #[track_caller]
    fn from(values: Vec<T>) -> Self {
        Array::from_slice(&values, [values.len()])
    }
}

/// A 1-D array of the elements, of `T`'s dtype.
impl<T: Element> From<&[T]> for Array {
    #[track_caller]
    fn from(values: &[T]) -> Self {
        Array::from_slice(values, [values.len()])
    }
}

/// A 1-D array of the elements, of `T`'s dtype.
///
/// ```
/// use nuvec_core::{Array, DType};
///
/// let a: Array = (0..4).map(|i| i as f32 * 0.5).collect();
/// assert_eq!((a.shape(), a.dtype()), (&[4][..], DType::F32));
/// ```
impl<T: Element> FromIterator<T> for Array {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values: Vec<T> = iter.into_iter().collect();
        Array::from_slice(&values, [values.len()])
    }
}

/// An element type of an [`array!`](crate::array!) literal: Rust's default float literal
/// type `f64` becomes float32 (MLX's default); every other element type keeps its
/// dtype.
#[doc(hidden)]
pub trait Literal: Copy {
    fn into_array(data: &[Self], shape: &[usize], at: &'static Location<'static>) -> Array;
}

macro_rules! literal {
    ($($t:ty),*) => {$(
        impl Literal for $t {
            fn into_array(data: &[Self], shape: &[usize], at: &'static Location<'static>) -> Array {
                from_data(data, shape).unwrap_or_else(|e| {
                    Array::failed(e.locate("array", at), <$t as Element>::DTYPE, shape)
                })
            }
        }
    )*};
}

literal!(
    bool, u8, u16, u32, u64, i8, i16, i32, i64, f16, bf16, f32, Complex64
);

impl Literal for f64 {
    fn into_array(data: &[Self], shape: &[usize], at: &'static Location<'static>) -> Array {
        let data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        f32::into_array(&data, shape, at)
    }
}

/// Support of the [`array!`](crate::array!) macro.
#[doc(hidden)]
pub mod __private {
    use std::panic::Location;

    use super::Literal;
    use crate::Array;

    #[track_caller]
    pub fn from_nested1<T: Literal, const A: usize>(x: [T; A]) -> Array {
        T::into_array(&x, &[A], Location::caller())
    }

    #[track_caller]
    pub fn from_nested2<T: Literal, const A: usize, const B: usize>(x: [[T; B]; A]) -> Array {
        let flat: Vec<T> = x.iter().flatten().copied().collect();
        T::into_array(&flat, &[A, B], Location::caller())
    }

    #[track_caller]
    pub fn from_nested3<T: Literal, const A: usize, const B: usize, const C: usize>(
        x: [[[T; C]; B]; A],
    ) -> Array {
        let flat: Vec<T> = x.iter().flatten().flatten().copied().collect();
        T::into_array(&flat, &[A, B, C], Location::caller())
    }

    #[track_caller]
    pub fn from_nested4<
        T: Literal,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
    >(
        x: [[[[T; D]; C]; B]; A],
    ) -> Array {
        let flat: Vec<T> = x.iter().flatten().flatten().flatten().copied().collect();
        T::into_array(&flat, &[A, B, C, D], Location::caller())
    }
}

/// An array from nested literals, as MLX's `mx.array([...])`: the macro's brackets are the
/// outermost dimension, so `array![1, 2, 3]` is 1-D and `array![[1.0, 2.0], [3.0, 4.0]]` is
/// 2-D (up to 4-D). Unsuffixed float literals give float32 and unsuffixed integers int32;
/// suffixed literals keep their type (`array![1u8, 2, 3]` is uint8), except `f64`, which
/// gives float32 like MLX's Python floats (use [`Array::from`] for a float64 array).
/// Ragged rows do not compile.
///
/// A 0-D array is [`Array::from`] a number: a macro cannot tell `array!(2.5)` from
/// `array![2.5]` (the invocation's delimiters are not part of its input), and the latter
/// is the 1-element vector, as with `vec!`.
///
/// ```
/// use nuvec_core::{Array, DType, array};
///
/// let m = array![[1.0, 2.0], [3.0, 4.0]];
/// assert_eq!((m.shape(), m.dtype()), (&[2, 2][..], DType::F32));
/// assert_eq!(array![1, 2, 3].dtype(), DType::I32);
/// assert_eq!(array![1u8, 2, 3].dtype(), DType::U8);
/// assert_eq!(array![2.5].shape(), [1]);
/// assert_eq!(Array::from(2.5f32).ndim(), 0);
/// ```
#[macro_export]
macro_rules! array {
    ($([$([$([$($x:expr),* $(,)?]),* $(,)?]),* $(,)?]),+ $(,)?) => {
        $crate::__private::from_nested4([$([$([$([$($x),*]),*]),*]),+])
    };
    ($([$([$($x:expr),* $(,)?]),* $(,)?]),+ $(,)?) => {
        $crate::__private::from_nested3([$([$([$($x),*]),*]),+])
    };
    ($([$($x:expr),* $(,)?]),+ $(,)?) => {
        $crate::__private::from_nested2([$([$($x),*]),+])
    };
    ($($x:expr),+ $(,)?) => {
        $crate::__private::from_nested1([$($x),+])
    };
}

#[cfg(test)]
mod tests {
    use crate::{Array, DType};

    #[test]
    fn literals() {
        let a = crate::array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        assert_eq!((a.shape(), a.dtype()), (&[2, 2, 2][..], DType::I32));
        let b = crate::array![[[[1.5f64]]]];
        assert_eq!((b.shape(), b.dtype()), (&[1, 1, 1, 1][..], DType::F32));
        assert_eq!(crate::array![true, false].dtype(), DType::Bool);
        assert_eq!(Array::from(vec![1i64; 6]).shape(), [6]);
        assert_eq!(Array::from(3.0f64).dtype(), DType::F64);
    }
}
