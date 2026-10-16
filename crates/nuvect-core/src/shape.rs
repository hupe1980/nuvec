//! Shapes, strides and contiguity.

use smallvec::SmallVec;

use crate::{Error, Result};

/// Dimension sizes. Inline for up to four dimensions.
pub(crate) type Shape = SmallVec<[usize; 4]>;
/// Per-dimension steps, in elements. Inline for up to four dimensions.
pub(crate) type Strides = SmallVec<[isize; 4]>;

/// How an array's elements are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flags {
    /// Elements occupy one dense block, in some dimension order.
    pub contiguous: bool,
    /// Dense, last dimension varies fastest (C order).
    pub row_contiguous: bool,
    /// Dense, first dimension varies fastest (Fortran order).
    pub col_contiguous: bool,
}

/// Element count of `shape`, or an error if it overflows.
pub(crate) fn checked_size(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            Error::InvalidArgument(format!("element count of shape {shape:?} overflows"))
        })
}

/// Byte size of `shape` with elements of `itemsize` bytes, or an error if it overflows.
pub(crate) fn checked_nbytes(shape: &[usize], itemsize: usize) -> Result<usize> {
    checked_size(shape)?
        .checked_mul(itemsize)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| Error::InvalidArgument(format!("byte size of shape {shape:?} overflows")))
}

/// Strides of a dense row-major array. Requires a shape whose size does not overflow.
pub(crate) fn row_contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides: Strides = SmallVec::from_elem(0, shape.len());
    let mut step = 1isize;
    for (s, &d) in strides.iter_mut().zip(shape).rev() {
        *s = step;
        step = step.saturating_mul(d.max(1) as isize);
    }
    strides
}

/// Contiguity flags for a layout. Dimensions of size 1 do not affect contiguity.
pub(crate) fn flags_for(shape: &[usize], strides: &[isize]) -> Flags {
    let dense = |order: &mut dyn Iterator<Item = (&usize, &isize)>| {
        let mut expected = 1isize;
        for (&d, &s) in order {
            if d != 1 && s != expected {
                return false;
            }
            expected = expected.saturating_mul(d as isize);
        }
        true
    };
    let row = dense(&mut shape.iter().zip(strides).rev());
    let col = dense(&mut shape.iter().zip(strides));
    Flags {
        contiguous: row || col,
        row_contiguous: row,
        col_contiguous: col,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides() {
        assert_eq!(row_contiguous_strides(&[2, 3, 4]).as_slice(), &[12, 4, 1]);
        assert!(row_contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn size_and_overflow() {
        assert_eq!(checked_size(&[]).unwrap(), 1);
        assert_eq!(checked_size(&[2, 0, 5]).unwrap(), 0);
        assert!(matches!(
            checked_size(&[usize::MAX, 2]),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn flags() {
        let f = flags_for(&[2, 3], &[3, 1]);
        assert!(f.row_contiguous && f.contiguous && !f.col_contiguous);
        let f = flags_for(&[2, 3], &[1, 2]);
        assert!(f.col_contiguous && !f.row_contiguous);
        let f = flags_for(&[1, 4], &[4, 1]);
        assert!(f.row_contiguous && f.col_contiguous);
        let scalar = flags_for(&[], &[]);
        assert!(scalar.row_contiguous && scalar.col_contiguous);
    }
}
