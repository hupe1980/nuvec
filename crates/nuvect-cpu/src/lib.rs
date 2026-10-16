//! Pure-Rust CPU kernels for nuvect.
//!
//! Kernels work on typed slices and know nothing about arrays or graphs; `nuvect-core`
//! dispatches to them. Inputs may have any layout (contiguous, transposed, sliced with
//! negative steps, broadcast with zero strides); outputs are always row-contiguous.
//!
//! **Status:** pre-alpha. Element-wise, reduction and matrix-multiplication kernels.

mod matmul;

pub use matmul::{Gemm, MatView, matmul};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A strided view of elements: element `i` (a multi-index) lives at
/// `data[offset + Σ i_k · strides_k]`.
#[derive(Debug, Clone, Copy)]
pub struct View<'a, T> {
    /// The underlying storage.
    pub data: &'a [T],
    /// Position of the first element, in elements.
    pub offset: isize,
    /// Per-dimension steps, in elements. May be zero (broadcast) or negative.
    pub strides: &'a [isize],
}

/// Applies `f` element-wise: `out[i] = f(a[i])` in row-major order of `shape`.
///
/// # Panics
///
/// If `out.len()` is not the element count of `shape`, or a view indexes outside its data.
/// Both are caller bugs: `nuvect-core` validates shapes and layouts beforehand.
pub fn map1<A: Copy, U>(shape: &[usize], a: View<'_, A>, out: &mut [U], f: impl Fn(A) -> U) {
    for_rows(
        shape,
        [a.strides],
        [a.offset],
        out,
        |row, [ba], [sa]| match lane(a.data, ba, sa, row.len()) {
            Lane::Slice(x) => row.iter_mut().zip(x).for_each(|(o, &x)| *o = f(x)),
            la => row
                .iter_mut()
                .enumerate()
                .for_each(|(j, o)| *o = f(la.get(j))),
        },
    );
}

/// Applies `f` element-wise: `out[i] = f(a[i], b[i])` in row-major order of `shape`.
///
/// # Panics
///
/// As [`map1`].
pub fn map2<A: Copy, B: Copy, U>(
    shape: &[usize],
    a: View<'_, A>,
    b: View<'_, B>,
    out: &mut [U],
    f: impl Fn(A, B) -> U,
) {
    for_rows(
        shape,
        [a.strides, b.strides],
        [a.offset, b.offset],
        out,
        |row, [ba, bb], [sa, sb]| {
            let n = row.len();
            match (lane(a.data, ba, sa, n), lane(b.data, bb, sb, n)) {
                (Lane::Slice(x), Lane::Slice(y)) => {
                    row.iter_mut()
                        .zip(x)
                        .zip(y)
                        .for_each(|((o, &x), &y)| *o = f(x, y));
                }
                (Lane::Slice(x), Lane::Repeat(y)) => {
                    row.iter_mut().zip(x).for_each(|(o, &x)| *o = f(x, y))
                }
                (Lane::Repeat(x), Lane::Slice(y)) => {
                    row.iter_mut().zip(y).for_each(|(o, &y)| *o = f(x, y))
                }
                (la, lb) => row
                    .iter_mut()
                    .enumerate()
                    .for_each(|(j, o)| *o = f(la.get(j), lb.get(j))),
            }
        },
    );
}

/// Applies `f` element-wise: `out[i] = f(a[i], b[i], c[i])` in row-major order of `shape`.
///
/// # Panics
///
/// As [`map1`].
pub fn map3<A: Copy, B: Copy, C: Copy, U>(
    shape: &[usize],
    a: View<'_, A>,
    b: View<'_, B>,
    c: View<'_, C>,
    out: &mut [U],
    f: impl Fn(A, B, C) -> U,
) {
    let strides = [a.strides, b.strides, c.strides];
    for_rows(
        shape,
        strides,
        [a.offset, b.offset, c.offset],
        out,
        |row, [ba, bb, bc], [sa, sb, sc]| {
            let n = row.len();
            match (
                lane(a.data, ba, sa, n),
                lane(b.data, bb, sb, n),
                lane(c.data, bc, sc, n),
            ) {
                (Lane::Slice(x), Lane::Slice(y), Lane::Slice(z)) => {
                    row.iter_mut()
                        .zip(x)
                        .zip(y)
                        .zip(z)
                        .for_each(|(((o, &x), &y), &z)| *o = f(x, y, z));
                }
                (la, lb, lc) => row
                    .iter_mut()
                    .enumerate()
                    .for_each(|(j, o)| *o = f(la.get(j), lb.get(j), lc.get(j))),
            }
        },
    );
}

/// One operand's elements along an output row.
enum Lane<'a, T> {
    /// Contiguous (stride 1).
    Slice(&'a [T]),
    /// Broadcast (stride 0).
    Repeat(T),
    /// Any other stride.
    Strided {
        data: &'a [T],
        start: isize,
        step: isize,
    },
}

impl<T: Copy> Lane<'_, T> {
    fn get(&self, j: usize) -> T {
        match self {
            Self::Slice(x) => x[j],
            Self::Repeat(x) => *x,
            Self::Strided { data, start, step } => data[index(start + j as isize * step)],
        }
    }
}

/// The lane of `n` elements starting at `base` with `step`. Bounds are checked once here
/// for contiguous lanes.
fn lane<T: Copy>(data: &[T], base: isize, step: isize, n: usize) -> Lane<'_, T> {
    match step {
        1 => Lane::Slice(&data[index(base)..index(base) + n]),
        0 => Lane::Repeat(data[index(base)]),
        _ => Lane::Strided {
            data,
            start: base,
            step,
        },
    }
}

/// Calls `row_fn` for each row of the (collapsed) output, in row-major order, with every
/// operand's offset at the row start and its stride along the row.
fn for_rows<const N: usize, U>(
    shape: &[usize],
    strides: [&[isize]; N],
    offsets: [isize; N],
    out: &mut [U],
    mut row_fn: impl FnMut(&mut [U], [isize; N], [isize; N]),
) {
    let size: usize = shape.iter().product();
    assert_eq!(out.len(), size, "output length does not match shape");
    if size == 0 {
        return;
    }
    let (shape, strides) = collapse_dims(shape, strides);
    let ndim = shape.len();
    if ndim == 0 {
        row_fn(out, offsets, [0; N]);
        return;
    }

    let inner = shape[ndim - 1];
    let inner_strides: [isize; N] = std::array::from_fn(|k| strides[k][ndim - 1]);
    let mut position = vec![0usize; ndim - 1];
    let mut base = offsets;
    for row in out.chunks_exact_mut(inner) {
        row_fn(row, base, inner_strides);
        // Advance the odometer over the outer dimensions.
        for d in (0..ndim - 1).rev() {
            position[d] += 1;
            for k in 0..N {
                base[k] += strides[k][d];
            }
            if position[d] < shape[d] {
                break;
            }
            for k in 0..N {
                base[k] -= strides[k][d] * shape[d] as isize;
            }
            position[d] = 0;
        }
    }
}

/// Folds blocks of elements: `out[i]` is the fold of the `reduced` block at output position
/// `i`, starting from `init`. `kept` and `reduced` are `(shape, strides)` pairs that split the
/// input's dimensions; the output is row-major over `kept`.
///
/// Contiguous blocks are passed whole to `block`; other blocks are folded element by element
/// with `f`, which also receives each element's row-major index within its block. Both must
/// compute the same fold.
///
/// # Panics
///
/// As [`map1`].
#[allow(clippy::too_many_arguments)]
pub fn reduce<T: Copy, A: Clone>(
    kept: (&[usize], &[isize]),
    reduced: (&[usize], &[isize]),
    data: &[T],
    offset: isize,
    out: &mut [A],
    init: A,
    f: impl Fn(A, T, usize) -> A,
    block: impl Fn(A, &[T]) -> A,
) {
    let (shape, [strides]) = collapse_dims(reduced.0, [reduced.1]);
    let n: usize = shape.iter().product();
    let contiguous = shape.len() == 1 && strides[0] == 1;
    let fold = |base: isize| -> A {
        let mut acc = init.clone();
        if n == 0 {
            return acc;
        }
        if contiguous || shape.is_empty() {
            return block(acc, &data[index(base)..index(base) + n]);
        }
        let mut position = vec![0usize; shape.len()];
        let mut cursor = base;
        for i in 0..n {
            acc = f(acc, data[index(cursor)], i);
            for d in (0..shape.len()).rev() {
                position[d] += 1;
                cursor += strides[d];
                if position[d] < shape[d] {
                    break;
                }
                cursor -= strides[d] * shape[d] as isize;
                position[d] = 0;
            }
        }
        acc
    };
    for_rows(kept.0, [kept.1], [offset], out, |row, [base], [step]| {
        for (j, o) in row.iter_mut().enumerate() {
            *o = fold(base + j as isize * step);
        }
    });
}

fn index(offset: isize) -> usize {
    usize::try_from(offset).expect("view offset is negative")
}

/// Merges adjacent dimensions that are contiguous for every operand and drops unit
/// dimensions, so that common layouts iterate as one flat loop.
///
/// Dimension `d` merges into `d + 1` when `strides[d] == strides[d + 1] * shape[d + 1]`
/// holds for every operand.
pub fn collapse_dims<const N: usize>(
    shape: &[usize],
    strides: [&[isize]; N],
) -> (Vec<usize>, [Vec<isize>; N]) {
    let mut out_shape: Vec<usize> = Vec::with_capacity(shape.len());
    let mut out_strides: [Vec<isize>; N] = std::array::from_fn(|_| Vec::with_capacity(shape.len()));
    for d in 0..shape.len() {
        if shape[d] == 1 {
            continue;
        }
        let mergeable = out_shape.last().is_some() && {
            let last = out_strides[0].len() - 1;
            (0..N).all(|k| out_strides[k][last] == strides[k][d] * shape[d] as isize)
        };
        if mergeable {
            let last = out_shape.len() - 1;
            out_shape[last] *= shape[d];
            for k in 0..N {
                out_strides[k][last] = strides[k][d];
            }
        } else {
            out_shape.push(shape[d]);
            for k in 0..N {
                out_strides[k].push(strides[k][d]);
            }
        }
    }
    (out_shape, out_strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn contiguous_binary() {
        let mut out = [0i32; 6];
        let s = [3, 1];
        let a = View {
            data: &[1, 2, 3, 4, 5, 6],
            offset: 0,
            strides: &s,
        };
        let b = View {
            data: &[10, 20, 30, 40, 50, 60],
            offset: 0,
            strides: &s,
        };
        map2(&[2, 3], a, b, &mut out, |x, y| x + y);
        assert_eq!(out, [11, 22, 33, 44, 55, 66]);
    }

    #[test]
    fn transposed_and_broadcast() {
        // a = [[1,2,3],[4,5,6]] read transposed (3x2); b = [10, 20] broadcast over rows.
        let mut out = [0i32; 6];
        let a = View {
            data: &[1, 2, 3, 4, 5, 6],
            offset: 0,
            strides: &[1, 3],
        };
        let b = View {
            data: &[10, 20],
            offset: 0,
            strides: &[0, 1],
        };
        map2(&[3, 2], a, b, &mut out, |x, y| x + y);
        assert_eq!(out, [11, 24, 12, 25, 13, 26]);
    }

    #[test]
    fn negative_strides_and_offsets() {
        // Reverse of [0, 1, 2, 3, 4], every second element: [4, 2, 0].
        let mut out = [0u8; 3];
        let a = View {
            data: &[0, 1, 2, 3, 4],
            offset: 4,
            strides: &[-2],
        };
        map1(&[3], a, &mut out, |x| x);
        assert_eq!(out, [4, 2, 0]);
    }

    #[test]
    fn scalar_and_empty() {
        let mut out = [0f32; 1];
        map1(
            &[],
            View {
                data: &[2.5],
                offset: 0,
                strides: &[],
            },
            &mut out,
            |x| x * 2.0,
        );
        assert_eq!(out, [5.0]);
        let mut none: [f32; 0] = [];
        map1(
            &[3, 0],
            View {
                data: &[],
                offset: 0,
                strides: &[0, 1],
            },
            &mut none,
            |x: f32| x,
        );
    }

    #[test]
    fn ternary() {
        let mut out = [0i32; 3];
        let s = [1];
        map3(
            &[3],
            View {
                data: &[1u8, 0, 1],
                offset: 0,
                strides: &s,
            },
            View {
                data: &[1, 2, 3],
                offset: 0,
                strides: &s,
            },
            View {
                data: &[-1],
                offset: 0,
                strides: &[0],
            },
            &mut out,
            |c, x, y| if c != 0 { x } else { y },
        );
        assert_eq!(out, [1, -1, 3]);
    }

    #[test]
    fn reduce_over_middle_axis_of_transposed_layout() {
        // Logical a[i][j][k] = data[k*6 + j*2 + i] for shape [2, 3, 4] (fully transposed).
        let data: Vec<i32> = (0..24).collect();
        let (shape, strides) = ([2usize, 3, 4], [1isize, 2, 6]);
        let mut out = [0i32; 8];
        // Keep axes 0 and 2, reduce axis 1.
        reduce(
            (&[shape[0], shape[2]], &[strides[0], strides[2]]),
            (&[shape[1]], &[strides[1]]),
            &data,
            0,
            &mut out,
            0,
            |acc, x, _| acc + x,
            |acc, xs| xs.iter().fold(acc, |a, &x| a + x),
        );
        let expected: Vec<i32> = (0..2)
            .flat_map(|i| (0..4).map(move |k| (0..3).map(|j| k * 6 + j * 2 + i).sum()))
            .collect();
        assert_eq!(out.to_vec(), expected);
    }

    #[test]
    fn reduce_passes_block_index_and_handles_empty() {
        let pick = |acc: (i32, usize), x: i32, i: usize| if x > acc.0 { (x, i) } else { acc };
        let block = |acc, xs: &[i32]| xs.iter().enumerate().fold(acc, |a, (i, &x)| pick(a, x, i));
        // Contiguous block (passed whole) and a strided one (element by element).
        let mut out = [(0, 0usize); 2];
        reduce(
            (&[2], &[4]),
            (&[4], &[1]),
            &[5, 9, 1, 7, 3, 2, 8, 0],
            0,
            &mut out,
            (i32::MIN, 0),
            pick,
            block,
        );
        assert_eq!(out, [(9, 1), (8, 2)]);
        reduce(
            (&[2], &[1]),
            (&[4], &[2]),
            &[5, 9, 1, 7, 3, 2, 8, 0],
            0,
            &mut out,
            (i32::MIN, 0),
            pick,
            block,
        );
        assert_eq!(out, [(8, 3), (9, 0)]);
        let mut sums = [7i32; 3];
        let add = |a: i32, x: i32, _| a + x;
        reduce(
            (&[3], &[1]),
            (&[0], &[1]),
            &[0i32; 0],
            0,
            &mut sums,
            0,
            add,
            |a, xs| xs.iter().fold(a, |a, &x| a + x),
        );
        assert_eq!(sums, [0, 0, 0]);
    }

    #[test]
    fn collapse_merges_contiguous_dims() {
        let (shape, [s]) = collapse_dims(&[2, 3, 4], [&[12, 4, 1]]);
        assert_eq!((shape, s), (vec![24], vec![1]));
        let (shape, [s]) = collapse_dims(&[2, 1, 3], [&[3, 99, 1]]);
        assert_eq!((shape, s), (vec![6], vec![1]));
        let (shape, [a, b]) = collapse_dims(&[2, 3], [&[3, 1], &[0, 1]]);
        assert_eq!((shape, a, b), (vec![2, 3], vec![3, 1], vec![0, 1]));
    }
}
