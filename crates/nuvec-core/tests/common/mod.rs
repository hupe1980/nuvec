//! Reference oracle for differential tests.
//!
//! [`Ref`] holds row-major host data and implements every view through *logical index
//! formulas* (permuted indices, Python `range` enumeration, modular broadcasting), never
//! through strides, so it is independent of the implementation under test.

#![allow(dead_code, unreachable_pub)]

use nuvec_core::{Array, Device, Element, SliceArg, Stream, set_default_stream, slice};

/// Deterministic xorshift64* generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn range(&mut self, lo: isize, hi: isize) -> isize {
        lo + self.below((hi - lo + 1) as usize) as isize
    }

    pub fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// Uniform in [-10, 10).
    pub fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0
    }

    /// A shape of 0–4 dimensions with sizes 1–5.
    pub fn shape(&mut self) -> Vec<usize> {
        (0..self.below(5)).map(|_| 1 + self.below(5)).collect()
    }

    /// `shape` with some dimensions set to 1 and some leading dimensions dropped: a shape
    /// that broadcasts to `shape`.
    pub fn broadcastable(&mut self, shape: &[usize]) -> Vec<usize> {
        let drop = self.below(shape.len() + 1);
        shape[drop..]
            .iter()
            .map(|&d| if self.below(3) == 0 { 1 } else { d })
            .collect()
    }
}

/// Records this thread's operations on the CPU: for tests of CPU semantics and accuracy,
/// and oracle sweeps that run thousands of tiny evaluations (GPU coverage lives in
/// `tests/gpu.rs` and in each suite's GPU tests).
pub fn on_cpu() {
    set_default_stream(Stream::default_for(Device::cpu()));
}

pub fn size(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = i % shape[d];
        i /= shape[d];
    }
    idx
}

fn ravel(idx: &[usize], shape: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (&i, &d)| acc * d + i)
}

/// Indices selected by Python's `range` over a slice of a dimension of size `n`.
pub fn slice_indices(n: usize, arg: SliceArg) -> Vec<usize> {
    let n = n as isize;
    let wrap = |i: isize| if i < 0 { i + n } else { i };
    let mut out = Vec::new();
    if arg.step > 0 {
        let mut i = arg.start.map_or(0, wrap).clamp(0, n);
        let stop = arg.stop.map_or(n, wrap).clamp(0, n);
        while i < stop {
            out.push(i as usize);
            i += arg.step;
        }
    } else {
        let mut i = arg.start.map_or(n - 1, wrap).clamp(-1, n - 1);
        let stop = arg.stop.map_or(-1, wrap).clamp(-1, n - 1);
        while i > stop {
            out.push(i as usize);
            i += arg.step;
        }
    }
    out
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() < n {
            1
        } else {
            s[i + s.len() - n]
        }
    };
    (0..n)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => Some(x),
            (1, y) => Some(y),
            (x, 1) => Some(x),
            _ => None,
        })
        .collect()
}

/// Row-major host data with a shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Ref<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Copy> Ref<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert_eq!(size(&shape), data.len());
        Self { shape, data }
    }

    fn build(shape: Vec<usize>, f: impl Fn(&[usize]) -> T) -> Self {
        let data = (0..size(&shape)).map(|i| f(&unravel(i, &shape))).collect();
        Self { shape, data }
    }

    fn at(&self, idx: &[usize]) -> T {
        self.data[ravel(idx, &self.shape)]
    }

    pub fn transpose(&self, axes: &[usize]) -> Self {
        let shape = axes.iter().map(|&a| self.shape[a]).collect();
        Self::build(shape, |out| {
            let mut idx = vec![0; axes.len()];
            for (o, &a) in axes.iter().enumerate() {
                idx[a] = out[o];
            }
            self.at(&idx)
        })
    }

    pub fn slice(&self, args: &[SliceArg]) -> Self {
        let picks: Vec<Vec<usize>> = self
            .shape
            .iter()
            .zip(args)
            .map(|(&n, &a)| slice_indices(n, a))
            .collect();
        let shape = picks.iter().map(Vec::len).collect();
        Self::build(shape, |out| {
            let idx: Vec<usize> = out.iter().zip(&picks).map(|(&o, p)| p[o]).collect();
            self.at(&idx)
        })
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let lead = shape.len() - self.shape.len();
        Self::build(shape.to_vec(), |out| {
            let idx: Vec<usize> = self
                .shape
                .iter()
                .enumerate()
                .map(|(d, &n)| if n == 1 { 0 } else { out[d + lead] })
                .collect();
            self.at(&idx)
        })
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        Self::new(shape, self.data.clone())
    }

    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Ref<U> {
        Ref {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Folds the elements over `axes` (sorted, unique) in row-major order. `f` receives each
    /// element's index within its block. Reduced dimensions become 1 with `keepdims`.
    pub fn reduce<A: Copy>(
        &self,
        axes: &[usize],
        keepdims: bool,
        init: A,
        f: impl Fn(A, T, usize) -> A,
    ) -> Ref<A> {
        let kept: Vec<usize> = (0..self.shape.len())
            .filter(|d| !axes.contains(d))
            .collect();
        let out_shape: Vec<usize> = kept.iter().map(|&d| self.shape[d]).collect();
        let mut acc = vec![init; size(&out_shape)];
        let mut seen = vec![0usize; acc.len()];
        for (i, &x) in self.data.iter().enumerate() {
            let idx = unravel(i, &self.shape);
            let key: Vec<usize> = kept.iter().map(|&d| idx[d]).collect();
            let o = ravel(&key, &out_shape);
            acc[o] = f(acc[o], x, seen[o]);
            seen[o] += 1;
        }
        let shape = if keepdims {
            (0..self.shape.len())
                .map(|d| if axes.contains(&d) { 1 } else { self.shape[d] })
                .collect()
        } else {
            out_shape
        };
        Ref { shape, data: acc }
    }

    /// Element-wise combination after broadcasting both operands.
    pub fn zip<U: Copy, V: Copy>(&self, other: &Ref<U>, f: impl Fn(T, U) -> V) -> Ref<V> {
        let shape = broadcast_shape(&self.shape, &other.shape).expect("broadcastable");
        let (a, b) = (self.broadcast(&shape), other.broadcast(&shape));
        Ref {
            shape,
            data: a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect(),
        }
    }
}

/// Values of `shape` with a random memory layout: contiguous, transposed, or a strided
/// (possibly reversed) slice of a larger buffer. The logical contents are `values`.
pub fn random_layout<T: Element>(
    rng: &mut Rng,
    shape: &[usize],
    values: Vec<T>,
) -> (Array, Ref<T>) {
    let reference = Ref::new(shape.to_vec(), values);
    let layout = if reference.data.is_empty() {
        0
    } else {
        rng.below(3)
    };
    let array = match layout {
        0 => Array::from_slice(&reference.data, shape).unwrap(),
        1 => {
            let axes: Vec<usize> = (0..shape.len()).rev().collect();
            let stored = reference.transpose(&axes);
            Array::from_slice(&stored.data, &stored.shape)
                .unwrap()
                .t()
                .unwrap()
        }
        _ => {
            // Every logical element sits at an even (or mirrored) position of a 2x-larger
            // buffer; slicing with step ±2 recovers it.
            let reversed: Vec<bool> = shape.iter().map(|_| rng.coin()).collect();
            let big: Vec<usize> = shape.iter().map(|&d| 2 * d).collect();
            let filler = reference.data.first().copied();
            let mut stored = vec![filler.expect("non-empty"); size(&big)];
            for i in 0..reference.data.len() {
                let idx = unravel(i, shape);
                let pos: Vec<usize> = idx
                    .iter()
                    .zip(shape)
                    .zip(&reversed)
                    .map(|((&j, &d), &r)| if r { 2 * d - 1 - 2 * j } else { 2 * j })
                    .collect();
                stored[ravel(&pos, &big)] = reference.data[i];
            }
            let args: Vec<SliceArg> = reversed
                .iter()
                .map(|&r| SliceArg::new(None, None, if r { -2 } else { 2 }))
                .collect();
            slice(&Array::from_slice(&stored, &big).unwrap(), &args).unwrap()
        }
    };
    (array, reference)
}

/// Equality that treats all NaNs as equal and distinguishes nothing else.
pub trait Same: Copy + std::fmt::Debug {
    fn same(self, other: Self) -> bool;
}

macro_rules! same_eq {
    ($($t:ty),*) => {$( impl Same for $t { fn same(self, o: Self) -> bool { self == o } } )*};
}
same_eq!(bool, u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! same_float {
    ($($t:ty),*) => {$( impl Same for $t {
        fn same(self, o: Self) -> bool { self.to_bits() == o.to_bits() || (self.is_nan() && o.is_nan()) }
    } )*};
}
same_float!(f32, f64, nuvec_core::f16, nuvec_core::bf16);

impl Same for nuvec_core::Complex64 {
    fn same(self, o: Self) -> bool {
        self.re.same(o.re) && self.im.same(o.im)
    }
}

/// Asserts that `array` holds exactly the reference values.
pub fn assert_matches<T: Element + Same>(array: &Array, expected: &Ref<T>, context: &str) {
    assert_eq!(array.shape(), expected.shape.as_slice(), "{context}: shape");
    let got = array.to_vec::<T>().unwrap();
    for (i, (g, e)) in got.iter().zip(&expected.data).enumerate() {
        assert!(
            g.same(*e),
            "{context}: element {i}: got {g:?}, expected {e:?}"
        );
    }
}

/// Matrix product of logical operands in f64, broadcasting batch dimensions (both operands
/// at least 2-D).
pub fn matmul_ref(a: &Ref<f64>, b: &Ref<f64>) -> Ref<f64> {
    let (na, nb) = (a.shape.len(), b.shape.len());
    let (m, k, n) = (a.shape[na - 2], a.shape[na - 1], b.shape[nb - 1]);
    let batch = broadcast_shape(&a.shape[..na - 2], &b.shape[..nb - 2]).expect("broadcastable");
    let full = |r: &Ref<f64>, rows: usize, cols: usize| {
        let mut s = batch.clone();
        s.extend([rows, cols]);
        r.broadcast(&s)
    };
    let (a, b) = (full(a, m, k), full(b, k, n));
    let mut shape = batch.clone();
    shape.extend([m, n]);
    let data = (0..size(&shape))
        .map(|i| {
            let idx = unravel(i, &shape);
            let (bi, r, c) = (
                &idx[..idx.len() - 2],
                idx[idx.len() - 2],
                idx[idx.len() - 1],
            );
            (0..k)
                .map(|p| {
                    let at = |x: &Ref<f64>, row: usize, col: usize| {
                        let mut j = bi.to_vec();
                        j.extend([row, col]);
                        x.data[ravel(&j, &x.shape)]
                    };
                    at(&a, r, p) * at(&b, p, c)
                })
                .sum()
        })
        .collect();
    Ref { shape, data }
}
