//! CPU dispatch: picks the storage type for a dtype and runs a `nuvect-cpu` kernel.
//!
//! Operands have any layout; results are new row-contiguous buffers. The ops layer has
//! already validated shapes, dtypes and (op, dtype) support, so the per-type rules below
//! only see supported combinations.

use bytemuck::Pod;
use nuvect_cpu::{Gemm, MatView, View, map1, map2, map3};

use crate::array::{Data, Output};
use crate::buffer::Buffer;
use crate::cast::{B8, Canon, Scalar, convert, with_type};
use crate::primitive::{ArgReduceOp, BinaryOp, ReduceOp, UnaryOp};
use crate::shape::{self, Shape, Strides};
use crate::{Category, Complex64, DType, Result, bf16, f16};

/// An evaluated operand: its metadata and data.
pub(crate) type Operand<'a> = (&'a Output, &'a Data);

fn view<'a, T: Pod>((meta, data): Operand<'a>) -> View<'a, T> {
    View {
        data: data.buffer.typed::<T>(),
        offset: data.offset,
        strides: &meta.strides,
    }
}

fn output<T: Pod>(size: usize) -> Result<Buffer> {
    Buffer::try_zeroed(size * size_of::<T>())
}

/// The operand's elements, gathered into a row-contiguous buffer.
pub(crate) fn contiguous(meta: &Output, data: &Data) -> Result<Buffer> {
    fn copy<T: Pod>(a: Operand<'_>) -> Result<Buffer> {
        let mut out = output::<T>(a.0.size)?;
        map1(&a.0.shape, view::<T>(a), out.typed_mut::<T>(), |x| x);
        Ok(out)
    }
    match meta.dtype.size() {
        1 => copy::<u8>((meta, data)),
        2 => copy::<u16>((meta, data)),
        4 => copy::<u32>((meta, data)),
        _ => copy::<u64>((meta, data)),
    }
}

/// Conversion to `to`.
pub(crate) fn cast(a: Operand<'_>, to: DType) -> Result<Buffer> {
    with_type!(a.0.dtype, S => with_type!(to, D => {
        let mut out = output::<D>(a.0.size)?;
        map1(&a.0.shape, view::<S>(a), out.typed_mut::<D>(), convert::<S, D>);
        Ok(out)
    }))
}

/// `where(c, a, b)`. Type-agnostic: elements are moved by size.
pub(crate) fn select(c: Operand<'_>, a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
    fn run<T: Pod>(c: Operand<'_>, a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
        let mut out = output::<T>(a.0.size)?;
        map3(
            &a.0.shape,
            view::<B8>(c),
            view::<T>(a),
            view::<T>(b),
            out.typed_mut::<T>(),
            |c, x, y| {
                if c.0 != 0 { x } else { y }
            },
        );
        Ok(out)
    }
    match a.0.dtype.size() {
        1 => run::<u8>(c, a, b),
        2 => run::<u16>(c, a, b),
        4 => run::<u32>(c, a, b),
        _ => run::<u64>(c, a, b),
    }
}

/// An element-wise function of one operand.
///
/// Every arm passes a distinct function item (not a `fn` pointer), so each operation is
/// monomorphised and inlined into the kernel's loop.
pub(crate) fn unary(op: UnaryOp, a: Operand<'_>) -> Result<Buffer> {
    if (op, a.0.dtype) == (UnaryOp::Abs, DType::Complex64) {
        let mut out = output::<f32>(a.0.size)?;
        map1(
            &a.0.shape,
            view::<Complex64>(a),
            out.typed_mut::<f32>(),
            |z| z.re.hypot(z.im),
        );
        return Ok(out);
    }
    with_type!(a.0.dtype, T => {
        let mut out = output::<T>(a.0.size)?;
        let (shape, va, dst) = (&a.0.shape, view::<T>(a), out.typed_mut::<T>());
        macro_rules! go {
            ($f:expr) => { map1(shape, va, dst, $f) };
        }
        match op {
            UnaryOp::Negative => go!(T::neg_),
            UnaryOp::Abs => go!(T::abs_),
            UnaryOp::Exp => go!(|x: T| x.float_(UnaryOp::Exp)),
            UnaryOp::Log => go!(|x: T| x.float_(UnaryOp::Log)),
            UnaryOp::Sqrt => go!(|x: T| x.float_(UnaryOp::Sqrt)),
            UnaryOp::Sin => go!(|x: T| x.float_(UnaryOp::Sin)),
            UnaryOp::Cos => go!(|x: T| x.float_(UnaryOp::Cos)),
            UnaryOp::Tanh => go!(|x: T| x.float_(UnaryOp::Tanh)),
            UnaryOp::Sigmoid => go!(|x: T| x.float_(UnaryOp::Sigmoid)),
        }
        Ok(out)
    })
}

/// An element-wise function of two operands of equal shape and dtype. Monomorphised per
/// operation, as in [`unary`].
pub(crate) fn binary(op: BinaryOp, a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
    with_type!(a.0.dtype, T => {
        let (va, vb, shape) = (view::<T>(a), view::<T>(b), &a.0.shape);
        if op.is_comparison() {
            let mut out = output::<B8>(a.0.size)?;
            let dst = out.typed_mut::<B8>();
            macro_rules! cmp {
                ($f:expr) => { map2(shape, va, vb, dst, |x: T, y: T| B8(u8::from($f(x, y)))) };
            }
            match op {
                BinaryOp::Equal => cmp!(|x: T, y| x.eq_(y)),
                BinaryOp::NotEqual => cmp!(|x: T, y| !x.eq_(y)),
                BinaryOp::Less => cmp!(|x: T, y| x.lt_(y)),
                BinaryOp::LessEqual => cmp!(|x: T, y| x.le_(y)),
                BinaryOp::Greater => cmp!(|x, y: T| y.lt_(x)),
                _ => cmp!(|x, y: T| y.le_(x)),
            }
            Ok(out)
        } else {
            let mut out = output::<T>(a.0.size)?;
            let dst = out.typed_mut::<T>();
            macro_rules! go {
                ($f:expr) => { map2(shape, va, vb, dst, $f) };
            }
            match op {
                BinaryOp::Add => go!(T::add_),
                BinaryOp::Subtract => go!(T::sub_),
                BinaryOp::Multiply => go!(T::mul_),
                BinaryOp::Divide => go!(T::div_),
                BinaryOp::Maximum => go!(T::max_),
                _ => go!(T::min_),
            }
            Ok(out)
        }
    })
}

/// The kept and reduced `(shape, strides)` of an operand, split by `axes`.
fn split(meta: &Output, axes: &[usize]) -> (Shape, Strides, Shape, Strides) {
    let (mut ks, mut kt, mut rs, mut rt) =
        (Shape::new(), Strides::new(), Shape::new(), Strides::new());
    for (d, (&n, &s)) in meta.shape.iter().zip(&meta.strides).enumerate() {
        if axes.contains(&d) {
            rs.push(n);
            rt.push(s);
        } else {
            ks.push(n);
            kt.push(s);
        }
    }
    (ks, kt, rs, rt)
}

/// Folds every block of `a` reduced over `axes` into `out` (one element per kept position).
/// Contiguous blocks go to `block`, others element by element to `f`.
fn fold_with<T: Pod, A: Clone>(
    a: Operand<'_>,
    axes: &[usize],
    out: &mut [A],
    init: A,
    f: impl Fn(A, T, usize) -> A,
    block: impl Fn(A, &[T]) -> A,
) {
    let (ks, kt, rs, rt) = split(a.0, axes);
    nuvect_cpu::reduce(
        (&ks, &kt),
        (&rs, &rt),
        a.1.buffer.typed::<T>(),
        a.1.offset,
        out,
        init,
        f,
        block,
    );
}

/// [`fold_with`] where contiguous blocks are folded with `f` as well.
fn fold<T: Pod, A: Clone>(
    a: Operand<'_>,
    axes: &[usize],
    out: &mut [A],
    init: A,
    f: impl Fn(A, T, usize) -> A,
) {
    let block = |acc, xs: &[T]| xs.iter().enumerate().fold(acc, |acc, (i, &x)| f(acc, x, i));
    fold_with(a, axes, out, init, &f, block);
}

/// Sum of `xs` in f64 over 8 independent lanes, combined in a fixed order: vectorisable,
/// deterministic, and far more accurate than single-precision accumulation.
fn lane_sum<T: Scalar>(xs: &[T]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (chunks, rest) = xs.as_chunks::<8>();
    for chunk in chunks {
        for (a, &x) in acc.iter_mut().zip(chunk) {
            *a += convert::<T, f64>(x);
        }
    }
    for (a, &x) in acc.iter_mut().zip(rest) {
        *a += convert::<T, f64>(x);
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// A reduction over `axes`. The output keeps the input dtype (the ops layer casts first);
/// half and single precision sums and products accumulate in f64.
pub(crate) fn reduce(op: ReduceOp, a: Operand<'_>, axes: &[usize]) -> Result<Buffer> {
    let size: usize = split(a.0, axes).0.iter().product();
    match op {
        ReduceOp::Sum | ReduceOp::Prod if a.0.dtype.category() == Category::Float => {
            with_type!(a.0.dtype, T => {
                let mut acc = vec![0.0f64; size];
                if op == ReduceOp::Sum {
                    let f = |s: f64, x: T, _| s + convert::<T, f64>(x);
                    fold_with::<T, f64>(a, axes, &mut acc, 0.0, f, |s, xs| s + lane_sum(xs));
                } else {
                    fold::<T, f64>(a, axes, &mut acc, 1.0, |s, x, _| s * convert::<T, f64>(x));
                }
                let mut out = output::<T>(size)?;
                for (o, s) in out.typed_mut::<T>().iter_mut().zip(acc) {
                    *o = convert::<f64, T>(s);
                }
                Ok(out)
            })
        }
        ReduceOp::Sum | ReduceOp::Prod => with_type!(a.0.dtype, T => {
            let mut out = output::<T>(size)?;
            let dst = out.typed_mut::<T>();
            if op == ReduceOp::Sum {
                fold::<T, T>(a, axes, dst, T::from_canon(Canon::Int(0)), |s, x, _| s.add_(x));
            } else {
                fold::<T, T>(a, axes, dst, T::from_canon(Canon::Int(1)), |s, x, _| s.mul_(x));
            }
            Ok(out)
        }),
        ReduceOp::Max | ReduceOp::Min => with_type!(a.0.dtype, T => {
            let mut best: Vec<Option<T>> = vec![None; size];
            if op == ReduceOp::Max {
                fold::<T, Option<T>>(a, axes, &mut best, None, |b, x, _| Some(b.map_or(x, |b| b.max_(x))));
            } else {
                fold::<T, Option<T>>(a, axes, &mut best, None, |b, x, _| Some(b.map_or(x, |b| b.min_(x))));
            }
            let mut out = output::<T>(size)?;
            for (o, b) in out.typed_mut::<T>().iter_mut().zip(best) {
                *o = b.unwrap_or_else(|| T::from_canon(Canon::Int(0)));
            }
            Ok(out)
        }),
        ReduceOp::And | ReduceOp::Or => {
            let mut out = output::<B8>(size)?;
            let dst = out.typed_mut::<B8>();
            if op == ReduceOp::And {
                fold::<B8, B8>(a, axes, dst, B8(1), |s, x, _| B8(s.0 & x.0));
            } else {
                fold::<B8, B8>(a, axes, dst, B8(0), |s, x, _| B8(s.0 | x.0));
            }
            Ok(out)
        }
    }
}

/// Index (uint32) of the maximum or minimum along `axis`. First occurrence wins; the first
/// NaN wins over any number.
pub(crate) fn arg_reduce(op: ArgReduceOp, a: Operand<'_>, axis: usize) -> Result<Buffer> {
    fn run<T: Num>(
        a: Operand<'_>,
        axis: usize,
        size: usize,
        better: impl Fn(T, T) -> bool,
    ) -> Result<Buffer> {
        let mut best: Vec<Option<(T, usize)>> = vec![None; size];
        fold::<T, Option<(T, usize)>>(a, &[axis], &mut best, None, |b, x, i| match b {
            Some((v, _)) if v.is_nan_() => b,
            Some((v, _)) if !x.is_nan_() && !better(v, x) => b,
            _ => Some((x, i)),
        });
        let mut out = output::<u32>(size)?;
        for (o, b) in out.typed_mut::<u32>().iter_mut().zip(best) {
            *o = b.map_or(0, |(_, i)| i as u32);
        }
        Ok(out)
    }
    let size = split(a.0, &[axis]).0.iter().product();
    with_type!(a.0.dtype, T => match op {
        ArgReduceOp::Max => run::<T>(a, axis, size, |best, x| best.lt_(x)),
        ArgReduceOp::Min => run::<T>(a, axis, size, |best, x| x.lt_(best)),
    })
}

/// Batched `a · b` for `[B..., M, K]` and `[B..., K, N]` operands with equal batch dims.
pub(crate) fn matmul(a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
    match a.0.dtype {
        DType::F32 => matmul_typed::<f32, f32>(a, b),
        DType::F64 => matmul_typed::<f64, f64>(a, b),
        DType::F16 => matmul_typed::<f16, f16>(a, b),
        DType::Complex64 => matmul_typed::<Complex64, [f32; 2]>(a, b),
        _ => {
            // bfloat16 (no GEMM kernel): multiply in float32, round once.
            let (fa, fb) = (cast(a, DType::F32)?, cast(b, DType::F32)?);
            let row = |m: &Output| Output {
                dtype: DType::F32,
                shape: m.shape.clone(),
                strides: shape::row_contiguous_strides(&m.shape),
                flags: m.flags,
                size: m.size,
                slot: Default::default(),
            };
            let (ma, mb) = (row(a.0), row(b.0));
            let (da, db) = (Data::new(fa), Data::new(fb));
            let product = matmul_typed::<f32, f32>((&ma, &da), (&mb, &db))?;
            let mut out = output::<bf16>(product.typed::<f32>().len())?;
            for (o, &x) in out
                .typed_mut::<bf16>()
                .iter_mut()
                .zip(product.typed::<f32>())
            {
                *o = bf16::from_f32(x);
            }
            Ok(out)
        }
    }
}

/// `T` is the storage type, `G` the GEMM element type with the same layout.
fn matmul_typed<T: Pod, G: Gemm + Pod>(a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
    let (sa, sb) = (&a.0.shape, &b.0.shape);
    let nd = sa.len();
    let (m, k, n) = (sa[nd - 2], sa[nd - 1], sb[nd - 1]);
    let batch: usize = sa[..nd - 2].iter().product();
    let mut out = output::<T>(batch * m * n)?;
    let (da, db): (&[G], &[G]) = (
        bytemuck::cast_slice(a.1.buffer.typed::<T>()),
        bytemuck::cast_slice(b.1.buffer.typed::<T>()),
    );
    let dst: &mut [G] = bytemuck::cast_slice_mut(out.typed_mut::<T>());
    let mut index = vec![0usize; nd - 2];
    for chunk in dst.chunks_exact_mut((m * n).max(1)).take(batch) {
        let offset = |meta: &Output, data: &Data| {
            data.offset
                + index
                    .iter()
                    .zip(&meta.strides)
                    .map(|(&i, &s)| i as isize * s)
                    .sum::<isize>()
        };
        let va = MatView {
            data: da,
            offset: offset(a.0, a.1),
            row_stride: a.0.strides[nd - 2],
            col_stride: a.0.strides[nd - 1],
        };
        let vb = MatView {
            data: db,
            offset: offset(b.0, b.1),
            row_stride: b.0.strides[nd - 2],
            col_stride: b.0.strides[nd - 1],
        };
        nuvect_cpu::matmul(m, n, k, va, vb, &mut chunk[..m * n]);
        for d in (0..nd - 2).rev() {
            index[d] += 1;
            if index[d] < sa[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// Per-type element semantics, matching MLX: integers wrap; bool add/subtract/multiply
/// are OR/XOR/AND; float maximum/minimum propagate NaN; f16/bf16 compute in f32.
///
/// Operations the ops layer rejects for a type (ordering of complex, negation of bool,
/// transcendental functions of integers or complex, division of integers) are unreachable.
trait Num: Scalar {
    fn add_(self, o: Self) -> Self;
    fn sub_(self, o: Self) -> Self;
    fn mul_(self, o: Self) -> Self;
    fn div_(self, o: Self) -> Self;
    fn max_(self, o: Self) -> Self;
    fn min_(self, o: Self) -> Self;
    fn eq_(self, o: Self) -> bool;
    fn lt_(self, o: Self) -> bool;
    fn le_(self, o: Self) -> bool;
    fn neg_(self) -> Self;
    fn abs_(self) -> Self;
    fn float_(self, op: UnaryOp) -> Self;
    fn is_nan_(self) -> bool {
        false
    }
}

const REJECTED: &str = "rejected by the ops layer";

impl Num for B8 {
    fn add_(self, o: Self) -> Self {
        B8(self.0 | o.0)
    }
    fn sub_(self, o: Self) -> Self {
        B8(self.0 ^ o.0)
    }
    fn mul_(self, o: Self) -> Self {
        B8(self.0 & o.0)
    }
    fn div_(self, _: Self) -> Self {
        unreachable!("bool division: {REJECTED}")
    }
    fn max_(self, o: Self) -> Self {
        B8(self.0 | o.0)
    }
    fn min_(self, o: Self) -> Self {
        B8(self.0 & o.0)
    }
    fn eq_(self, o: Self) -> bool {
        self.0 == o.0
    }
    fn lt_(self, o: Self) -> bool {
        self.0 < o.0
    }
    fn le_(self, o: Self) -> bool {
        self.0 <= o.0
    }
    fn neg_(self) -> Self {
        unreachable!("bool negation: {REJECTED}")
    }
    fn abs_(self) -> Self {
        self
    }
    fn float_(self, _: UnaryOp) -> Self {
        unreachable!("bool transcendental: {REJECTED}")
    }
}

macro_rules! int_num {
    ($abs:expr; $($t:ty),*) => {$(
        impl Num for $t {
            fn add_(self, o: Self) -> Self { self.wrapping_add(o) }
            fn sub_(self, o: Self) -> Self { self.wrapping_sub(o) }
            fn mul_(self, o: Self) -> Self { self.wrapping_mul(o) }
            fn div_(self, _: Self) -> Self { unreachable!("integer division: {REJECTED}") }
            fn max_(self, o: Self) -> Self { Ord::max(self, o) }
            fn min_(self, o: Self) -> Self { Ord::min(self, o) }
            fn eq_(self, o: Self) -> bool { self == o }
            fn lt_(self, o: Self) -> bool { self < o }
            fn le_(self, o: Self) -> bool { self <= o }
            fn neg_(self) -> Self { self.wrapping_neg() }
            fn abs_(self) -> Self { let f: fn($t) -> $t = $abs; f(self) }
            fn float_(self, _: UnaryOp) -> Self { unreachable!("integer transcendental: {REJECTED}") }
        }
    )*};
}
int_num!(|x| x; u8, u16, u32, u64);
int_num!(|x| x.wrapping_abs(); i8, i16, i32, i64);

macro_rules! float_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            fn add_(self, o: Self) -> Self { self + o }
            fn sub_(self, o: Self) -> Self { self - o }
            fn mul_(self, o: Self) -> Self { self * o }
            fn div_(self, o: Self) -> Self { self / o }
            fn max_(self, o: Self) -> Self { if self.is_nan() || o.is_nan() { <$t>::NAN } else { self.max(o) } }
            fn min_(self, o: Self) -> Self { if self.is_nan() || o.is_nan() { <$t>::NAN } else { self.min(o) } }
            fn eq_(self, o: Self) -> bool { self == o }
            fn lt_(self, o: Self) -> bool { self < o }
            fn le_(self, o: Self) -> bool { self <= o }
            fn neg_(self) -> Self { -self }
            fn abs_(self) -> Self { self.abs() }
            fn is_nan_(self) -> bool { self.is_nan() }
            fn float_(self, op: UnaryOp) -> Self {
                match op {
                    UnaryOp::Exp => self.exp(),
                    UnaryOp::Log => self.ln(),
                    UnaryOp::Sqrt => self.sqrt(),
                    UnaryOp::Sin => self.sin(),
                    UnaryOp::Cos => self.cos(),
                    UnaryOp::Tanh => self.tanh(),
                    UnaryOp::Sigmoid => sigmoid(self),
                    UnaryOp::Negative | UnaryOp::Abs => unreachable!("not transcendental"),
                }
            }
        }
    )*};
}
float_num!(f32, f64);

/// Numerically stable logistic function.
pub(crate) fn sigmoid<T: num_like::Float>(x: T) -> T {
    if x >= T::ZERO {
        T::ONE / (T::ONE + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::ONE + e)
    }
}

mod num_like {
    /// The float operations `sigmoid` needs.
    pub(crate) trait Float:
        Copy
        + PartialOrd
        + std::ops::Neg<Output = Self>
        + std::ops::Add<Output = Self>
        + std::ops::Div<Output = Self>
    {
        const ZERO: Self;
        const ONE: Self;
        fn exp(self) -> Self;
    }
    impl Float for f32 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        fn exp(self) -> Self {
            f32::exp(self)
        }
    }
    impl Float for f64 {
        const ZERO: Self = 0.0;
        const ONE: Self = 1.0;
        fn exp(self) -> Self {
            f64::exp(self)
        }
    }
}

macro_rules! half_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            fn add_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32() + o.to_f32()) }
            fn sub_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32() - o.to_f32()) }
            fn mul_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32() * o.to_f32()) }
            fn div_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32() / o.to_f32()) }
            fn max_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32().max_(o.to_f32())) }
            fn min_(self, o: Self) -> Self { <$t>::from_f32(self.to_f32().min_(o.to_f32())) }
            fn eq_(self, o: Self) -> bool { self.to_f32() == o.to_f32() }
            fn lt_(self, o: Self) -> bool { self.to_f32() < o.to_f32() }
            fn le_(self, o: Self) -> bool { self.to_f32() <= o.to_f32() }
            fn neg_(self) -> Self { -self }
            fn abs_(self) -> Self { <$t>::from_f32(self.to_f32().abs()) }
            fn is_nan_(self) -> bool { self.is_nan() }
            fn float_(self, op: UnaryOp) -> Self { <$t>::from_f32(self.to_f32().float_(op)) }
        }
    )*};
}
half_num!(f16, bf16);

impl Num for Complex64 {
    fn add_(self, o: Self) -> Self {
        self + o
    }
    fn sub_(self, o: Self) -> Self {
        Complex64::new(self.re - o.re, self.im - o.im)
    }
    fn mul_(self, o: Self) -> Self {
        self * o
    }
    fn div_(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Complex64::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn max_(self, _: Self) -> Self {
        unreachable!("complex maximum: {REJECTED}")
    }
    fn min_(self, _: Self) -> Self {
        unreachable!("complex minimum: {REJECTED}")
    }
    fn eq_(self, o: Self) -> bool {
        self == o
    }
    fn lt_(self, _: Self) -> bool {
        unreachable!("complex ordering: {REJECTED}")
    }
    fn le_(self, _: Self) -> bool {
        unreachable!("complex ordering: {REJECTED}")
    }
    fn neg_(self) -> Self {
        Complex64::new(-self.re, -self.im)
    }
    fn abs_(self) -> Self {
        unreachable!("complex abs produces float32 and is handled separately")
    }
    fn float_(self, _: UnaryOp) -> Self {
        unreachable!("complex transcendental: {REJECTED}")
    }
}
