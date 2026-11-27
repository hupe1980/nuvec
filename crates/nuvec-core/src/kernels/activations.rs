//! MLX CPU uses float32 intermediates for scalar __fp16 arithmetic, but rounds after
//! each operation in a complete SIMD vector. The final at-most-seven values in a
//! contiguous run and all values in a scalar-strided run take the scalar formulas.

use std::sync::LazyLock;

use super::*;

const WIDTH: usize = 8;

/// Count physically contiguous distinct values, allowing singleton and broadcast axes.
fn dense_span(a: &Output) -> Option<usize> {
    if a.flags.row_contiguous || a.flags.col_contiguous {
        return Some(a.size);
    }
    let mut dims: Vec<_> = a
        .strides
        .iter()
        .zip(&a.shape)
        .filter(|(s, n)| **s != 0 && **n > 1)
        .collect();
    dims.sort_unstable_by_key(|(s, _)| **s);
    let mut len = 1;
    for (&step, &n) in dims {
        if step != len as isize {
            return None;
        }
        len *= n;
    }
    Some(len)
}

/// Number of contiguous or scalar-broadcast elements shared by both operands at
/// the inner edge of their shape. MLX's general binary dispatch specializes runs of 16+.
fn common_run(a: &Output, b: &Output) -> usize {
    fn suffix(a: &Output, broadcast: bool) -> usize {
        let mut len = 1;
        for (&n, &step) in a.shape.iter().zip(&a.strides).rev() {
            if n != 1 && step != if broadcast { 0 } else { len as isize } {
                break;
            }
            len *= n;
        }
        len
    }
    let (va, vb) = (suffix(a, false), suffix(b, false));
    let width = va
        .min(vb)
        .max(va.min(suffix(b, true)))
        .max(vb.min(suffix(a, true)));
    if width >= 16 { width } else { 1 }
}

/// Identifies vector positions in logical (row-major) result order.
struct Run<'a> {
    width: usize,
    physical: Option<&'a Output>,
}

impl<'a> Run<'a> {
    fn dense(a: &'a Output, width: usize) -> Self {
        Self {
            width,
            physical: (!a.flags.row_contiguous).then_some(a),
        }
    }

    fn for_binary(a: &'a Output, b: &'a Output) -> Self {
        let a_span = dense_span(a);
        let b_span = dense_span(b);
        if a_span == Some(1)
            && let Some(n) = b_span
        {
            return Self::dense(b, n);
        }
        if b_span == Some(1)
            && let Some(n) = a_span
        {
            return Self::dense(a, n);
        }
        if a.flags.row_contiguous && b.flags.row_contiguous
            || a.flags.col_contiguous && b.flags.col_contiguous
        {
            return Self::dense(a, a.size);
        }
        Self {
            width: common_run(a, b),
            physical: None,
        }
    }

    fn vector_at(&self, i: usize) -> bool {
        let index = if let Some(a) = self.physical {
            let mut flat = i;
            let mut address = 0;
            for (&n, &step) in a.shape.iter().zip(&a.strides).rev() {
                address += flat % n * step as usize;
                flat /= n;
            }
            address
        } else {
            i % self.width
        };
        index < self.width / WIDTH * WIDTH
    }
}

fn scalar_sigmoid(x: f16) -> f16 {
    let x = x.to_f32();
    let exp = f16::from_f32(mlx_cpu_exp(x.abs())).to_f32();
    let y = 1.0 / (1.0 + exp);
    f16::from_f32(if x < 0.0 { y } else { 1.0 - y })
}

fn scalar_logaddexp(x: f16, y: f16) -> f16 {
    let (x, y) = (x.to_f32(), y.to_f32());
    let (hi, lo) = (x.max_(y), x.min_(y));
    f16::from_f32(if lo == f32::NEG_INFINITY || hi == f32::INFINITY {
        hi
    } else {
        hi + mlx_cpu_exp(lo - hi).ln_1p()
    })
}

// Repeated evaluations on wide vectors can reuse the rounded transcendentals.
// The tables for a format occupy 192 KiB when both operations have been used.
// Small inputs keep their direct calculation and never initialize a table.
const TABLE_MIN_ELEMENTS: usize = 512;

fn lookup<T>(count: u32, value: impl Fn(u16) -> T) -> Box<[T]> {
    (0..count).map(|bits| value(bits as u16)).collect()
}

macro_rules! vector_activations {
    ($half:ty, $sigmoid:ident, $logaddexp:ident) => {
        fn $sigmoid(size: usize) -> impl Fn($half) -> $half {
            static TABLE: LazyLock<Box<[$half]>> = LazyLock::new(|| {
                lookup(1 << 16, |bits| {
                    <$half>::from_bits(bits).float_(UnaryOp::Sigmoid)
                })
            });
            let table = (size >= TABLE_MIN_ELEMENTS).then(|| &**TABLE);
            move |x| match table {
                Some(t) => t[usize::from(x.to_bits())],
                None => x.float_(UnaryOp::Sigmoid),
            }
        }

        fn $logaddexp(size: usize) -> impl Fn($half, $half) -> $half {
            // After half rounding, the magnitude of x-y takes at most 32,768 values.
            static TABLE: LazyLock<Box<[$half]>> = LazyLock::new(|| {
                lookup(1 << 15, |bits| {
                    let delta = <$half>::from_bits(bits).to_f32();
                    let exp = <$half>::from_f32(mlx_cpu_exp(-delta));
                    <$half>::from_f32(exp.to_f32().ln_1p())
                })
            });
            let table = (size >= TABLE_MIN_ELEMENTS).then(|| &**TABLE);
            move |x, y| {
                let Some(table) = table else {
                    return x.logaddexp_(y);
                };
                let (x, y) = (x.to_f32(), y.to_f32());
                let hi = x.max_(y);
                if x.is_infinite() || y.is_infinite() {
                    return <$half>::from_f32(hi);
                }
                let delta = <$half>::from_f32((x - y).abs());
                <$half>::from_f32(hi + table[usize::from(delta.to_bits() & 0x7fff)].to_f32())
            }
        }
    };
}
vector_activations!(f16, vector_sigmoid_f16, vector_logaddexp_f16);
vector_activations!(bf16, sigmoid_bf16, logaddexp_bf16);

pub(super) fn sigmoid(a: Operand<'_>) -> Result<Buffer> {
    if a.0.dtype == DType::BF16 {
        let mut result = output::<bf16>(a.0.size)?;
        map1(
            &a.0.shape,
            view(a),
            result.typed_mut::<bf16>(),
            sigmoid_bf16(a.0.size),
        );
        return Ok(result);
    }
    let run = Run::dense(a.0, dense_span(a.0).unwrap_or(1));
    let mut result = output::<f16>(a.0.size)?;
    let dst = result.typed_mut::<f16>();
    let vector = vector_sigmoid_f16(if run.width >= WIDTH { a.0.size } else { 0 });
    if run.width < WIDTH {
        map1(&a.0.shape, view(a), dst, scalar_sigmoid);
    } else if run.width.is_multiple_of(WIDTH) {
        map1(&a.0.shape, view(a), dst, vector);
    } else {
        let index = std::cell::Cell::new(0usize);
        map1(&a.0.shape, view(a), dst, |x| {
            let i = index.get();
            index.set(i + 1);
            if run.vector_at(i) {
                vector(x)
            } else {
                scalar_sigmoid(x)
            }
        });
    }
    Ok(result)
}

pub(super) fn logaddexp(a: Operand<'_>, b: Operand<'_>) -> Result<Buffer> {
    if a.0.dtype == DType::BF16 {
        let mut result = output::<bf16>(a.0.size)?;
        map2(
            &a.0.shape,
            view(a),
            view(b),
            result.typed_mut::<bf16>(),
            logaddexp_bf16(a.0.size),
        );
        return Ok(result);
    }
    let run = Run::for_binary(a.0, b.0);
    let vector = vector_logaddexp_f16(if run.width >= WIDTH { a.0.size } else { 0 });
    let mut result = output::<f16>(a.0.size)?;
    let dst = result.typed_mut::<f16>();
    if run.width < WIDTH {
        map2(&a.0.shape, view(a), view(b), dst, scalar_logaddexp);
    } else if run.width.is_multiple_of(WIDTH) {
        map2(&a.0.shape, view(a), view(b), dst, vector);
    } else {
        let index = std::cell::Cell::new(0usize);
        map2(&a.0.shape, view(a), view(b), dst, |x: f16, y| {
            let i = index.get();
            index.set(i + 1);
            if run.vector_at(i) {
                vector(x, y)
            } else {
                scalar_logaddexp(x, y)
            }
        });
    }
    Ok(result)
}
