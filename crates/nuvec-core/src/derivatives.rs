//! Derivative rules: `Primitive::vjp` (reverse mode) and `Primitive::jvp` (forward mode).
//!
//! The rules follow MLX's `primitives.cpp` and are built from ops, so derivatives are lazy
//! graphs that can be evaluated anywhere and differentiated again. Every primitive has both
//! rules (D-015: a missing rule is a compile error). Complex reverse-mode rules follow MLX's
//! convention: the cotangent times the conjugate derivative (D-030).

use crate::array::Array;
use crate::cast::Canon;
use crate::ops::{
    self, broadcast_to, conjugate, cos, divide, equal, greater, greater_equal, less, less_equal,
    matmul, multiply, negative, reshape, sin, subtract, sum, transpose, where_,
};
use crate::primitive::{BinaryOp, Primitive, ReduceOp, ScatterReduce, UnaryOp};
use crate::quantized::{dequantize, quantized_matmul};
use crate::{DType, Error, Result};

/// Records `primitive` on `inputs` (single output).
fn record(primitive: Primitive, inputs: Vec<Array>) -> Result<Array> {
    Ok(Array::from_op(primitive, inputs)?.swap_remove(0))
}

/// Zeros shaped like `a`, as a broadcast scalar (no allocation).
pub(crate) fn zeros_like(a: &Array) -> Result<Array> {
    let zero = ops::filled(&[], a.dtype(), |_| Canon::Int(0))?;
    broadcast_to(&zero, a.shape())
}

/// Ones shaped like `a`, as a broadcast scalar.
fn ones_like(a: &Array) -> Result<Array> {
    let one = ops::filled(&[], a.dtype(), |_| Canon::Int(1))?;
    broadcast_to(&one, a.shape())
}

/// `a` with its last axis split into groups: `[..., C]` → `[..., C / size, size]`.
fn grouped(a: &Array, size: usize) -> Result<Array> {
    let mut shape = isize_shape(a.shape());
    let last = shape.pop().expect("at least 1-D");
    shape.extend([last / size as isize, size as isize]);
    reshape(a, &shape)
}

/// `a` with its leading axes flattened into one, leaving `ndim` axes.
fn flatten_front(a: &Array, ndim: usize) -> Result<Array> {
    let mut shape = vec![-1];
    shape.extend(isize_shape(&a.shape()[a.ndim() + 1 - ndim..]));
    reshape(a, &shape)
}

/// The gradients of `scales` and `biases` from `dw`, the gradient of the dequantized
/// weights (MLX's `QuantizedMatmul::vjp`): per group, `Σ dw · code` and `Σ dw`.
fn scale_bias_grads(
    dw: &Array,
    inputs: &[Array],
    group_size: usize,
    bits: u8,
    wanted: [bool; 2],
) -> Result<[Option<Array>; 2]> {
    let (w, scales, biases) = (&inputs[0], &inputs[1], &inputs[2]);
    let dw = grouped(dw, group_size)?;
    let ds = match wanted[0] {
        true => {
            let codes = dequantize(
                w,
                &ones_like(scales)?,
                &zeros_like(biases)?,
                group_size,
                u32::from(bits),
            )?;
            Some(sum(
                &multiply(&dw, grouped(&codes, group_size)?)?,
                &[-1],
                false,
            )?)
        }
        false => None,
    };
    let db = match wanted[1] {
        true => Some(sum(&dw, &[-1], false)?),
        false => None,
    };
    Ok([ds, db])
}

fn no_weight_gradient(op: &str) -> Error {
    Error::InvalidArgument(format!(
        "{op}: there is no gradient with respect to the packed (quantized) weights"
    ))
}

fn isize_shape(shape: &[usize]) -> Vec<isize> {
    shape.iter().map(|&n| n as isize).collect()
}

/// `a` with its last two axes swapped.
fn swap_last(a: &Array) -> Result<Array> {
    let n = a.ndim();
    let mut axes: Vec<usize> = (0..n).collect();
    axes.swap(n - 2, n - 1);
    transpose(a, &axes)
}

/// `-1`, `0` or `1` per element, in `x`'s dtype.
fn sign(x: &Array) -> Result<Array> {
    subtract(
        greater(x, 0)?.astype(x.dtype())?,
        less(x, 0)?.astype(x.dtype())?,
    )
}

/// `g`, of the reduced (keepdims) shape, summed down from a broadcast of `shape`.
fn sum_to(g: &Array, shape: &[usize]) -> Result<Array> {
    let extra = g.ndim() - shape.len();
    let axes: Vec<isize> = (0..g.ndim())
        .filter(|&i| i < extra || (shape[i - extra] == 1 && g.shape()[i] != 1))
        .map(|i| i as isize)
        .collect();
    let summed = if axes.is_empty() {
        g.clone()
    } else {
        sum(g, &axes, true)?
    };
    reshape(&summed, &isize_shape(shape))
}

/// `∂out/∂x` of `prod` over `axes`: the product of the other elements, exact with zeros.
fn prod_weights(x: &Array, out: &Array, axes: &[usize]) -> Result<Array> {
    let axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
    let zero = equal(x, 0)?;
    let zeros = sum(&zero.astype(x.dtype())?, &axes, true)?;
    let others = ops::prod(&where_(&zero, 1, x)?, &axes, true)?;
    let only_zero = multiply(&zero, equal(&zeros, 1)?)?;
    where_(
        equal(&zeros, 0)?,
        divide(out, x)?,
        where_(&only_zero, &others, 0)?,
    )
}

/// A one-hot mask (in `x`'s dtype) of the first maximum (`ReduceOp::Max`) or minimum over
/// `axes`, in row-major order of the reduced axes. MLX's forward-mode rule takes the
/// tangent there, while its reverse-mode rule splits ties evenly ([`extreme_weights`]).
fn first_extreme(x: &Array, axes: &[usize], op: ReduceOp) -> Result<Array> {
    let kept: Vec<usize> = (0..x.ndim()).filter(|a| !axes.contains(a)).collect();
    let perm: Vec<usize> = kept.iter().chain(axes).copied().collect();
    let moved = transpose(x, &perm)?;
    let mut flat = isize_shape(&moved.shape()[..kept.len()]);
    let reduced: usize = axes.iter().map(|&a| x.shape()[a]).product();
    flat.push(reduced as isize);
    let rows = reshape(&moved, &flat)?;
    let index = match op {
        ReduceOp::Max => ops::argmax(&rows, Some(-1), true)?,
        _ => ops::argmin(&rows, Some(-1), true)?,
    };
    let positions = ops::arange(0.0, reduced as f64, 1.0, DType::U32)?;
    let onehot = equal(&positions, &index)?.astype(x.dtype())?;
    let mut inverse = vec![0; perm.len()];
    for (i, &a) in perm.iter().enumerate() {
        inverse[a] = i;
    }
    transpose(&reshape(&onehot, &isize_shape(moved.shape()))?, &inverse)
}

/// `z / |z|` as complex64 (0 at 0): the reverse-mode derivative of complex `abs`.
fn complex_sign(z: &Array, magnitude: &Array) -> Result<Array> {
    let unit = divide(z, magnitude.astype(DType::Complex64)?)?;
    where_(equal(magnitude, 0)?, 0, &unit)
}

/// `∂out/∂x` of `max`/`min` over reduced axes: ties share the derivative evenly.
fn extreme_weights(x: &Array, out: &Array, axes: &[usize]) -> Result<Array> {
    let axes: Vec<isize> = axes.iter().map(|&a| a as isize).collect();
    let mask = equal(x, out)?.astype(x.dtype())?;
    let count = sum(&mask, &axes, true)?;
    divide(&mask, &count)
}

impl Primitive {
    /// `J_i · v` for element-wise primitives, whose Jacobians are diagonal: the same rule
    /// serves reverse mode (`v` a cotangent) and forward mode (`v` a tangent). `None` for
    /// inputs without a derivative.
    fn elementwise_partial(
        &self,
        inputs: &[Array],
        out: &Array,
        i: usize,
        v: &Array,
    ) -> Result<Option<Array>> {
        let x = &inputs[0];
        Ok(Some(match self {
            Self::Unary(op) => match op {
                UnaryOp::Negative => negative(v)?,
                UnaryOp::Abs => multiply(v, sign(x)?)?,
                UnaryOp::Exp => multiply(v, out)?,
                UnaryOp::Log => divide(v, x)?,
                UnaryOp::Sqrt => divide(v, multiply(2, out)?)?,
                UnaryOp::Sin => multiply(v, cos(x)?)?,
                UnaryOp::Cos => multiply(v, negative(&sin(x)?)?)?,
                UnaryOp::Tanh => multiply(v, subtract(1, multiply(out, out)?)?)?,
                UnaryOp::Sigmoid => multiply(v, multiply(out, subtract(1, out)?)?)?,
                UnaryOp::Conjugate => conjugate(v)?,
                // d/dx erfinv(x) = √π/2 · exp(erfinv(x)²)
                UnaryOp::ErfInv => multiply(
                    multiply(v, std::f64::consts::PI.sqrt() / 2.0)?,
                    ops::exp(&multiply(out, out)?)?,
                )?,
            },
            Self::Binary(op) => {
                let (a, b) = (&inputs[0], &inputs[1]);
                match (op, i) {
                    (BinaryOp::Add, _) => v.clone(),
                    (BinaryOp::Subtract, 0) => v.clone(),
                    (BinaryOp::Subtract, _) => negative(v)?,
                    (BinaryOp::Multiply, 0) => multiply(v, b)?,
                    (BinaryOp::Multiply, _) => multiply(v, a)?,
                    (BinaryOp::Divide, 0) => divide(v, b)?,
                    (BinaryOp::Divide, _) => negative(&divide(multiply(v, a)?, multiply(b, b)?)?)?,
                    // Ties go to `b`, as in MLX.
                    (BinaryOp::Maximum, 0) => multiply(v, greater(a, b)?.astype(a.dtype())?)?,
                    (BinaryOp::Maximum, _) => multiply(v, less_equal(a, b)?.astype(a.dtype())?)?,
                    (BinaryOp::Minimum, 0) => multiply(v, less(a, b)?.astype(a.dtype())?)?,
                    (BinaryOp::Minimum, _) => multiply(v, greater_equal(a, b)?.astype(a.dtype())?)?,
                    // Comparisons have no derivative.
                    _ => return Ok(None),
                }
            }
            Self::Select => match i {
                0 => return Ok(None),
                1 => where_(x, v, 0)?,
                _ => where_(x, 0, v)?,
            },
            _ => unreachable!("not element-wise: {}", self.name()),
        }))
    }

    fn is_elementwise(&self) -> bool {
        matches!(self, Self::Unary(_) | Self::Binary(_) | Self::Select)
    }

    /// Cotangents of the inputs marked in `wanted`, given one cotangent per output.
    /// `None` means a zero cotangent.
    pub(crate) fn vjp(
        &self,
        inputs: &[Array],
        outputs: &[Array],
        cotangents: &[Array],
        wanted: &[bool],
    ) -> Result<Vec<Option<Array>>> {
        let g = &cotangents[0];
        if let (Self::Unary(UnaryOp::Abs), DType::Complex64) = (self, inputs[0].dtype()) {
            return Ok(vec![Some(multiply(
                g.astype(DType::Complex64)?,
                complex_sign(&inputs[0], &outputs[0])?,
            )?)]);
        }
        if self.is_elementwise() {
            // MLX's convention: the cotangent times the conjugate derivative. These
            // operations have real coefficients, so that is the rule evaluated on conjugated
            // primals (`conjugate` is the identity on real arrays).
            let inputs = inputs.iter().map(conjugate).collect::<Result<Vec<_>>>()?;
            let out = conjugate(&outputs[0])?;
            return (0..inputs.len())
                .map(|i| match wanted[i] {
                    true => self.elementwise_partial(&inputs, &out, i, g),
                    false => Ok(None),
                })
                .collect();
        }
        let x = &inputs[0];
        let one = |a: Result<Array>| a.map(|a| vec![Some(a)]);
        match self {
            Self::AsType(_) => one(g.astype(x.dtype())),
            Self::Reshape(_) => one(reshape(g, &isize_shape(x.shape()))),
            Self::Transpose(axes) => {
                let mut inverse = vec![0; axes.len()];
                for (i, &a) in axes.iter().enumerate() {
                    inverse[a] = i;
                }
                one(transpose(g, &inverse))
            }
            Self::Slice { start, step, len } => {
                let update = Self::SliceUpdate {
                    start: start.clone(),
                    step: step.clone(),
                    len: len.clone(),
                };
                one(record(update, vec![zeros_like(x)?, g.clone()]))
            }
            Self::GatherAxis { axis } => {
                // Repeated indices accumulate their cotangents.
                let scatter = Self::ScatterAxis {
                    axis: *axis,
                    reduce: ScatterReduce::Sum,
                };
                let src = match wanted[0] {
                    true => Some(record(
                        scatter,
                        vec![zeros_like(x)?, inputs[1].clone(), g.clone()],
                    )?),
                    false => None,
                };
                Ok(vec![src, None])
            }
            Self::ScatterAxis { axis, reduce } => {
                let (indices, updates) = (&inputs[1], &inputs[2]);
                let dst = match (wanted[0], reduce) {
                    (false, _) => None,
                    // Overwritten positions get no gradient.
                    (true, ScatterReduce::None) => Some(record(
                        self.clone(),
                        vec![g.clone(), indices.clone(), zeros_like(updates)?],
                    )?),
                    (true, _) => Some(g.clone()),
                };
                let gathered = match wanted[2] {
                    true => Some(record(
                        Self::GatherAxis { axis: *axis },
                        vec![g.clone(), indices.clone()],
                    )?),
                    false => None,
                };
                Ok(vec![dst, None, gathered])
            }
            Self::Concatenate { axis } => {
                let mut start = 0;
                inputs
                    .iter()
                    .zip(wanted)
                    .map(|(a, &w)| {
                        let first = start;
                        start += a.shape()[*axis];
                        if !w {
                            return Ok(None);
                        }
                        let mut begin = vec![0; a.ndim()];
                        begin[*axis] = first;
                        let part = Self::Slice {
                            start: begin,
                            step: vec![1; a.ndim()],
                            len: a.shape().to_vec(),
                        };
                        Ok(Some(record(part, vec![g.clone()])?))
                    })
                    .collect()
            }
            Self::SliceUpdate { start, step, len } => {
                let update = &inputs[1];
                let base = match wanted[0] {
                    true => Some(record(self.clone(), vec![g.clone(), zeros_like(update)?])?),
                    false => None,
                };
                let slice = Self::Slice {
                    start: start.clone(),
                    step: step.clone(),
                    len: len.clone(),
                };
                let update = match wanted[1] {
                    true => Some(record(slice, vec![g.clone()])?),
                    false => None,
                };
                Ok(vec![base, update])
            }
            Self::Broadcast(_) => one(sum_to(g, x.shape())),
            Self::StopGradient | Self::ArgReduce { .. } | Self::RandomBits { .. } => Ok(vec![None]),
            Self::Reduce { op, axes } => match op {
                ReduceOp::Sum => one(broadcast_to(g, x.shape())),
                ReduceOp::Prod => one(multiply(
                    g,
                    conjugate(&prod_weights(x, &outputs[0], axes)?)?,
                )),
                ReduceOp::Max | ReduceOp::Min => {
                    one(multiply(g, extreme_weights(x, &outputs[0], axes)?))
                }
                ReduceOp::And | ReduceOp::Or => Ok(vec![None]),
            },
            Self::Matmul => {
                let (a, b) = (&inputs[0], &inputs[1]);
                let da = match wanted[0] {
                    true => Some(matmul(g, &conjugate(&swap_last(b)?)?)?),
                    false => None,
                };
                let db = match wanted[1] {
                    true => Some(matmul(&conjugate(&swap_last(a)?)?, g)?),
                    false => None,
                };
                Ok(vec![da, db])
            }
            Self::AddMM { .. } => {
                let (alpha, beta) = self.addmm_factors();
                let (a, b) = (&inputs[0], &inputs[1]);
                let scaled = multiply(g, f64::from(alpha))?;
                let da = match wanted[0] {
                    true => Some(matmul(&scaled, &conjugate(&swap_last(b)?)?)?),
                    false => None,
                };
                let db = match wanted[1] {
                    true => Some(matmul(&conjugate(&swap_last(a)?)?, &scaled)?),
                    false => None,
                };
                let dc = match wanted[2] {
                    true => Some(multiply(g, f64::from(beta))?),
                    false => None,
                };
                Ok(vec![da, db, dc])
            }
            Self::Fast(op) => op.vjp(inputs, outputs, cotangents, wanted),
            Self::Compiled(fused) => fused.vjp(inputs, cotangents, wanted),
            Self::Quantize { .. } => Ok(vec![None]),
            Self::Dequantize { group_size, bits } => {
                if wanted[0] {
                    return Err(no_weight_gradient("dequantize"));
                }
                let [ds, db] =
                    scale_bias_grads(g, inputs, *group_size, *bits, [wanted[1], wanted[2]])?;
                Ok(vec![None, ds, db])
            }
            Self::QuantizedMatmul {
                group_size,
                bits,
                transpose,
            } => {
                if wanted[1] {
                    return Err(no_weight_gradient("quantized_matmul"));
                }
                let (x, w, scales, biases) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
                let dx = match wanted[0] {
                    true => Some(quantized_matmul(
                        g,
                        w,
                        scales,
                        biases,
                        !transpose,
                        *group_size,
                        u32::from(*bits),
                    )?),
                    false => None,
                };
                let [ds, db] = match wanted[2] || wanted[3] {
                    true => {
                        // The gradient of the dequantized weights, summed over the batch
                        // dimensions `w` does not have.
                        let keep = w.ndim();
                        let (fg, fx) = (flatten_front(g, keep)?, flatten_front(x, keep)?);
                        let dw = match transpose {
                            true => matmul(&swap_last(&fg)?, &fx)?,
                            false => matmul(&swap_last(&fx)?, &fg)?,
                        };
                        let rest = [w.clone(), scales.clone(), biases.clone()];
                        scale_bias_grads(&dw, &rest, *group_size, *bits, [wanted[2], wanted[3]])?
                    }
                    false => [None, None],
                };
                Ok(vec![dx, None, ds, db])
            }
            Self::Custom(call) => {
                let n = call.primals;
                let primals = &inputs[..n];
                let grads = match &call.rules.vjp {
                    Some(rule) => rule(primals, outputs, cotangents)?,
                    None => crate::vjp(|xs| (call.rules.body)(xs), primals, cotangents)?.1,
                };
                if grads.len() != n {
                    return Err(Error::InvalidArgument(format!(
                        "custom_function: the vjp rule returned {} cotangents for {n} primals",
                        grads.len()
                    )));
                }
                // The function's outputs (the remaining inputs) get no cotangent.
                Ok(grads
                    .into_iter()
                    .zip(wanted)
                    .map(|(g, &w)| w.then_some(g))
                    .chain(std::iter::repeat_n(None, inputs.len() - n))
                    .collect())
            }
            #[cfg(test)]
            Self::AddMultiply => {
                let (a, b) = (&inputs[0], &inputs[1]);
                let h = &cotangents[1];
                Ok(vec![
                    Some(ops::add(g, multiply(h, b)?)?),
                    Some(ops::add(g, multiply(h, a)?)?),
                ])
            }
            Self::Unary(_) | Self::Binary(_) | Self::Select => unreachable!(),
        }
    }

    /// One tangent per output, given the inputs' tangents (`None` is zero; at least one is
    /// present).
    pub(crate) fn jvp(
        &self,
        inputs: &[Array],
        outputs: &[Array],
        tangents: &[Option<Array>],
    ) -> Result<Vec<Array>> {
        if let (Self::Unary(UnaryOp::Abs), DType::Complex64) = (self, inputs[0].dtype()) {
            // d|z| = Re(conj(z) dz) / |z|, zero at z = 0 (MLX).
            let (z, out) = (&inputs[0], &outputs[0]);
            let t = tangents[0].clone().map_or_else(|| zeros_like(z), Ok)?;
            let re = multiply(conjugate(z)?, &t)?.astype(DType::F32)?;
            return Ok(vec![where_(equal(out, 0)?, 0, divide(&re, out)?)?]);
        }
        let tangent = |i: usize| -> Result<Array> {
            match &tangents[i] {
                Some(t) => Ok(t.clone()),
                None => zeros_like(&inputs[i]),
            }
        };
        let out = &outputs[0];
        if self.is_elementwise() {
            let mut total: Option<Array> = None;
            for (i, t) in tangents.iter().enumerate() {
                let Some(t) = t else { continue };
                if let Some(d) = self.elementwise_partial(inputs, out, i, t)? {
                    total = Some(match total {
                        Some(sum) => ops::add(&sum, &d)?,
                        None => d,
                    });
                }
            }
            return Ok(vec![match total {
                Some(t) => t,
                None => zeros_like(out)?,
            }]);
        }
        let x = &inputs[0];
        let one = |a: Result<Array>| a.map(|a| vec![a]);
        match self {
            Self::AsType(dtype) => one(tangent(0)?.astype(*dtype)),
            Self::Reshape(_) => one(reshape(&tangent(0)?, &isize_shape(out.shape()))),
            Self::Transpose(axes) => one(transpose(&tangent(0)?, axes)),
            Self::Slice { .. } | Self::Broadcast(_) => one(record(self.clone(), vec![tangent(0)?])),
            Self::SliceUpdate { .. } => one(record(self.clone(), vec![tangent(0)?, tangent(1)?])),
            Self::GatherAxis { .. } => {
                one(record(self.clone(), vec![tangent(0)?, inputs[1].clone()]))
            }
            Self::ScatterAxis { .. } => one(record(
                self.clone(),
                vec![tangent(0)?, inputs[1].clone(), tangent(2)?],
            )),
            Self::Concatenate { .. } => one(record(
                self.clone(),
                (0..inputs.len()).map(tangent).collect::<Result<_>>()?,
            )),
            Self::StopGradient | Self::ArgReduce { .. } | Self::RandomBits { .. } => {
                one(zeros_like(out))
            }
            Self::Reduce { op, axes } => {
                let summed = |t: Array| {
                    record(
                        Self::Reduce {
                            op: ReduceOp::Sum,
                            axes: axes.clone(),
                        },
                        vec![t],
                    )
                };
                match op {
                    ReduceOp::Sum => one(summed(tangent(0)?)),
                    ReduceOp::Prod => {
                        one(summed(multiply(tangent(0)?, prod_weights(x, out, axes)?)?))
                    }
                    ReduceOp::Max | ReduceOp::Min => {
                        one(summed(multiply(tangent(0)?, first_extreme(x, axes, *op)?)?))
                    }
                    ReduceOp::And | ReduceOp::Or => one(zeros_like(out)),
                }
            }
            Self::Matmul => {
                let (a, b) = (&inputs[0], &inputs[1]);
                let total = match (&tangents[0], &tangents[1]) {
                    (Some(ta), Some(tb)) => ops::add(matmul(ta, b)?, matmul(a, tb)?)?,
                    (Some(ta), None) => matmul(ta, b)?,
                    (None, Some(tb)) => matmul(a, tb)?,
                    (None, None) => zeros_like(out)?,
                };
                one(Ok(total))
            }
            Self::AddMM { .. } => {
                let (alpha, beta) = self.addmm_factors();
                let (a, b) = (&inputs[0], &inputs[1]);
                let mut total = match (&tangents[0], &tangents[1]) {
                    (Some(ta), Some(tb)) => ops::add(matmul(ta, b)?, matmul(a, tb)?)?,
                    (Some(ta), None) => matmul(ta, b)?,
                    (None, Some(tb)) => matmul(a, tb)?,
                    (None, None) => zeros_like(out)?,
                };
                total = multiply(&total, f64::from(alpha))?;
                if let Some(tc) = &tangents[2] {
                    total = ops::add(&total, multiply(tc, f64::from(beta))?)?;
                }
                one(Ok(total))
            }
            Self::Fast(op) => op.jvp(inputs, tangents),
            Self::Compiled(fused) => fused.jvp(inputs, tangents),
            Self::Quantize { .. } => outputs.iter().map(zeros_like).collect(),
            // Linear in `(scales, biases)`: `dequantize(w, dscales, dbiases)`.
            Self::Dequantize { group_size, bits } => one(dequantize(
                &inputs[0],
                &tangent(1)?,
                &tangent(2)?,
                *group_size,
                u32::from(*bits),
            )),
            Self::QuantizedMatmul {
                group_size,
                bits,
                transpose,
            } => {
                let (x, w) = (&inputs[0], &inputs[1]);
                let mut total = match &tangents[0] {
                    Some(t) => quantized_matmul(
                        t,
                        w,
                        &inputs[2],
                        &inputs[3],
                        *transpose,
                        *group_size,
                        u32::from(*bits),
                    )?,
                    None => zeros_like(out)?,
                };
                if tangents[2].is_some() || tangents[3].is_some() {
                    let dw =
                        dequantize(w, &tangent(2)?, &tangent(3)?, *group_size, u32::from(*bits))?;
                    let dw = if *transpose { swap_last(&dw)? } else { dw };
                    total = ops::add(&total, matmul(x, &dw)?)?;
                }
                one(Ok(total))
            }
            Self::Custom(call) => {
                let n = call.primals;
                let primals = &inputs[..n];
                let tangents = (0..n).map(tangent).collect::<Result<Vec<_>>>()?;
                let result = match &call.rules.jvp {
                    Some(rule) => rule(primals, &tangents)?,
                    None => crate::jvp(|xs| (call.rules.body)(xs), primals, &tangents)?.1,
                };
                if result.len() != outputs.len() {
                    return Err(Error::InvalidArgument(format!(
                        "custom_function: the jvp rule returned {} tangents for {} outputs",
                        result.len(),
                        outputs.len()
                    )));
                }
                Ok(result)
            }
            #[cfg(test)]
            Self::AddMultiply => {
                let (a, b) = (&inputs[0], &inputs[1]);
                let (ta, tb) = (tangent(0)?, tangent(1)?);
                Ok(vec![
                    ops::add(&ta, &tb)?,
                    ops::add(multiply(&ta, b)?, multiply(a, &tb)?)?,
                ])
            }
            Self::Unary(_) | Self::Binary(_) | Self::Select => unreachable!(),
        }
    }
}
