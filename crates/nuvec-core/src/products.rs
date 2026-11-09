//! MLX's special matrix products: `segmented_mm` (one product per segment of the inner
//! axis), `block_masked_mm` (block masks on the operands and the output) and the
//! products of quantized activations and weights, `qqmm` and `gather_qqmm`. They are
//! compositions of nuvec's products, so they differentiate and batch as those do.

use crate::custom::custom_function_flat;
use crate::{
    Array, Category, DType, Error, GatherQmm, IntoResult, Quant, QuantMode, Quantized, Result, ops,
};

fn invalid(name: &str, msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("{name}: {msg}"))
}

fn floating(name: &str, a: &Array, b: &Array) -> Result<DType> {
    let dtype = crate::promote_types(a.dtype(), b.dtype());
    if dtype.category() != Category::Float {
        return Err(invalid(
            name,
            format!("only real floating types are supported, got {dtype}"),
        ));
    }
    Ok(dtype)
}

/// `a [M, K] · b [K, N]` per segment `[start, end)` of the inner axis: `segments [..., 2]`
/// (integers) gives `[..., M, N]` (MLX's `segmented_mm`). An empty segment gives zeros.
pub(crate) fn segmented_mm(a: &Array, b: &Array, segments: &Array) -> Result<Array> {
    let name = "segmented_mm";
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(invalid(name, "batched matmul not supported"));
    }
    if segments.ndim() < 1 || segments.dims().last() != Some(&2) {
        return Err(invalid(
            name,
            format!("the segments must be [..., 2], got {:?}", segments.dims()),
        ));
    }
    if segments.dtype().category() == Category::Float || segments.dtype() == DType::Bool {
        return Err(invalid(name, "the segments must be integers"));
    }
    if a.dims()[1] != b.dims()[0] {
        return Err(invalid(
            name,
            format!("{:?} and {:?} do not multiply", a.dims(), b.dims()),
        ));
    }
    let dtype = floating(name, a, b)?;
    let (a, b) = (a.try_astype(dtype)?, b.try_astype(dtype)?);
    let op = crate::fast::imp::FastOp::SegmentedMm;
    let inputs = vec![a, b, segments.try_astype(DType::U32)?];
    Ok(crate::fast::imp::record_or_fallback(op, inputs)?.swap_remove(0))
}

/// [`segmented_mm`] composed: a mask of each segment's inner positions, then one batched
/// product (the fallback, and the derivatives).
pub(crate) fn segmented_compose(a: &Array, b: &Array, segments: &Array) -> Result<Array> {
    let k = a.dims()[1] as i32;
    // mask[..., k] = start ≤ k < end, then (a · mask) b per segment.
    let seg = segments.try_astype(DType::I32)?;
    let dtype = a.dtype();
    let start = seg.i((crate::Ellipsis, 0..1));
    let stop = seg.i((crate::Ellipsis, 1..2));
    let positions = ops::arange(0, k, 1);
    let mask = ops::logical_and(positions.greater_equal(&start), positions.less(&stop))
        .astype(dtype)
        .expand_dims(-2);
    a.multiply(&mask).matmul(b).into_result()
}

/// MLX's `block_masked_mm`: `a [..., M, K] · b [..., K, N]` with `block_size`-square masks
/// (32 or 64) on the output (`[..., ⌈M/bs⌉, ⌈N/bs⌉]`), the left operand (`[..., ⌈M/bs⌉,
/// ⌈K/bs⌉]`) and the right one (`[..., ⌈K/bs⌉, ⌈N/bs⌉]`): boolean masks select blocks,
/// float masks scale them. Without masks it is `matmul`.
pub(crate) fn block_masked_mm(
    a: &Array,
    b: &Array,
    block_size: i32,
    masks: [Option<&Array>; 3],
) -> Result<Array> {
    let name = "block_masked_mm";
    let [mask_out, mask_lhs, mask_rhs] = masks;
    if masks.iter().all(Option::is_none) {
        return a.matmul(b).into_result();
    }
    if block_size != 32 && block_size != 64 {
        return Err(invalid(
            name,
            format!("only block sizes 32 and 64 are supported, got {block_size}"),
        ));
    }
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(invalid(name, "the inputs need at least one dimension"));
    }
    let a = if a.ndim() == 1 {
        a.expand_dims(0)
    } else {
        a.clone()
    };
    let b = if b.ndim() == 1 {
        b.expand_dims(1)
    } else {
        b.clone()
    };
    let (m, k) = (a.dims()[a.ndim() - 2], a.dims()[a.ndim() - 1]);
    let (k2, n) = (b.dims()[b.ndim() - 2], b.dims()[b.ndim() - 1]);
    if k != k2 {
        return Err(invalid(
            name,
            format!("{:?} and {:?} do not multiply", a.dims(), b.dims()),
        ));
    }
    let dtype = floating(name, &a, &b)?;
    let (a, b) = (a.try_astype(dtype)?, b.try_astype(dtype)?);
    let bs = block_size as usize;
    let blocks = |v: usize| v.div_ceil(bs);
    // A block mask spread over its elements and cut to `rows × cols`.
    let expand = |mask: &Array, what: &str, rows: usize, cols: usize| -> Result<Array> {
        let nd = mask.ndim();
        if nd < 2 || mask.dims()[nd - 2] != blocks(rows) || mask.dims()[nd - 1] != blocks(cols) {
            return Err(invalid(
                name,
                format!(
                    "the {what} mask must be [..., {}, {}], got {:?}",
                    blocks(rows),
                    blocks(cols),
                    mask.dims()
                ),
            ));
        }
        let spread = ops::repeat(&ops::repeat(mask, block_size, -2), block_size, -1);
        let cut = spread.i((crate::Ellipsis, 0..rows, 0..cols));
        Ok(if mask.dtype() == DType::Bool {
            cut
        } else {
            cut.astype(dtype)
        })
    };
    let apply =
        |x: &Array, mask: Option<&Array>, what: &str, rows: usize, cols: usize| -> Result<Array> {
            Ok(match mask {
                None => x.clone(),
                Some(mask) => {
                    let e = expand(mask, what, rows, cols)?;
                    if e.dtype() == DType::Bool {
                        ops::where_(&e, x, 0.0).astype(dtype)
                    } else {
                        x.multiply(&e)
                    }
                }
            })
        };
    let a = apply(&a, mask_lhs, "left", m, k)?;
    let b = apply(&b, mask_rhs, "right", k, n)?;
    apply(&a.matmul(&b), mask_out, "output", m, n)?.into_result()
}

/// The weight of [`qqmm`](crate::ops::qqmm): dense (quantized with its mode for the
/// product) or already quantized.
#[derive(Clone, Copy, Debug)]
pub enum QqmmWeight<'a> {
    /// A float weight `[N, K]` and the mode to quantize it (and the activations) in.
    Dense(&'a Array, Quant),
    /// A weight quantized with a floating-point mode.
    Quantized(&'a Quantized),
}

impl<'a> From<&'a Array> for QqmmWeight<'a> {
    /// A dense weight in MLX's default mode, NVFP4.
    fn from(w: &'a Array) -> Self {
        Self::Dense(w, Quant::new(QuantMode::Nvfp4))
    }
}

impl<'a> From<(&'a Array, Quant)> for QqmmWeight<'a> {
    fn from((w, quant): (&'a Array, Quant)) -> Self {
        Self::Dense(w, quant)
    }
}

impl<'a> From<(&'a Array, QuantMode)> for QqmmWeight<'a> {
    fn from((w, mode): (&'a Array, QuantMode)) -> Self {
        Self::Dense(w, Quant::new(mode))
    }
}

impl<'a> From<&'a Quantized> for QqmmWeight<'a> {
    fn from(w: &'a Quantized) -> Self {
        Self::Quantized(w)
    }
}

fn qq_quant(name: &str, quant: Quant) -> Result<Quant> {
    if quant.mode == QuantMode::Affine {
        return Err(invalid(
            name,
            "needs a floating-point mode (mxfp4, mxfp8 or nvfp4)",
        ));
    }
    Ok(quant)
}

fn real_not_f64(name: &str, what: &str, x: &Array) -> Result<()> {
    if x.dtype().category() != Category::Float || x.dtype() == DType::F64 {
        return Err(invalid(
            name,
            format!(
                "the {what} must be float32, float16 or bfloat16, got {}",
                x.dtype()
            ),
        ));
    }
    Ok(())
}

/// MLX's `qqmm` product: its matrix-vector kernels for every row count (`dispatch_qmv`).
fn row_product(x: &Array, w: &Quantized) -> Result<Array> {
    crate::quantized::quantized_product(
        x,
        &w.weight,
        &w.scales,
        w.biases.as_ref(),
        true,
        w.quant,
        true,
    )
}

/// `x` through its quantization: the values `quant` represents, in `x`'s dtype.
fn fake_quant(x: &Array, quant: Quant) -> Result<Array> {
    ops::dequantize(&ops::quantize(x, quant), x.dtype()).into_result()
}

/// MLX's `qqmm` on a 2-D activation: `x_q · w_qᵀ` with both quantized in `quant`'s mode.
fn qqmm_2d(x: &Array, w: QqmmWeight<'_>) -> Result<Array> {
    let name = "qqmm";
    match w {
        QqmmWeight::Quantized(q) => {
            let quant = qq_quant(name, q.quant)?;
            row_product(&fake_quant(x, quant)?, q)
        }
        QqmmWeight::Dense(w, quant) => {
            let quant = qq_quant(name, quant)?;
            real_not_f64(name, "weight", w)?;
            // MLX's backward: the cotangent and the transposed operands quantized too.
            let f = custom_function_flat(move |p: &[Array]| {
                let wq = ops::quantize(&p[1], quant);
                Ok(vec![row_product(&fake_quant(&p[0], quant)?, &wq)?])
            })
            .with_vjp(move |p: &[Array], _o: &[Array], c: &[Array]| {
                let wt = ops::swapaxes(&p[1], -1, -2);
                let xt = ops::swapaxes(&p[0], -1, -2);
                let ct = ops::swapaxes(&c[0], -1, -2);
                Ok(vec![
                    qqmm_2d(&c[0], QqmmWeight::Dense(&wt, quant))?,
                    qqmm_2d(&ct, QqmmWeight::Dense(&xt, quant))?,
                ])
            });
            Ok(f.call(&[x.clone(), w.clone()])?.remove(0))
        }
    }
}

/// MLX's `qqmm`: `x` (`[..., K]`, a vector or a batch of rows) times `wᵀ` (`[N, K]`) with
/// both quantized in a floating-point mode.
pub(crate) fn qqmm(x: &Array, w: QqmmWeight<'_>) -> Result<Array> {
    real_not_f64("qqmm", "activation", x)?;
    let (lead, x2) = match x.ndim() {
        0 => {
            return Err(invalid(
                "qqmm",
                "the activation needs at least one dimension",
            ));
        }
        1 => (None, x.expand_dims(0)),
        2 => (Some(x.dims()[..1].to_vec()), x.clone()),
        _ => {
            let k = *x.dims().last().expect("an axis");
            (
                Some(x.dims()[..x.ndim() - 1].to_vec()),
                x.reshape([-1, k as i32]),
            )
        }
    };
    let y = qqmm_2d(&x2, w)?;
    let n = *y.dims().last().expect("an axis");
    Ok(match lead {
        None => y.reshape([n as i32]),
        Some(mut lead) => {
            lead.push(n);
            y.reshape(&lead[..])
        }
    })
}

/// MLX's `gather_qqmm`: [`gather_qmm`](crate::ops::gather_qmm) of the quantized
/// activations with quantized experts (`w [E, N, K]`, dense or quantized), transposed.
pub(crate) fn gather_qqmm(
    x: &Array,
    w: QqmmWeight<'_>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let name = "gather_qqmm";
    real_not_f64(name, "activation", x)?;
    let owned;
    let (q, quant) = match w {
        QqmmWeight::Quantized(q) => (q, qq_quant(name, q.quant)?),
        QqmmWeight::Dense(w, quant) => {
            let quant = qq_quant(name, quant)?;
            real_not_f64(name, "weight", w)?;
            owned = ops::quantize(w, quant);
            (&owned, quant)
        }
    };
    let options = GatherQmm::default().sorted_indices(sorted_indices);
    ops::gather_qmm(&fake_quant(x, quant)?, q, lhs_indices, rhs_indices, options).into_result()
}
