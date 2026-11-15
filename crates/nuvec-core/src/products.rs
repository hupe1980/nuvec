//! MLX's special matrix products: `segmented_mm` (one product per segment of the inner
//! axis), `block_masked_mm` (block masks on the operands and the output) and the
//! products of quantized activations and weights, `qqmm` and `gather_qqmm`.

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
/// float masks scale them. Without masks it is `matmul`. MLX's masked GEMM, which skips
/// masked blocks; the operands, masks and batches broadcast as MLX's.
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
    let check = |mask: &Array, what: &str, rows: usize, cols: usize| -> Result<()> {
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
        Ok(())
    };
    let batch_of = |x: &Array| x.dims()[..x.ndim() - 2].to_vec();
    let mut batch = crate::graph::broadcast_shapes(name, &batch_of(&a), &batch_of(&b))?;
    for (mask, what, rows, cols) in [
        (mask_out, "output", m, n),
        (mask_lhs, "left", m, k),
        (mask_rhs, "right", k, n),
    ] {
        if let Some(mask) = mask {
            check(mask, what, rows, cols)?;
            batch = crate::graph::broadcast_shapes(name, &batch, &batch_of(mask))?;
        }
    }
    let full = |x: &Array, rows: usize, cols: usize| -> Result<Array> {
        let mut shape = batch.clone();
        shape.extend([rows, cols]);
        ops::broadcast_to(x, shape.as_slice()).into_result()
    };
    // Boolean masks stay boolean; the others take the output's dtype (MLX's casts). The
    // operand masks share one type: a missing one is ones.
    let typed = |mask: &Array| -> Result<Array> {
        match mask.dtype() {
            DType::Bool => Ok(mask.clone()),
            _ => mask.try_astype(dtype),
        }
    };
    let mut inputs = vec![full(&a, m, k)?, full(&b, k, n)?];
    if let Some(mask) = mask_out {
        inputs.push(full(&typed(mask)?, blocks(m), blocks(n))?);
    }
    let op_mask = mask_lhs.is_some() || mask_rhs.is_some();
    if op_mask {
        let kind = match [mask_lhs, mask_rhs]
            .iter()
            .flatten()
            .all(|m| m.dtype() == DType::Bool)
        {
            true => DType::Bool,
            false => dtype,
        };
        for (mask, rows, cols) in [(mask_lhs, m, k), (mask_rhs, k, n)] {
            let mask = match mask {
                Some(mask) => mask.try_astype(kind)?,
                None => ops::ones([blocks(rows), blocks(cols)].as_slice(), kind),
            };
            inputs.push(full(&mask, blocks(rows), blocks(cols))?);
        }
    }
    let op = crate::fast::imp::FastOp::BlockMaskedMm {
        block_size: bs,
        out_mask: mask_out.is_some(),
        op_mask,
    };
    Ok(crate::fast::imp::record_or_fallback(op, inputs)?.swap_remove(0))
}

/// [`block_masked_mm`] composed (the fallback, and the derivatives): the masks spread over
/// their blocks' elements, the operands masked, one product, the output masked. `inputs`
/// are [`FastOp::BlockMaskedMm`](crate::fast::imp::FastOp::BlockMaskedMm)'s.
pub(crate) fn block_masked_compose(
    inputs: &[Array],
    block_size: usize,
    out_mask: bool,
) -> Result<Array> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let dtype = a.dtype();
    let (m, k) = (a.dims()[a.ndim() - 2], a.dims()[a.ndim() - 1]);
    let n = b.dims()[b.ndim() - 1];
    let bs = block_size as i32;
    let apply = |x: &Array, mask: Option<&Array>, rows: usize, cols: usize| -> Array {
        let Some(mask) = mask else {
            return x.clone();
        };
        let spread = ops::repeat(&ops::repeat(mask, bs, -2), bs, -1);
        let cut = spread.i((crate::Ellipsis, 0..rows, 0..cols));
        match mask.dtype() {
            DType::Bool => ops::where_(&cut, x, 0.0).astype(dtype),
            _ => x.multiply(cut.astype(dtype)),
        }
    };
    let at = 2 + usize::from(out_mask);
    let (lhs, rhs) = (inputs.get(at), inputs.get(at + 1));
    let a = apply(a, lhs, m, k);
    let b = apply(b, rhs, k, n);
    let out = if out_mask { inputs.get(2) } else { None };
    apply(&a.matmul(&b), out, m, n).into_result()
}

/// The weight of [`qqmm`](crate::ops::qqmm): dense (quantized with its mode for the
/// product) or already quantized.
#[derive(Clone, Copy, Debug)]
pub enum QqmmWeight<'a> {
    /// A float weight `[N, K]` and the mode to quantize it (and the activations) in.
    Dense(&'a Array, Quant),
    /// A float weight `[N, K]` quantized to NVFP4 with this float32 global scale
    /// (MLX's `global_scale_w`).
    Scaled(&'a Array, &'a Array),
    /// A weight quantized with a floating-point mode (and, NVFP4, its global scale).
    Quantized(&'a Quantized),
}

impl QqmmWeight<'_> {
    /// The arrays the product reads.
    pub(crate) fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Dense(w, _) => vec![w],
            Self::Scaled(w, g) => vec![w, g],
            Self::Quantized(q) => q.arrays().collect(),
        }
    }
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

/// MLX's `qqmm` product: its matrix-vector kernels for every row count (`dispatch_qmv`),
/// or with global scales and weights quantized beforehand (`matrix`), its `qmm_t` from the
/// vector limit.
fn row_product(x: &Array, w: &Quantized, matrix: bool) -> Result<Array> {
    crate::quantized::quantized_product(
        x,
        &w.weight,
        &w.scales,
        w.biases.as_ref(),
        true,
        w.quant,
        !matrix,
        w.global_scale.as_ref(),
    )
}

/// `x` through its quantization: the values `quant` represents, in `x`'s dtype (MLX's
/// `fp_quantize_dequantize`, with an optional NVFP4 global scale).
fn fake_quant(x: &Array, quant: Quant, global_scale: Option<&Array>) -> Result<Array> {
    let op = crate::fast::imp::FastOp::QuantizeDequantize { quant };
    let mut inputs = vec![x.clone()];
    inputs.extend(global_scale.cloned());
    // Recorded on both devices: the CPU runs the kernel's arithmetic.
    Ok(Array::from_op(crate::graph::Primitive::Fast(op), inputs)?.swap_remove(0))
}

/// MLX's `qqmm` rule: NVFP4 global scales for both operands or neither.
fn qq_scales<'a>(
    name: &str,
    w: &QqmmWeight<'a>,
    global_scale_x: Option<&'a Array>,
) -> Result<Option<(&'a Array, &'a Array)>> {
    let gw = match w {
        QqmmWeight::Dense(..) => None,
        QqmmWeight::Scaled(_, g) => Some(*g),
        QqmmWeight::Quantized(q) => q.global_scale.as_ref(),
    };
    match (global_scale_x, gw) {
        (Some(x), Some(w)) => Ok(Some((x, w))),
        (None, None) => Ok(None),
        _ => Err(invalid(
            name,
            "for nvfp4, give global scales for both the activations and the weight, or neither",
        )),
    }
}

/// MLX's `qqmm` on a 2-D activation: `x_q · w_qᵀ` with both quantized in `quant`'s mode.
fn qqmm_2d(x: &Array, w: QqmmWeight<'_>, global_scale_x: Option<&Array>) -> Result<Array> {
    let name = "qqmm";
    let scales = qq_scales(name, &w, global_scale_x)?;
    let gx = scales.map(|(x, _)| x);
    match w {
        QqmmWeight::Quantized(q) => {
            let quant = qq_quant(name, q.quant)?;
            row_product(&fake_quant(x, quant, gx)?, q, scales.is_some())
        }
        QqmmWeight::Dense(w, quant) => dense_qqmm(x, w, qq_quant(name, quant)?, None),
        QqmmWeight::Scaled(w, gw) => {
            let gx = gx.expect("both scales");
            dense_qqmm(x, w, Quant::new(QuantMode::Nvfp4), Some((gx, gw)))
        }
    }
}

/// `qqmm` of a dense weight, quantized for the product, with MLX's backward
/// (`QQMatmul::vjp`): the cotangent and the transposed operands quantized too, the
/// cotangent's global scale its largest magnitude (float32) when global scales are given.
fn dense_qqmm(
    x: &Array,
    w: &Array,
    quant: Quant,
    scales: Option<(&Array, &Array)>,
) -> Result<Array> {
    real_not_f64("qqmm", "weight", w)?;
    let product = move |x: &Array, w: &Array, gx: Option<&Array>, gw: Option<&Array>| {
        let wq = match gw {
            Some(g) => ops::quantize_scaled(w, g),
            None => ops::quantize(w, quant),
        };
        row_product(&fake_quant(x, quant, gx)?, &wq, false)
    };
    let mut primals = vec![x.clone(), w.clone()];
    if let Some((gx, gw)) = scales {
        primals.extend([gx.clone(), gw.clone()]);
    }
    let f = custom_function_flat(move |p: &[Array]| {
        Ok(vec![product(&p[0], &p[1], p.get(2), p.get(3))?])
    })
    .with_vjp(move |p: &[Array], _o: &[Array], c: &[Array]| {
        let c = &c[0];
        let gc = (p.len() == 4).then(|| c.abs().max(..).astype(DType::F32));
        let wt = ops::swapaxes(&p[1], -1, -2);
        let xt = ops::swapaxes(&p[0], -1, -2);
        let ct = ops::swapaxes(c, -1, -2);
        let mut grads = vec![
            product(c, &wt, gc.as_ref(), p.get(3))?,
            product(&ct, &xt, gc.as_ref(), p.get(2))?,
        ];
        grads.extend(p[2..].iter().map(ops::zeros_like));
        Ok(grads)
    });
    Ok(f.call(&primals)?.remove(0))
}

/// MLX's `qqmm`: `x` (`[..., K]`, a vector or a batch of rows) times `wᵀ` (`[N, K]`) with
/// both quantized in a floating-point mode.
pub(crate) fn qqmm(x: &Array, w: QqmmWeight<'_>, global_scale_x: Option<&Array>) -> Result<Array> {
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
    let y = qqmm_2d(&x2, w, global_scale_x)?;
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
    global_scale_x: Option<&Array>,
) -> Result<Array> {
    let name = "gather_qqmm";
    real_not_f64(name, "activation", x)?;
    let scales = qq_scales(name, &w, global_scale_x)?;
    let mut q = match w {
        QqmmWeight::Quantized(q) => {
            qq_quant(name, q.quant)?;
            q.clone()
        }
        QqmmWeight::Dense(w, quant) => {
            real_not_f64(name, "weight", w)?;
            ops::quantize(w, qq_quant(name, quant)?)
        }
        QqmmWeight::Scaled(w, g) => {
            real_not_f64(name, "weight", w)?;
            ops::quantize_scaled(w, g)
        }
    };
    // MLX broadcasts the weight's one global scale to every expert.
    if let Some(g) = q.global_scale.take() {
        let experts = &q.weight.dims()[..q.weight.ndim() - 2];
        q.global_scale = Some(g.reshape([] as [i32; 0]).broadcast_to(experts));
    }
    let x = fake_quant(x, q.quant, scales.map(|(x, _)| x))?;
    let options = GatherQmm::default().sorted_indices(sorted_indices);
    ops::gather_qmm(&x, &q, lhs_indices, rhs_indices, options).into_result()
}
