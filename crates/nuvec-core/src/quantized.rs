//! Quantization, as in MLX: weights stored with a few bits per value and one scale per group
//! of values along the last axis.
//!
//! - **Affine** (`bits` = 1–6 or 8, groups of 32, 64 or 128): a value `w` is stored as
//!   `round((w - bias) / scale)`, clamped to `[0, 2^bits - 1]`; scales and biases keep the
//!   weights' dtype.
//! - **Microscaling floating point**, no biases: `mxfp4` (E2M1 values, groups of 32, an E8M0
//!   scale), `mxfp8` (E4M3 values, groups of 32, an E8M0 scale) and `nvfp4` (E2M1 values,
//!   groups of 16, an E4M3 scale). Scales are uint8.
//!
//! The packed form is MLX's, so weights quantized by either library load in the other:
//! codes form a little-endian bit stream per row, stored as uint32.

use std::fmt;
use std::str::FromStr;

use crate::ops::imp::{astype, broadcast_to, expand_dims, multiply, reshape, squeeze};
use crate::primitive::Primitive;
use crate::{Array, Category, DType, Error, Result, promote_types, view};

/// A quantization mode (MLX's `mode` strings: `"affine"`, `"mxfp4"`, `"mxfp8"`, `"nvfp4"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantMode {
    /// `bits` per value with a scale and a bias per group (MLX's default).
    #[default]
    Affine,
    /// E2M1 values in groups of 32 with an E8M0 (power-of-two) scale.
    Mxfp4,
    /// E4M3 values in groups of 32 with an E8M0 scale.
    Mxfp8,
    /// E2M1 values in groups of 16 with an E4M3 scale.
    Nvfp4,
}

impl QuantMode {
    /// MLX's name of the mode.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// The default group size and bit width (MLX's `quantization_params_from_mode`).
    const fn defaults(self) -> (i32, i32) {
        match self {
            Self::Affine => (64, 4),
            Self::Mxfp4 => (32, 4),
            Self::Mxfp8 => (32, 8),
            Self::Nvfp4 => (16, 4),
        }
    }
}

impl fmt::Display for QuantMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for QuantMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "affine" => Self::Affine,
            "mxfp4" => Self::Mxfp4,
            "mxfp8" => Self::Mxfp8,
            "nvfp4" => Self::Nvfp4,
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "invalid quantization mode '{s}' (affine, mxfp4, mxfp8 or nvfp4)"
                )));
            }
        })
    }
}

/// How weights are quantized: a mode, a group size and a bit width. [`Quant::new`] gives
/// MLX's defaults for a mode; an unsupported combination fails in the operation that uses it
/// (a failed array, as every operation's errors).
///
/// ```
/// use nuvec_core::{Quant, QuantMode};
///
/// assert_eq!(Quant::new(QuantMode::Mxfp4), Quant { mode: QuantMode::Mxfp4, group_size: 32, bits: 4 });
/// assert_eq!(Quant::affine(32, 8).bits, 8);
/// assert_eq!(Quant::default(), Quant::affine(64, 4));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Quant {
    /// The mode.
    pub mode: QuantMode,
    /// Values per scale (affine: 32, 64 or 128; mxfp4 and mxfp8: 32; nvfp4: 16).
    pub group_size: i32,
    /// Bits per value (affine: 1–6 or 8; mxfp4 and nvfp4: 4; mxfp8: 8).
    pub bits: i32,
}

impl Quant {
    /// `mode` with MLX's default group size and bits: affine 64/4, mxfp4 32/4, mxfp8 32/8,
    /// nvfp4 16/4.
    pub const fn new(mode: QuantMode) -> Self {
        let (group_size, bits) = mode.defaults();
        Self {
            mode,
            group_size,
            bits,
        }
    }

    /// Affine quantization with `group_size` values per scale and bias and `bits` per value.
    pub const fn affine(group_size: i32, bits: i32) -> Self {
        Self {
            mode: QuantMode::Affine,
            group_size,
            bits,
        }
    }

    /// This quantization with `group_size` values per scale.
    #[must_use]
    pub const fn group_size(self, group_size: i32) -> Self {
        Self { group_size, ..self }
    }

    /// This quantization with `bits` bits per value.
    #[must_use]
    pub const fn bits(self, bits: i32) -> Self {
        Self { bits, ..self }
    }

    /// The group size and bit width, after MLX's checks for the mode.
    pub(crate) fn checked(self, op: &str) -> Result<(usize, usize)> {
        let invalid = |msg: String| Err(Error::InvalidArgument(format!("{op}: {msg}")));
        let (group_size, bits) = (self.group_size, self.bits);
        if self.mode == QuantMode::Affine {
            if !matches!(group_size, 32 | 64 | 128) {
                return invalid(format!(
                    "group size {group_size} is not supported (32, 64 or 128)"
                ));
            }
            if !matches!(bits, 1..=6 | 8) {
                return invalid(format!(
                    "{bits} bits are not supported (1, 2, 3, 4, 5, 6 or 8)"
                ));
            }
        } else {
            let (expected_group, expected_bits) = self.mode.defaults();
            if group_size != expected_group {
                return invalid(format!(
                    "{} quantization requires group size {expected_group} but got {group_size}",
                    self.mode
                ));
            }
            if bits != expected_bits {
                return invalid(format!(
                    "{} quantization requires bits to be {expected_bits} but got {bits}",
                    self.mode
                ));
            }
        }
        Ok((group_size as usize, bits as usize))
    }
}

impl Default for Quant {
    /// Affine, 64 values per group, 4 bits.
    fn default() -> Self {
        Self::new(QuantMode::Affine)
    }
}

impl From<QuantMode> for Quant {
    fn from(mode: QuantMode) -> Self {
        Self::new(mode)
    }
}

/// Quantized weights: packed codes (uint32), one scale per group (affine: the weights'
/// dtype; floating-point modes: uint8 codes), affine biases, and how they were quantized.
///
/// ```
/// use nuvec_core::{DType, QuantMode, ops::ones};
///
/// let w = ones([2, 64], DType::F32);
/// let q = w.quantize(QuantMode::Mxfp4);
/// assert_eq!((q.weight.shape(), q.scales.shape(), q.biases.is_none()), (&[2, 8][..], &[2, 2][..], true));
/// assert_eq!(q.dequantize(DType::F32).to_vec::<f32>().unwrap(), vec![1.0; 128]);
/// ```
#[derive(Clone, Debug)]
pub struct Quantized {
    /// The packed codes (uint32, last dimension `values · bits / 32`).
    pub weight: Array,
    /// One scale per group (last dimension `values / group_size`).
    pub scales: Array,
    /// One bias per group: required for affine, absent for the floating-point modes.
    pub biases: Option<Array>,
    /// How the weights were quantized.
    pub quant: Quant,
}

impl Quantized {
    /// Affine weights quantized elsewhere: the biases are required, as MLX requires them
    /// (a checkpoint without biases quantized with zero offsets: pass
    /// `ops::zeros_like(&scales)`).
    pub fn affine(weight: Array, scales: Array, biases: Array, group_size: i32, bits: i32) -> Self {
        Self::new(
            weight,
            scales,
            Some(biases),
            Quant::affine(group_size, bits),
        )
    }

    /// Floating-point-mode weights quantized elsewhere (MXFP4, MXFP8, NVFP4), which have no
    /// biases.
    pub fn fp(weight: Array, scales: Array, mode: QuantMode) -> Self {
        Self::new(weight, scales, None, Quant::new(mode))
    }

    /// Weights quantized elsewhere (a checkpoint's tensors). `biases` must be `Some` for
    /// [`QuantMode::Affine`] and `None` for the floating-point modes, as in MLX; otherwise
    /// every operation on them fails with an error saying so. [`Quantized::affine`] and
    /// [`Quantized::fp`] make the rule part of the signature.
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        quant: impl Into<Quant>,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            quant: quant.into(),
        }
    }

    /// The weights restored (MLX's `dequantize`), in `dtype` (default: the scales' dtype for
    /// affine, bfloat16 for the floating-point modes).
    #[track_caller]
    pub fn dequantize(&self, dtype: impl Into<Option<DType>>) -> Array {
        crate::ops::dequantize(self, dtype)
    }

    /// The first error any of the arrays carries.
    pub fn check(&self) -> Result<()> {
        for a in self.arrays() {
            let _ = a.clone().check()?;
        }
        Ok(())
    }

    /// The arrays, in MLX's operand order: weight, scales and (affine) biases.
    pub(crate) fn arrays(&self) -> impl Iterator<Item = &Array> {
        [&self.weight, &self.scales].into_iter().chain(&self.biases)
    }
}

impl crate::tree::Static for QuantMode {
    fn hash_static(&self, mut h: &mut dyn std::hash::Hasher) {
        std::hash::Hash::hash(self, &mut h);
    }
}

impl crate::tree::Static for Quant {
    fn hash_static(&self, mut h: &mut dyn std::hash::Hasher) {
        std::hash::Hash::hash(self, &mut h);
    }
}

/// A tree of buffers (never trainable, as quantized layers' tensors) with MLX's names
/// `weight`, `scales` and `biases`, and `quant` as a static field: `eval(&q)`, the
/// transforms and `save_safetensors` take quantized weights directly.
impl crate::tree::Leaves for Quantized {
    fn visit(&self, v: &mut crate::tree::Visit<'_>) {
        v.buffer("weight", &self.weight);
        v.buffer("scales", &self.scales);
        v.buffer("biases", &self.biases);
        v.static_field("quant", &self.quant);
    }
}

impl crate::tree::Tree for Quantized {
    fn visit_mut(&mut self, v: &mut crate::tree::VisitMut<'_>) {
        v.buffer("weight", &mut self.weight);
        v.buffer("scales", &mut self.scales);
        v.buffer("biases", &mut self.biases);
    }
}

/// Options of [`gather_qmm`](crate::ops::gather_qmm), built with setters.
/// `GatherQmm::default()` multiplies by the transposed weights (MLX's default).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GatherQmm<'a> {
    /// `x · wᵀ` (MLX's default) rather than `x · w`.
    pub transpose: bool,
    /// The right indices are sorted (rows grouped by expert), so one kernel can walk each
    /// expert's weights once.
    pub sorted_indices: bool,
    /// nvfp4 only: one float32 scale per expert (`w.weight`'s batch shape), multiplying the
    /// expert's decoded weights by `global_scale / (448 · 6)`. Needs the right indices.
    pub global_scale: Option<&'a Array>,
}

impl Default for GatherQmm<'_> {
    fn default() -> Self {
        Self {
            transpose: true,
            sorted_indices: false,
            global_scale: None,
        }
    }
}

impl<'a> GatherQmm<'a> {
    /// Whether a gather of `rows` sorted index rows over `experts` weight batches takes the
    /// grouped kernel (each expert's rows in tiles), rather than one product per row. Two
    /// gathers that take the same kernel give a row the same bits, whatever other rows are
    /// gathered with it: a caller batching gathers keeps each one's bits when its rows and
    /// the batch's agree here.
    pub fn groups(rows: usize, experts: usize) -> bool {
        grouped(rows, experts)
    }

    /// Whether the weights are used transposed.
    #[must_use]
    pub fn transpose(self, transpose: bool) -> Self {
        Self { transpose, ..self }
    }

    /// Whether the right indices are sorted.
    #[must_use]
    pub fn sorted_indices(self, sorted_indices: bool) -> Self {
        Self {
            sorted_indices,
            ..self
        }
    }

    /// The per-expert nvfp4 global scale.
    #[must_use]
    pub fn global_scale(self, global_scale: &'a Array) -> Self {
        Self {
            global_scale: Some(global_scale),
            ..self
        }
    }
}

/// Checks the scales (and biases) against the mode, as MLX's `validate_mode_with_type`:
/// affine needs float biases, the floating-point modes uint8 scales and no biases.
fn check_groups(op: &str, mode: QuantMode, scales: &Array, biases: Option<&Array>) -> Result<()> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("{op}: {msg}")));
    match (mode, biases) {
        (QuantMode::Affine, None) => {
            invalid("biases must be provided for affine quantization".into())
        }
        (QuantMode::Affine, Some(_)) => Ok(()),
        (_, Some(_)) => invalid(format!(
            "biases must be null for quantization mode '{mode}'"
        )),
        (_, None) if scales.dtype() != DType::U8 => invalid(format!(
            "scale type must be uint8 but received type {}",
            scales.dtype()
        )),
        (_, None) => Ok(()),
    }
}

/// Quantizes the last axis of `w` (float16, bfloat16 or float32, at least 2-D, the last
/// dimension divisible by the group size). Returns the packed codes (uint32), the scales and,
/// for affine, the biases, equal to MLX's bit for bit (the floating-point modes: MLX's Metal
/// quantizer's).
pub(crate) fn quantize(w: &Array, quant: Quant) -> Result<(Array, Array, Option<Array>)> {
    quant.checked("quantize")?;
    let mut out = Array::from_op(Primitive::Quantize { quant }, vec![w.clone()])?;
    let biases = (quant.mode == QuantMode::Affine).then(|| out.pop().expect("three outputs"));
    let scales = out.pop().expect("scales");
    Ok((out.pop().expect("codes"), scales, biases))
}

/// Restores the values packed by [`quantize`], in `dtype` (default: the scales' dtype for
/// affine, bfloat16 for the floating-point modes).
pub(crate) fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    quant: Quant,
    dtype: Option<DType>,
) -> Result<Array> {
    quant.checked("dequantize")?;
    check_groups("dequantize", quant.mode, scales, biases)?;
    if let Some(d) = dtype
        && d.category() != Category::Float
    {
        return Err(Error::InvalidArgument(format!(
            "dequantize: only real floating types are supported but the output dtype is {d}"
        )));
    }
    match biases {
        Some(b) => {
            let out = Array::from_op(
                Primitive::Dequantize {
                    quant,
                    dtype: scales.dtype(),
                },
                vec![w.clone(), scales.clone(), b.clone()],
            )?
            .swap_remove(0);
            match dtype {
                Some(d) if d != out.dtype() => astype(&out, d),
                _ => Ok(out),
            }
        }
        None => {
            let dtype = dtype.unwrap_or(DType::BF16);
            if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
                return Err(Error::unsupported_dtype("dequantize", dtype));
            }
            let primitive = Primitive::Dequantize { quant, dtype };
            Ok(Array::from_op(primitive, vec![w.clone(), scales.clone()])?.swap_remove(0))
        }
    }
}

/// The matrix product of `x` with quantized weights, `x · dequantize(w)ᵀ` if `transpose`
/// (MLX's default, for weights stored as `[out, in]`) or `x · dequantize(w)`, without
/// materialising the dequantized matrix.
///
/// As in MLX, affine promotes `x`, `scales` and `biases` to one floating-point type; the
/// floating-point modes keep `x`'s dtype. Batch dimensions broadcast when `x` and `w` both
/// have more than two. A one-dimensional `x` is a row vector.
pub(crate) fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    quant: Quant,
) -> Result<Array> {
    quant.checked("quantized_matmul")?;
    check_groups("quantized_matmul", quant.mode, scales, biases)?;
    let dtype = match biases {
        Some(b) => promote_types(promote_types(x.dtype(), scales.dtype()), b.dtype()),
        None => x.dtype(),
    };
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::unsupported_dtype("quantized_matmul", dtype));
    }
    let x = astype(x, dtype)?;
    let mut groups = vec![w.clone()];
    match biases {
        Some(b) => groups.extend([astype(scales, dtype)?, astype(b, dtype)?]),
        None => groups.push(scales.clone()),
    }
    let vector = x.ndim() == 1;
    let mut x = if vector { expand_dims(&x, 0)? } else { x };
    let (xn, wn) = (x.ndim(), w.ndim());
    if wn > 2 && xn >= 2 {
        // Broadcast the batch dimensions of every operand together (MLX broadcasts them only
        // when both have batch dimensions; nuvec also lifts a 2-D `x` to `w`'s batch).
        let batch =
            view::broadcast_shapes("quantized_matmul", &x.dims()[..xn - 2], &w.dims()[..wn - 2])?;
        let full = |a: &Array| -> Result<Array> {
            let mut shape = batch.to_vec();
            shape.extend_from_slice(&a.dims()[a.ndim() - 2..]);
            broadcast_to(a, &shape)
        };
        x = full(&x)?;
        for g in &mut groups {
            *g = full(g)?;
        }
    }
    let mut inputs = vec![x];
    inputs.extend(groups);
    let primitive = Primitive::QuantizedMatmul { quant, transpose };
    let out = Array::from_op(primitive, inputs)?.swap_remove(0);
    if vector {
        squeeze(&out, &[-2])
    } else {
        Ok(out)
    }
}

/// The operands of a recorded [`FastOp::GatherQmm`](crate::fast::imp::FastOp::GatherQmm):
/// `(x, w, scales[, biases][, global_scale], lhs_indices, rhs_indices[, offsets])`.
pub(crate) struct GatherOperands<'a> {
    pub x: &'a Array,
    pub w: &'a Array,
    pub scales: &'a Array,
    pub biases: Option<&'a Array>,
    pub global_scale: Option<&'a Array>,
    pub lhs: &'a Array,
    pub rhs: &'a Array,
    pub offsets: Option<&'a Array>,
}

impl<'a> GatherOperands<'a> {
    /// The operand count without the optional offsets.
    pub(crate) fn arity(mode: QuantMode, global_scale: bool) -> usize {
        5 + usize::from(mode == QuantMode::Affine) + usize::from(global_scale)
    }

    /// Splits `inputs` (`arity` or `arity + 1` arrays, checked by the caller).
    pub(crate) fn split(inputs: &'a [Array], mode: QuantMode, global_scale: bool) -> Self {
        let mut i = 3;
        let mut next = |present: bool| {
            present.then(|| {
                i += 1;
                &inputs[i - 1]
            })
        };
        let biases = next(mode == QuantMode::Affine);
        let global_scale = next(global_scale);
        let (lhs, rhs) = (&inputs[i], &inputs[i + 1]);
        Self {
            x: &inputs[0],
            w: &inputs[1],
            scales: &inputs[2],
            biases,
            global_scale,
            lhs,
            rhs,
            offsets: inputs.get(i + 2),
        }
    }
}

/// Quantized matrix products of gathered batches, as MLX's `gather_qmm` (mixture-of-experts
/// layers): output batch `i` is `x[lhs_indices[i]] · dequantize(w)[rhs_indices[i]]ᵀ` (or
/// without the transpose), indexing the flattened batch dimensions of `x` (`[..., M, K]`)
/// and of the weights. The indices (integers) broadcast together; either defaults to every
/// batch of its operand in order. Without indices this is [`quantized_matmul`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    quant: Quant,
    options: GatherQmm<'_>,
) -> Result<Array> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("gather_qmm: {msg}")));
    quant.checked("gather_qmm")?;
    if lhs_indices.is_none() && rhs_indices.is_none() {
        if options.global_scale.is_some() {
            return invalid("global scale is not supported without indices".into());
        }
        return quantized_matmul(x, w, scales, biases, options.transpose, quant);
    }
    check_groups("gather_qmm", quant.mode, scales, biases)?;
    let dtype = match biases {
        Some(b) => promote_types(promote_types(x.dtype(), scales.dtype()), b.dtype()),
        None => x.dtype(),
    };
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::unsupported_dtype("gather_qmm", dtype));
    }
    if x.ndim() < 2 || w.ndim() < 2 || biases.is_some_and(|b| scales.dims() != b.dims()) {
        return invalid(format!(
            "x must be [..., M, K] and w [..., N, K'], got {:?} and {:?}",
            x.dims(),
            w.dims()
        ));
    }
    if let Some(gs) = options.global_scale {
        if quant.mode != QuantMode::Nvfp4 {
            return invalid("global scale is only supported for 'nvfp4' quantization mode".into());
        }
        if gs.dtype() != DType::F32 {
            return invalid(format!(
                "global scale must have dtype float32 but got {}",
                gs.dtype()
            ));
        }
        let experts = &w.dims()[..w.ndim() - 2];
        if gs.dims() != experts {
            return invalid(format!(
                "global scale must have one entry per expert with shape {experts:?} but got {:?}",
                gs.dims()
            ));
        }
    }
    let x = astype(x, dtype)?;
    let (lhs, rhs) = gather_indices("gather_qmm", &x, w, lhs_indices, rhs_indices)?;
    let mut inputs = vec![x.clone(), w.clone()];
    match biases {
        Some(b) => inputs.extend([astype(scales, dtype)?, astype(b, dtype)?]),
        None => inputs.push(scales.clone()),
    }
    inputs.extend(options.global_scale.cloned());
    inputs.extend([lhs, rhs.clone()]);
    // MLX's rule (`GatherQMM::eval_gpu`): the sorted kernel tiles each expert's rows, so it
    // pays only with at least 16 rows and 4 per expert on average; fewer take one product
    // per row. nuvec also keeps one product per row up to `ROW_INDEPENDENT_ROWS` tokens of
    // top-8 routing, so rows stay independent (the row-independence contract).
    let experts: usize = w.dims()[..w.ndim() - 2].iter().product();
    let sorted = lhs_indices.is_none() && options.sorted_indices && grouped(rhs.size(), experts);
    if let Some(offsets) = sorted_offsets(&x, w, &rhs, sorted)? {
        inputs.push(offsets);
    }
    let op = crate::fast::imp::FastOp::GatherQmm {
        quant,
        transpose: options.transpose,
        global_scale: options.global_scale.is_some(),
    };
    // Affine gathers have kernels for transposed weights only.
    let kernel = (options.transpose || quant.mode != QuantMode::Affine)
        && crate::fast::imp::on_gpu(dtype, x.size());
    // With a global scale the primitive runs on the CPU too (its fallback): it refuses the
    // global scale's derivative, as MLX's `GatherQMM`, where the graph would differentiate
    // through it.
    let out = if kernel || options.global_scale.is_some() {
        Array::from_op(Primitive::Fast(op), inputs)?.swap_remove(0)
    } else {
        op.fallback(&inputs)?.swap_remove(0)
    };
    Ok(out)
}

/// The gathered product of [`FastOp::GatherQmm`](crate::fast::imp::FastOp::GatherQmm) as
/// a graph: each operand's gathered batches, then one batched product. A global scale
/// multiplies each expert's decoded weights by `global_scale / (448 · 6)` (MLX's point, per
/// group before the dot product).
pub(crate) fn gather_qmm_graph(
    inputs: &[Array],
    quant: Quant,
    transpose: bool,
    global_scale: bool,
) -> Result<Array> {
    let ops = GatherOperands::split(inputs, quant.mode, global_scale);
    let rows = |a: &Array, indices: &Array| -> Result<Array> {
        let nd = a.ndim();
        let mut shape = vec![-1];
        shape.extend(a.dims()[nd.saturating_sub(2)..].iter().map(|&d| d as isize));
        crate::indexing::take(&reshape(a, &shape)?, indices, Some(0))
    };
    let (x, w, s) = (
        rows(ops.x, ops.lhs)?,
        rows(ops.w, ops.rhs)?,
        rows(ops.scales, ops.rhs)?,
    );
    let Some(gs) = ops.global_scale else {
        let b = ops.biases.map(|b| rows(b, ops.rhs)).transpose()?;
        return quantized_matmul(&x, &w, &s, b.as_ref(), transpose, quant);
    };
    let factor = crate::ops::imp::divide(
        &rows(&expand_dims(&expand_dims(gs, -1)?, -1)?, ops.rhs)?,
        crate::fp8::E4M3_MAX * crate::fp8::E2M1_MAX,
    )?;
    let dense = multiply(&dequantize(&w, &s, None, quant, Some(DType::F32))?, &factor)?;
    let dense = if transpose {
        let n = dense.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        crate::ops::imp::transpose(&dense, &axes)?
    } else {
        dense
    };
    let product = crate::ops::imp::matmul(&astype(&x, DType::F32)?, &dense)?;
    astype(&product, x.dtype())
}

/// Whether a gather with `rows` index rows over `experts` weight batches takes the grouped
/// (sorted) kernel: MLX's rule, at least 16 rows and 4 per expert; row-exact
/// ([`crate::with_row_exact`]), more rows than the row-independent region (`ROW_TOKENS`).
pub(crate) fn grouped(rows: usize, experts: usize) -> bool {
    let least = if crate::row_exact() {
        ROW_TOKENS + 1
    } else {
        16
    };
    rows >= least && rows / experts.max(1) >= 4
}

/// Gathered rows up to which every row runs as its own product: `ROW_INDEPENDENT_ROWS`
/// tokens with up to 8 experts each.
pub(crate) const ROW_TOKENS: usize = crate::ROW_INDEPENDENT_ROWS * 8;

/// The batch indices of a gathered product, broadcast together; either defaults to every
/// batch of its operand, in order.
fn gather_indices(
    op: &'static str,
    a: &Array,
    b: &Array,
    lhs: Option<&Array>,
    rhs: Option<&Array>,
) -> Result<(Array, Array)> {
    let indices = |x: &Array, given: Option<&Array>| -> Result<Array> {
        let batch = &x.dims()[..x.ndim() - 2];
        match given {
            Some(i)
                if i.dtype().category() == Category::Float || i.dtype() == crate::DType::Bool =>
            {
                Err(Error::InvalidArgument(format!(
                    "{op}: indices must be integers"
                )))
            }
            Some(i) => astype(i, crate::DType::U32),
            None => {
                let n: usize = batch.iter().product();
                let all = crate::ops::imp::arange(0.0, n as f64, 1.0, crate::DType::U32)?;
                all.try_reshape(&batch.iter().map(|&d| d as isize).collect::<Vec<_>>())
            }
        }
    };
    let (l, r) = (indices(a, lhs)?, indices(b, rhs)?);
    let batch = view::broadcast_shapes(op, l.dims(), r.dims())?;
    Ok((broadcast_to(&l, &batch)?, broadcast_to(&r, &batch)?))
}

/// For rows sorted by weight batch (`a` is `[rows, 1, K]`, one row per output batch), where
/// each batch's rows start (MLX's `gather_mm_offsets`, one dispatch; MLX's grouped kernels
/// walk every weight batch once).
fn sorted_offsets(a: &Array, b: &Array, rhs: &Array, sorted: bool) -> Result<Option<Array>> {
    let m = a.dims()[a.ndim() - 2];
    if !(sorted && m == 1 && a.ndim() == 3 && rhs.ndim() == 1) {
        return Ok(None);
    }
    let groups: usize = b.dims()[..b.ndim() - 2].iter().product();
    let primitive = crate::primitive::Primitive::SortedOffsets { groups };
    Ok(Some(
        Array::from_op(primitive, vec![rhs.clone()])?.swap_remove(0),
    ))
}

/// Matrix products of gathered batches, as MLX's `gather_mm` (mixture-of-experts layers in
/// full precision): output batch `i` is `a[lhs_indices[i]] · b[rhs_indices[i]]` over the
/// flattened batch dimensions of `a` (`[..., M, K]`) and `b` (`[..., K, N]`), with the
/// indices broadcast together and either defaulting to every batch. `sorted_indices`
/// promises sorted `rhs_indices`, which lets one kernel walk each batch of `b` once.
pub(crate) fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let dtype = promote_types(a.dtype(), b.dtype());
    if dtype.category() != Category::Float {
        return Err(Error::unsupported_dtype("gather_mm", dtype));
    }
    if a.ndim() < 2 || b.ndim() < 2 || a.dims()[a.ndim() - 1] != b.dims()[b.ndim() - 2] {
        return Err(Error::InvalidArgument(format!(
            "gather_mm: a must be [..., M, K] and b [..., K, N], got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let (a, b) = (astype(a, dtype)?, astype(b, dtype)?);
    let (lhs, rhs) = gather_indices("gather_mm", &a, &b, lhs_indices, rhs_indices)?;
    let mut inputs = vec![a.clone(), b.clone(), lhs, rhs.clone()];
    let grouped = rhs.size() > ROW_TOKENS;
    if let Some(offsets) = sorted_offsets(
        &a,
        &b,
        &rhs,
        lhs_indices.is_none() && sorted_indices && grouped,
    )? {
        inputs.push(offsets);
    }
    Ok(
        crate::fast::imp::record_or_fallback(crate::fast::imp::FastOp::GatherMm, inputs)?
            .swap_remove(0),
    )
}
