//! Fused operations with dedicated GPU kernels, as MLX's `mx.fast`: [`rms_norm`],
//! [`layer_norm`], [`rope`], [`cross_entropy`], [`scaled_dot_product_attention`] and
//! [`gated_delta`] (mlx-lm's Gated DeltaNet kernels); nuvec's [`add_rms_norm`] (a residual
//! connection and its norm), [`norm_rope`] (norm, RoPE and a cache write) and
//! [`paged_attention`] with [`paged_write`] (a block-table KV cache); and, through
//! [`crate::Array::softmax`] and [`crate::Array::logsumexp`] over the last axis, MLX's
//! softmax and log-sum-exp kernels.
//!
//! Each operation is one GPU kernel (MLX's, or nuvec's own) with an equivalent graph of ordinary
//! operations, its *fallback*. As in MLX, the fallback runs on the CPU stream, and it
//! defines the forward-mode derivative and batching; reverse mode uses dedicated kernels
//! where MLX has them.
//!
//! Like every operation, these return arrays: invalid arguments give a failed array that
//! carries the error and the caller's location (see [`crate::Array::check`]).
//!
//! # Memory
//!
//! The norms, RoPE, `norm_rope`, cross entropy, softmax, log-sum-exp, the gated delta rule
//! and the paged kernels hold no scratch beyond their inputs and outputs on the GPU (the
//! fallback graphs on the CPU hold their intermediates). Attention's scratch depends on its
//! plan: see [`scaled_dot_product_attention`]. Rotation frequencies are computed per call,
//! never cached.
//!
//! ```
//! use nuvec_core::{array, fast};
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0]];
//! let y = fast::rms_norm(&x, None, 1e-5); // x / sqrt(mean(x²) + eps), per row
//! let v = y.to_vec::<f32>().unwrap();
//! assert!((v[0] - 1.0 / 2.5f32.sqrt()).abs() < 1e-5);
//! ```

pub(crate) mod imp;
pub(crate) mod plan;

use std::panic::Location;

use smallvec::SmallVec;

use crate::Array;

pub use crate::kernel::{KernelCall, KernelConfig, MetalKernel, TemplateArg};
pub use imp::{AttentionMask, FastOp, Rotation};

/// The arrays among required and optional operands, for failure reporting.
fn operands<'a>(
    required: &[&'a Array],
    optional: &[Option<&'a Array>],
) -> SmallVec<[&'a Array; 8]> {
    let mut out: SmallVec<[&Array; 8]> = SmallVec::from_slice(required);
    out.extend(optional.iter().flatten().copied());
    out
}

/// Root-mean-square normalisation over the last axis, `x / sqrt(mean(x²) + eps) · weight`,
/// as MLX's `fast.rms_norm`: statistics in float32, the result in the promoted type of `x`
/// and `weight` (1-D, the last axis's size).
#[track_caller]
pub fn rms_norm<'a>(x: &Array, weight: impl Into<Option<&'a Array>>, eps: f32) -> Array {
    let weight = weight.into();
    let ins = operands(&[x], &[weight]);
    Array::sticky("rms_norm", Location::caller(), &ins, || {
        imp::rms_norm(x, weight, eps)
    })
}

/// A residual connection followed by RMS normalisation: `h = x + residual` and
/// [`rms_norm`]`(h, weight, eps)`, returned as `(h, normed)`, from one GPU kernel. The
/// results equal `x + residual` then [`rms_norm`] exactly; a weight that would widen the
/// sum's dtype runs the two operations.
#[track_caller]
pub fn add_rms_norm<'a>(
    x: &Array,
    residual: &Array,
    weight: impl Into<Option<&'a Array>>,
    eps: f32,
) -> (Array, Array) {
    let weight = weight.into();
    let ins = operands(&[x, residual], &[weight]);
    let [h, normed] = Array::sticky_n("add_rms_norm", Location::caller(), &ins, || {
        let (h, n) = imp::add_rms_norm(x, residual, weight, eps)?;
        Ok([h, n])
    });
    (h, normed)
}

/// Layer normalisation over the last axis, `(x - mean) / sqrt(var + eps) · weight + bias`,
/// as MLX's `fast.layer_norm`: statistics in float32, the result in the promoted type.
#[track_caller]
pub fn layer_norm<'a>(
    x: &Array,
    weight: impl Into<Option<&'a Array>>,
    bias: impl Into<Option<&'a Array>>,
    eps: f32,
) -> Array {
    let weight = weight.into();
    let bias = bias.into();
    let ins = operands(&[x], &[weight, bias]);
    Array::sticky("layer_norm", Location::caller(), &ins, || {
        imp::layer_norm(x, weight, bias, eps)
    })
}

/// Rotary position embeddings, as MLX's `fast.rope`, of `x` (at least 3-D: `[B, ...,
/// T, D]`, positions along `T`): the first `dims` features (even) are rotated in pairs,
/// the two halves (or interleaved pairs, `traditional`) by angle `(offset + t) · scale ·
/// θᵢ`, with `θᵢ = base^(-2i/dims)` or `1 / freqs[i]`. Exactly one of `base` and `freqs`
/// is given. `offset` is an integer scalar or one per batch entry.
#[track_caller]
pub fn rope<'a>(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: impl Into<Option<&'a Array>>,
) -> Array {
    let freqs = freqs.into();
    let ins = operands(&[x, offset], &[freqs]);
    Array::sticky("rope", Location::caller(), &ins, || {
        let dims = usize::try_from(dims).map_err(|_| {
            crate::Error::InvalidArgument(format!("rope: dims must not be negative, got {dims}"))
        })?;
        imp::rope(x, dims, traditional, base, scale, offset, freqs)
    })
}

/// Cross entropy over the last axis of `logits` against integer class indices `targets`
/// (the logits' shape without the last axis; negative indices count from the end), as
/// MLX's `fast.cross_entropy`: `logsumexp(logits) - logits[target]` per row, in float32.
#[track_caller]
pub fn cross_entropy(logits: &Array, targets: &Array) -> Array {
    Array::sticky(
        "cross_entropy",
        Location::caller(),
        &[logits, targets],
        || imp::cross_entropy(logits, targets),
    )
}

/// An attention layer's query or key preparation in one kernel: each head's features of
/// `x` (`[B, H, L, D]`, any strides, as the transposed projection) optionally normalised
/// as [`rms_norm`] with `norm = (weight, eps)` (Qwen 3's `q_norm` and `k_norm`), then
/// rotated as [`rope`] at positions `offset..offset + L`.
///
/// Without `cache`, the result is a new `[B, H, L, D]` array. With `cache` (`[B, H, C, D]`,
/// `C ≥ offset + L`), the result is `cache` with positions `offset..offset + L` replaced
/// (as [`crate::Array::at`] with `set`, and written in place when the evaluation owns the
/// cache), so keys go into a key cache without a separate copy.
///
/// The results equal [`rms_norm`], [`rope`] and the update composed (on the GPU, bit for
/// bit); differing dtypes, other ranks and head sizes over 1024 run that composition.
#[track_caller]
pub fn norm_rope<'a>(
    x: &Array,
    norm: Option<(&Array, f32)>,
    rotation: Rotation<'_>,
    offset: i32,
    cache: impl Into<Option<&'a Array>>,
) -> Array {
    let cache = cache.into();
    let ins = operands(&[x], &[norm.map(|n| n.0), rotation.freqs, cache]);
    Array::sticky("norm_rope", Location::caller(), &ins, || {
        let offset = usize::try_from(offset).map_err(|_| {
            crate::Error::InvalidArgument(format!(
                "norm_rope: offset must not be negative, got {offset}"
            ))
        })?;
        imp::norm_rope(x, norm, rotation, offset, cache)
    })
}

/// Attention over a paged KV cache, for a batch of sequences of different lengths.
///
/// Keys and values live in blocks of positions, `k_cache` and `v_cache` being
/// `[blocks, kv_heads, block_size, D]`; sequence `b` keeps its positions in the blocks
/// `block_tables[b]` lists (`[B, max_blocks]`, integers; entries past the sequence's blocks
/// are ignored). `q` is `[B, Hq, L, D]` (`Hq` a multiple of `kv_heads`): the last `L`
/// positions of each sequence, whose keys are already in the cache. Query `l` of sequence
/// `b` attends positions `0..=lengths[b] − L + l` (causal at the end; `lengths[b] ≥ L`).
/// Returns `[B, Hq, L, D]`.
///
/// Out-of-range inputs never read outside the caches: block ids are clamped to
/// `0..blocks` and lengths to `0..=max_blocks · block_size` (the table's capacity), on
/// every device.
///
/// On the GPU this is MLX's vector attention (one pass, or two for long caches) reading
/// keys through the block table, for head sizes 64, 128 and 256 and power-of-two block
/// sizes (others run the composition). Each sequence's result equals
/// [`scaled_dot_product_attention`]'s vector kernel over the same keys stored contiguously
/// when both take the same number of passes.
#[track_caller]
pub fn paged_attention(
    q: &Array,
    k_cache: &Array,
    v_cache: &Array,
    block_tables: &Array,
    lengths: &Array,
    scale: f32,
) -> Array {
    Array::sticky(
        "paged_attention",
        Location::caller(),
        &[q, k_cache, v_cache, block_tables, lengths],
        || imp::paged_attention(q, k_cache, v_cache, block_tables, lengths, scale),
    )
}

/// `cache` (`[blocks, kv_heads, block_size, D]`) with the positions `new` holds
/// (`[T, kv_heads, D]`) written into their slots: `slots[t]` (integers, `[T]`) is block
/// `slots[t] / block_size`, position `slots[t] % block_size`. Slots must be distinct;
/// out-of-range slots (negative, or at least `blocks · block_size`) are skipped. In place
/// when the evaluation owns the cache.
#[track_caller]
pub fn paged_write(cache: &Array, new: &Array, slots: &Array) -> Array {
    Array::sticky(
        "paged_write",
        Location::caller(),
        &[cache, new, slots],
        || imp::paged_write(cache, new, slots),
    )
}

/// The options of [`scaled_dot_product_attention`] and [`plan_attention`]: the mask,
/// attention sinks, a sliding window and a cap on score memory. Build them with
/// [`AttentionOptions::causal`] or `Default` and the chained setters:
///
/// ```
/// use nuvec_core::fast::AttentionOptions;
///
/// let opts = AttentionOptions::causal().window(4096).max_score_bytes(512 << 20);
/// assert_eq!(opts.window, Some(4096));
/// ```
///
/// This is the one options type of attention: new features (window and head-size-512
/// kernels, backward routing) extend it, never a second entry point.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct AttentionOptions<'a> {
    /// The mask: none (the default), causal, or an array.
    pub mask: AttentionMask<'a>,
    /// Attention sinks (`[Hq]`): one logit per query head that takes part in the softmax
    /// but attends to nothing.
    pub sinks: Option<&'a Array>,
    /// A sliding window: query `i` attends only to the `window` keys up to and including
    /// its position (as mlx-lm's `create_causal_mask(..., window_size)`). Needs the causal
    /// mask.
    pub window: Option<i32>,
    /// A cap, in bytes, on every score-sized intermediate (scores, masked scores, the
    /// float32 sink-softmax intermediates, materialised masks); see
    /// [`scaled_dot_product_attention`].
    pub max_score_bytes: Option<usize>,
}

impl<'a> AttentionOptions<'a> {
    /// Causal masking: query `i` attends to keys `0..=i + Lk - Lq` (the queries are the last
    /// `Lq` positions).
    pub fn causal() -> Self {
        AttentionOptions {
            mask: AttentionMask::Causal,
            ..Self::default()
        }
    }

    /// A boolean (true attends) or additive mask, broadcastable to `[B, Hq, Lq, Lk]`.
    #[must_use]
    pub fn mask(mut self, mask: &'a Array) -> Self {
        self.mask = AttentionMask::Array(mask);
        self
    }

    /// Attention sinks, `[Hq]` (`None` for none).
    #[must_use]
    pub fn sinks(mut self, sinks: impl Into<Option<&'a Array>>) -> Self {
        self.sinks = sinks.into();
        self
    }

    /// A sliding window of `keys` keys (needs [`AttentionOptions::causal`]).
    #[must_use]
    pub fn window(mut self, keys: i32) -> Self {
        self.window = Some(keys);
        self
    }

    /// Caps every score-sized intermediate at `bytes` (positive).
    #[must_use]
    pub fn max_score_bytes(mut self, bytes: usize) -> Self {
        self.max_score_bytes = Some(bytes);
        self
    }
}

impl<'a> From<AttentionMask<'a>> for AttentionOptions<'a> {
    /// Options with this mask and nothing else.
    fn from(mask: AttentionMask<'a>) -> Self {
        AttentionOptions {
            mask,
            ..Self::default()
        }
    }
}

/// The kernel family an attention call runs (see [`plan_attention`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernel {
    /// At most 8 queries: MLX's vector kernels. `passes` is 1 or 2; `blocks` the key
    /// partitions of the first of two passes (1 for one pass; the largest over
    /// dispatches); `dispatches` is more than 1 when query rows need different one-query
    /// configurations (the serial rule).
    Vector {
        /// Passes over the keys (1 or 2).
        passes: u8,
        /// Key partitions of the first pass.
        blocks: i32,
        /// Dispatches (groups of query rows with one configuration).
        dispatches: i32,
    },
    /// MLX's tiled (steel) attention kernel: no scores are materialised.
    Tiled,
    /// The composition: `softmax(scale · q kᵀ + mask) v` from matrix products.
    Composed,
    /// The composition per block of `rows` queries, one block after another.
    ComposedBlocks {
        /// Queries per block (the last block may have fewer).
        rows: i32,
    },
}

/// How an attention call runs and the memory it holds (see [`plan_attention`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionPlan {
    /// The kernel family.
    pub kernel: AttentionKernel,
    /// Bytes held beyond the inputs and the output while the call runs (for one block when
    /// blocked; see the `# Memory` section of [`scaled_dot_product_attention`]).
    pub scratch_bytes: usize,
    /// The largest single score-sized intermediate: what `max_score_bytes` bounds.
    pub max_intermediate_bytes: usize,
    /// Whether the cap changed the choice.
    pub capped: bool,
}

/// How [`scaled_dot_product_attention`] will run for these shapes (`[B, H, L, D]` each),
/// dtype and options, on the calling thread's default device, without evaluating or
/// allocating anything. Masks and sinks in `options` are read for their shape and dtype
/// only. Errors are those the attention call would report.
///
/// The plan is a pure function of the shapes, the dtype, the options and the GPU's
/// generation and class: the same inputs give the same plan in every call, thread and
/// process, whatever memory is free; the attention op runs exactly this plan (one routing
/// function).
///
/// ```
/// use nuvec_core::{DType, fast::{AttentionKernel, AttentionOptions, plan_attention}};
///
/// # fn main() -> nuvec_core::Result<()> {
/// let (q, k) = ([1, 16, 1024, 256], [1, 2, 33_792, 256]);
/// let opts = AttentionOptions::causal().max_score_bytes(512 << 20);
/// let plan = plan_attention(&q, &k, &k, DType::BF16, &opts)?;
/// assert!(plan.max_intermediate_bytes <= 512 << 20);
/// assert!(matches!(
///     plan.kernel,
///     AttentionKernel::Tiled | AttentionKernel::ComposedBlocks { .. }
/// ));
/// # Ok(())
/// # }
/// ```
pub fn plan_attention(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    dtype: crate::DType,
    options: &AttentionOptions<'_>,
) -> crate::Result<AttentionPlan> {
    imp::plan_attention(q, k, v, dtype, options)
}

/// Scaled dot-product attention, as MLX's `fast.scaled_dot_product_attention`:
/// `softmax(scale · q kᵀ + mask) v` over `[B, H, L, D]` arrays, with the mask, sinks,
/// window and score cap in [`AttentionOptions`].
///
/// `k` and `v` may have fewer heads than `q` (grouped-query attention; `Hq` a multiple of
/// `Hkv`), and `v` its own head size. Sinks (`[Hq]`) add one logit per head that takes part
/// in the softmax but attends to nothing.
///
/// ```
/// use nuvec_core::{DType, fast::{self, AttentionOptions}, ops};
///
/// let q = ops::ones([1, 4, 3, 64], DType::F32);
/// let kv = ops::ones([1, 2, 5, 64], DType::F32);
/// let out = fast::scaled_dot_product_attention(&q, &kv, &kv, 0.125, &AttentionOptions::causal());
/// assert_eq!(out.shape(), [1, 4, 3, 64]);
/// ```
///
/// # Routing
///
/// [`plan_attention`] decides, and reports, how the call runs. On the GPU, up to 8 queries
/// use MLX's vector kernels, more use its tiled kernel or the composition (as MLX: the
/// composition is faster for some head sizes and shapes); the CPU runs the composition.
/// A window runs through the composition (keys sliced to the window).
///
/// **The serial rule.** With 2 to 8 queries (speculative verification), each query row
/// takes the configuration (passes, key partitions, kernel variant) of the same row run
/// alone as one query over its causal key prefix, and rows with equal configurations share
/// a dispatch. So each row's result equals, bit for bit, the one-query call over its
/// prefix (MLX picks one configuration for all rows instead). Rows that need the
/// one-key-per-step kernel variant (float32 at head size 256) and do not fit one
/// threadgroup together keep MLX's composition instead.
///
/// **The cap.** With `max_score_bytes = c`, no score-sized intermediate larger than `c`
/// bytes is allocated (element size 4 with sinks or float32): either the tiled kernel
/// runs (it materialises no scores) or the composition runs in blocks of query rows, one
/// block after another (each block's queries depend on the previous block's output, so one
/// block's scratch is live at a time), whichever measured faster. The result equals the
/// uncapped composition within rounding. A cap below one query row's intermediate selects
/// the tiled kernel where it supports the shape and is an `InvalidArgument` error naming
/// the smallest admissible cap otherwise; a cap of 0 is invalid. Vector kernels (decode)
/// never materialise scores, so the cap does not affect them.
///
/// # Memory
///
/// Scratch beyond inputs and output ([`AttentionPlan::scratch_bytes`]), with `s` the
/// dtype's size, `r` the rows of a block (`Lq` unblocked), `L` its keys and `n = B·Hq·r·L`:
/// vector one pass 0; vector two passes `B·Hq·rows·blocks·(Dv·s + 8)` for the largest
/// dispatch (plus the gathered queries and outputs with several dispatches); tiled 0;
/// composition `n·s` (the scores, masked and normalised in place) plus the larger of the
/// scaled queries (`B·Hq·r·D·s`) and, with sinks below float32, the float32 softmax copy
/// (`n·4`; three of them in blocks under 16 MiB), plus `r·L` bytes of causal mask; blocked,
/// one block's composition plus the output's bytes (the blocks before concatenation).
#[track_caller]
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    options: &AttentionOptions<'_>,
) -> Array {
    let mask_array = match options.mask {
        AttentionMask::Array(m) => Some(m),
        _ => None,
    };
    let ins = operands(&[q, k, v], &[mask_array, options.sinks]);
    Array::sticky(
        "scaled_dot_product_attention",
        Location::caller(),
        &ins,
        || imp::scaled_dot_product_attention(q, k, v, scale, options),
    )
}

/// The gated delta rule, Gated DeltaNet's recurrence (as mlx-lm's `gated_delta_update`
/// once its gates are computed). For each step `t` and value head, with `S` its state
/// (`[Dv, Dk]`):
///
/// `S ← g_t · S`, then `S ← S + β_t (v_t − S k_t) k_tᵀ`, and `y_t = S q_t`.
///
/// Shapes: `q` and `k` are `[B, T, Hk, Dk]`, `v` is `[B, T, Hv, Dv]` (value head `h` reads
/// key head `h / (Hv / Hk)`), `g` and `beta` are `[B, T, Hv]` and `state` is
/// `[B, Hv, Dv, Dk]`. Returns `y` (`[B, T, Hv, Dv]`, in `q`'s dtype) and the final state
/// (in `state`'s dtype); the recurrence runs in float32. On the GPU it is one kernel
/// (mlx-lm's, for `Dk` a multiple of 32), else the steps as ordinary operations, which also
/// define the derivatives.
///
/// ```
/// use nuvec_core::{DType, fast, ops::{ones, zeros}};
///
/// let q = ones([1, 2, 1, 32], DType::F32);
/// let v = ones([1, 2, 1, 4], DType::F32);
/// let (g, beta) = (ones([1, 2, 1], DType::F32), ones([1, 2, 1], DType::F32));
/// let state = zeros([1, 1, 4, 32], DType::F32);
/// let (y, s) = fast::gated_delta(&(&q * 0.25), &(&q * 0.125), &v, &g, &(&beta * 0.5), &state);
/// // k · k = 0.5 and k · q = 1. Step 1: S = 0.5 · v kᵀ, so y = 0.5. Step 2: S k = 0.25,
/// // so S gains (1 − 0.25) · 0.5 · v kᵀ, and y = 0.875.
/// assert_eq!(y.to_vec::<f32>().unwrap(), [0.5, 0.5, 0.5, 0.5, 0.875, 0.875, 0.875, 0.875]);
/// assert_eq!(s.shape(), [1, 1, 4, 32]);
/// ```
#[track_caller]
pub fn gated_delta(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> (Array, Array) {
    let [y, s] = Array::sticky_n(
        "gated_delta",
        Location::caller(),
        &[q, k, v, g, beta, state],
        || {
            let (y, s) = imp::gated_delta(q, k, v, g, beta, state)?;
            Ok([y, s])
        },
    );
    (y, s)
}

/// [`gated_delta`] returning the state after every step, `[B, T, Hv, Dv, Dk]` (step `t`'s
/// state at index `t`; the last is [`gated_delta`]'s final state), so that a caller can
/// roll the recurrence back to any step, as speculative decoding does after a rejected
/// draft. The outputs equal [`gated_delta`]'s bit for bit; writing every step's state
/// costs `T` times the state's bytes.
#[track_caller]
pub fn gated_delta_steps(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> (Array, Array) {
    let [y, s] = Array::sticky_n(
        "gated_delta_steps",
        Location::caller(),
        &[q, k, v, g, beta, state],
        || {
            let (y, s) = imp::gated_delta_steps(q, k, v, g, beta, state)?;
            Ok([y, s])
        },
    );
    (y, s)
}

/// [`gated_delta`] for a batch of sequences whose states live in a pool: sequence `b`
/// starts from `pool[slots[b]]` (`pool` is `[slots, Hv, Dv, Dk]`, `slots` `[B]` integers,
/// distinct) and its final state replaces it. Returns the outputs and the updated pool,
/// which is written in place when the evaluation owns it, so decoding many sequences
/// moves no states. Each sequence's results equal [`gated_delta`]'s on its state alone.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_slots(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    pool: &Array,
    slots: &Array,
) -> (Array, Array) {
    let [y, p] = Array::sticky_n(
        "gated_delta_slots",
        Location::caller(),
        &[q, k, v, g, beta, pool, slots],
        || {
            let (y, p) = imp::gated_delta_slots(q, k, v, g, beta, pool, slots)?;
            Ok([y, p])
        },
    );
    (y, p)
}
