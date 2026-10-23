//! The one attention router: [`plan`] decides, from shapes, dtype, options and the GPU's
//! class only, which kernel family an attention call runs, how the vector kernels split
//! their keys, how a score-byte cap is honoured, and the scratch memory the call holds.
//! The attention op, the GPU encoder and the public [`crate::fast::plan_attention`] all
//! use it; there is no other routing code.

use smallvec::SmallVec;

use crate::{DType, Error, Result};

/// Where the call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    /// No fused kernel: the composition.
    Cpu,
    /// A Metal GPU of this class letter (`'g'` when unknown).
    Gpu {
        /// MLX's class letter (`d`, `s`, `g`, …).
        class: char,
    },
}

/// The mask, as far as routing is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaskKind {
    None,
    Causal,
    /// A boolean or additive mask array.
    Array,
}

/// Everything routing depends on (a pure function of it and the GPU's pipelines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouteKey {
    pub(crate) b: usize,
    pub(crate) hq: usize,
    pub(crate) hkv: usize,
    pub(crate) lq: usize,
    pub(crate) lk: usize,
    pub(crate) d: usize,
    pub(crate) dv: usize,
    pub(crate) dtype: DType,
    pub(crate) mask: MaskKind,
    /// A boolean mask array (else additive), when `mask` is `Array`.
    pub(crate) bool_mask: bool,
    pub(crate) sinks: bool,
    pub(crate) window: Option<usize>,
    pub(crate) cap: Option<usize>,
    pub(crate) target: Target,
}

/// How one dispatch group of vector attention runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupKind {
    /// MLX's vector kernel: `passes` 1 or 2, `blocks` key partitions of the first of two
    /// passes, the decode-only GQA kernel, or the one-key-per-step (`_tk1`) variant.
    Vector {
        passes: u8,
        blocks: usize,
        gqa_kernel: bool,
        tk1: bool,
    },
    /// One row through the composition (a row no vector kernel covers).
    Composed,
}

/// Consecutive query rows run as one dispatch over the first `keys` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Group {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) keys: usize,
    pub(crate) kind: GroupKind,
}

/// The routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Route {
    /// Up to 8 queries: the vector kernels, one dispatch per group (the serial rule).
    Vector(SmallVec<[Group; 2]>),
    /// MLX's tiled (steel) attention kernel.
    Tiled,
    /// The composition, in blocks of `rows` queries (`rows == lq`: one block).
    Composed { rows: usize },
}

/// A plan: the route and its memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) route: Route,
    /// Bytes held beyond inputs and output (one block's, when blocked).
    pub(crate) scratch: usize,
    /// The largest single score-sized intermediate.
    pub(crate) max_intermediate: usize,
    /// The cap changed the decision.
    pub(crate) capped: bool,
}

/// Whether MLX's vector attention takes two passes over `lk` keys on a GPU of `class`.
pub(crate) fn vector_two_pass(class: char, lk: usize, grouped: bool) -> bool {
    (matches!(class, 'd' | 's') && lk >= 1024) || (grouped && lk >= 4096)
}

/// MLX's key partitions (blocks) for the first of two passes, by GPU class, keys and
/// SIMD-groups per threadgroup (`n_simds`, query heads per key head times queries).
pub(crate) fn vector_blocks(class: char, lk: usize, n_simds: usize) -> usize {
    match class {
        's' if lk > 1024 && n_simds > 4 => match lk {
            ..=8192 => 128,
            8193..=32768 => 256,
            32769..=65536 => 512,
            _ => 1024,
        },
        's' => 64,
        'd' if n_simds <= 2 && lk > 8192 => 256,
        'd' if n_simds >= 6 && lk >= 65536 => 1024,
        'd' if n_simds >= 6 && lk >= 16384 => 512,
        'd' => 128,
        _ if n_simds >= 4 => 64,
        _ => 32,
    }
}

/// Whether the vector kernels cover these head sizes.
pub(crate) fn vector_head(d: usize, dv: usize) -> bool {
    (d == dv && matches!(d, 64 | 96 | 128 | 256 | 512))
        || (d, dv) == (192, 128)
        || (d, dv) == (96, 64)
}

/// Whether the tiled kernel covers these head sizes.
pub(crate) fn tiled_head(d: usize, dv: usize) -> bool {
    (d == dv && matches!(d, 64 | 72 | 80 | 96 | 128 | 192 | 256)) || (d, dv) == (96, 64)
}

/// Whether the two-pass kernel has a one-key-per-step (`_tk1`) variant for these types.
pub(crate) fn tk1_variant(dtype: DType, d: usize, dv: usize) -> bool {
    (dtype == DType::BF16 && d == 256 && dv == 256)
        || (dtype == DType::F32 && d == dv && (d == 192 || d == 256))
}

/// The configuration of one query row over `lk` keys run alone (MLX's one-query rule), or
/// `None` when no vector kernel covers it.
fn one_row(key: &RouteKey, class: char, lk: usize) -> Option<GroupKind> {
    let (d, dv) = (key.d, key.dv);
    let gqa = key.hq / key.hkv;
    let array_mask = key.mask == MaskKind::Array;
    let d512 = d != 512 || (gqa == 8 && !array_mask && lk >= 1024);
    if !(vector_head(d, dv) && lk >= 1 && gqa <= 32 && d512) {
        return None;
    }
    if !vector_two_pass(class, lk, key.hkv < key.hq) {
        return Some(GroupKind::Vector {
            passes: 1,
            blocks: 1,
            gqa_kernel: false,
            tk1: false,
        });
    }
    let blocks = vector_blocks(class, lk, gqa);
    let gqa_kernel = !array_mask
        && !key.sinks
        && d == dv
        && lk >= 8192
        && ((gqa == 8 && (d == 64 || d == 128)) || ((gqa == 12 || gqa == 16) && d == 128));
    let tk1 =
        !gqa_kernel && tk1_variant(key.dtype, d, dv) && 32 * gqa > threads(key, blocks, false);
    Some(GroupKind::Vector {
        passes: 2,
        blocks,
        gqa_kernel,
        tk1,
    })
}

/// Threads per threadgroup the four-keys-per-step first pass allows, with the function
/// constants of a dispatch over several rows (`causal_rows`) or one.
fn threads(key: &RouteKey, blocks: usize, causal_rows: bool) -> usize {
    let array_mask = key.mask == MaskKind::Array;
    crate::gpu::vector_threads(
        key.dtype,
        key.d,
        key.dv,
        crate::gpu::VectorFlags {
            has_mask: array_mask,
            causal: causal_rows && key.mask == MaskKind::Causal,
            bool_mask: array_mask && key.bool_mask,
            float_mask: array_mask && !key.bool_mask,
            sinks: key.sinks,
        },
        blocks,
    )
}

/// Whether `rows` rows of configuration `kind` fit one dispatch.
fn fits(key: &RouteKey, kind: GroupKind, rows: usize) -> bool {
    match kind {
        GroupKind::Composed => rows == 1,
        GroupKind::Vector { passes: 1, .. } => true,
        GroupKind::Vector {
            gqa_kernel,
            tk1,
            blocks,
            ..
        } => {
            if rows == 1 {
                return true;
            }
            !gqa_kernel
                && !tk1
                && key.d != 512
                && 32 * (key.hq / key.hkv) * rows <= threads(key, blocks, true)
        }
    }
}

/// The vector schedule (the serial rule): each row's configuration is the one-query rule
/// at its key count (its causal prefix under a causal mask), and consecutive rows with
/// equal configurations share a dispatch. `None` when no row has a vector kernel.
fn vector_schedule(key: &RouteKey, class: char) -> Option<SmallVec<[Group; 2]>> {
    let (lq, lk) = (key.lq, key.lk);
    if lq > lk || !vector_head(key.d, key.dv) {
        return None;
    }
    let keys = |row: usize| match key.mask {
        MaskKind::Causal => lk - (lq - 1 - row),
        _ => lk,
    };
    let kinds: SmallVec<[Option<GroupKind>; 8]> =
        (0..lq).map(|row| one_row(key, class, keys(row))).collect();
    if kinds.iter().all(Option::is_none) {
        return None;
    }
    // Rows that each need the one-key-per-step variant run one dispatch apiece, several
    // times slower than the composition MLX picks when the rows do not fit one vector
    // threadgroup (`Lq · gqa > 32`): those calls keep the composition (no serial rule).
    let gqa = key.hq / key.hkv;
    let tk1 = |k: &Option<GroupKind>| matches!(k, Some(GroupKind::Vector { tk1: true, .. }));
    if lq > 1 && lq * gqa > 32 && kinds.iter().any(tk1) {
        return None;
    }
    let mut groups: SmallVec<[Group; 2]> = SmallVec::new();
    for (row, kind) in kinds.into_iter().enumerate() {
        let kind = kind.unwrap_or(GroupKind::Composed);
        if let Some(last) = groups.last_mut()
            && last.kind == kind
            && fits(key, kind, last.end - last.start + 1)
        {
            last.end = row + 1;
            last.keys = keys(row);
            continue;
        }
        groups.push(Group {
            start: row,
            end: row + 1,
            keys: keys(row),
            kind,
        });
    }
    Some(groups)
}

/// Whether the uncapped tiled kernel runs for more than 8 queries (MLX's `use_fallback`,
/// with nuvec's measured head-size-256 rule).
fn tiled_uncapped(key: &RouteKey) -> bool {
    let (lq, lk) = (key.lq, key.lk);
    let causal = key.mask == MaskKind::Causal;
    // MLX: the composition is faster for these head sizes, except long causal
    // self-attention in half precision. Measured on an M4 Pro (re-measured with mrun's
    // chunks, `just bench-attention-route`), the tiled kernel is faster for causal chunks
    // of 1,024 queries or more over up to 4,096 keys (2,048 over 4,096: 19.9 ms against
    // 23.5) and ties for 2,048 queries over 8,192 keys; the composition is faster beyond
    // (1,024 over 8,192: 23.7 ms against 24.9) and for shorter chunks. The
    // composition materialises the scores, so from 4 GiB of them (a 2,048-token chunk
    // over ~64K keys and more) the tiled kernel runs anyway: re-measured at mrun's chunks
    // (16 query heads, half precision), the tiled kernel is 5–7% faster at exactly 4 GiB
    // (1,024 queries over 131K keys), 1–6% slower below (342 queries over 8K–32K keys).
    let scores = key.b * key.hq * lq * lk * key.dtype.size();
    let huge = scores >= 4 << 30;
    let slow = match key.d {
        192 => !huge,
        256 => !(huge || (causal && lq >= 1024 && (lk <= 4096 || (lq >= 2048 && lk <= 8192)))),
        _ => false,
    };
    !slow
}

/// Bytes per element of the largest score-sized intermediate (float32 with sinks).
fn score_size(key: &RouteKey) -> usize {
    if key.sinks || key.dtype == DType::F32 {
        4
    } else {
        key.dtype.size()
    }
}

/// Keys one block of `rows` queries reads.
fn block_keys(key: &RouteKey, rows: usize) -> usize {
    match key.window {
        Some(w) => key.lk.min(w.saturating_add(rows - 1)),
        None => key.lk,
    }
}

/// The largest score-sized intermediate of a block of `rows` queries.
fn block_intermediate(key: &RouteKey, rows: usize) -> usize {
    key.b
        .saturating_mul(key.hq)
        .saturating_mul(rows)
        .saturating_mul(block_keys(key, rows))
        .saturating_mul(score_size(key))
}

/// The composition's scratch for one block of `rows` queries (the data model's formula,
/// fitted to measurements on the M4 Pro): the scores (masked and normalised in place) with,
/// before them, the scaled queries or, after them, the float32 sink softmax's intermediate
/// (whichever is larger; the scaled queries are released, or never materialised, before
/// the softmax), and the mask built in the graph.
fn composed_scratch(key: &RouteKey, rows: usize) -> usize {
    let s = key.dtype.size();
    let l = block_keys(key, rows);
    let n = key
        .b
        .saturating_mul(key.hq)
        .saturating_mul(rows)
        .saturating_mul(l);
    let scaled_q = key.b * key.hq * rows * key.d * s;
    // Float32 scores are their own float32 copy; narrower ones gain one. In blocks below
    // the GPU's early-release size (16 MiB: smaller intermediates return when their
    // command buffer is committed), the sink softmax's float32 steps can coexist: three.
    let f32_scores = n.saturating_mul(4);
    let blocked = rows < key.lq;
    let sinks = match key.sinks && key.dtype != DType::F32 {
        false => 0,
        true if blocked && f32_scores < 16 << 20 => f32_scores.saturating_mul(3),
        true => f32_scores,
    };
    let mask = match (key.mask, key.window) {
        (_, Some(_)) => 3 * rows * l + 4 * (rows + l),
        (MaskKind::Causal, None) => rows * l + 4 * (rows + l),
        _ => 0,
    };
    scaled_q
        .max(sinks)
        .saturating_add(n.saturating_mul(s))
        .saturating_add(mask)
}

/// The blocked composition's scratch: one block's, plus the output's bytes (the blocks'
/// results, held until they are concatenated; an upper bound).
fn blocked_scratch(key: &RouteKey, rows: usize) -> usize {
    let blocked = rows < key.lq;
    let outputs = key.b * key.hq * key.lq * key.dv * key.dtype.size();
    composed_scratch(key, rows).saturating_add(if blocked { outputs } else { 0 })
}

/// The vector schedule's scratch: one group's two-pass partial results (groups run one
/// after another), and the gathered queries and per-group outputs when the rows run in
/// several dispatches.
fn vector_scratch(key: &RouteKey, groups: &[Group]) -> usize {
    let s = key.dtype.size();
    let mut bytes = 0usize;
    for g in groups {
        let rows = g.end - g.start;
        let group = match g.kind {
            GroupKind::Vector {
                passes: 2, blocks, ..
            } => key.b * key.hq * rows * blocks * (key.dv * s + 8),
            GroupKind::Composed => {
                let one = RouteKey {
                    lq: 1,
                    lk: g.keys,
                    mask: MaskKind::None,
                    ..*key
                };
                composed_scratch(&one, 1)
            }
            GroupKind::Vector { .. } => 0,
        };
        bytes = bytes.max(group);
    }
    if groups.len() > 1 {
        bytes += key.b * key.hq * key.lq * (key.d + key.dv) * s;
    }
    bytes
}

/// The benchmarking override of the capped choice (`NUVEC_ATTENTION_CAPPED_KERNEL` set
/// to `tiled` or `blocks`, read once), which the capped routing table is measured with.
fn capped_override() -> Option<bool> {
    static CHOICE: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
        match std::env::var("NUVEC_ATTENTION_CAPPED_KERNEL").as_deref() {
            Ok("tiled") => Some(true),
            Ok("blocks") => Some(false),
            _ => None,
        }
    });
    *CHOICE
}

/// Under a cap, whether the tiled kernel (else the composition in query blocks of
/// `rows`) runs, for a shape both serve.
fn capped_tiled(_key: &RouteKey, rows: usize) -> bool {
    if let Some(tiled) = capped_override() {
        return tiled;
    }
    // Measured on an M4 Pro (`just bench-attention-route`; head size 256, 16 query and 2
    // key heads, 512 MiB cap, half precision): the composition in blocks of 512 rows over
    // 32K keys takes 94 ms against the tiled kernel's 104 (1,024 queries) and 185 against
    // 204 (2,048); blocks of 256 rows (float32) 124 against 131; at 128–171 rows both tie;
    // blocks of 114 rows over 131K keys take 157 ms against 148, of 57 rows over 262K keys
    // 357 against 299. Smaller blocks re-read the keys more often.
    rows < 192
}

/// The plan of `key`.
pub(crate) fn plan(key: &RouteKey) -> Result<Plan> {
    if key.cap == Some(0) {
        return Err(Error::InvalidArgument(
            "scaled_dot_product_attention: max_score_bytes must be positive".into(),
        ));
    }
    if key.window.is_some() && key.mask != MaskKind::Causal {
        return Err(Error::InvalidArgument(
            "scaled_dot_product_attention: a window needs the causal mask".into(),
        ));
    }
    if key.window == Some(0) {
        return Err(Error::InvalidArgument(
            "scaled_dot_product_attention: the window must be at least 1".into(),
        ));
    }
    let composed = |rows: usize, capped: bool| Plan {
        route: Route::Composed { rows },
        scratch: blocked_scratch(key, rows),
        max_intermediate: block_intermediate(key, rows),
        capped,
    };
    if key.lq == 0 || key.lk == 0 {
        return Ok(Plan {
            route: Route::Composed { rows: key.lq },
            scratch: 0,
            max_intermediate: 0,
            capped: false,
        });
    }
    let class = match key.target {
        Target::Gpu { class } => Some(class),
        Target::Cpu => None,
    };
    if let Some(class) = class
        && key.lq <= 8
        && key.window.is_none()
        && let Some(groups) = vector_schedule(key, class)
    {
        let scratch = vector_scratch(key, &groups);
        return Ok(Plan {
            route: Route::Vector(groups),
            scratch,
            max_intermediate: 0,
            capped: false,
        });
    }
    let tiled_ok = class.is_some()
        && key.window.is_none()
        && tiled_head(key.d, key.dv)
        && (key.mask != MaskKind::Causal || key.lq <= key.lk);
    let tiled = |capped: bool| Plan {
        route: Route::Tiled,
        scratch: 0,
        max_intermediate: 0,
        capped,
    };
    // (The benchmarking override `blocks` measures the composition under a cap even where
    // the tiled kernel would run uncapped.)
    let forced_blocks = key.cap.is_some() && capped_override() == Some(false);
    if tiled_ok && key.lq > 8 && tiled_uncapped(key) && !forced_blocks {
        return Ok(tiled(false));
    }
    let Some(cap) = key.cap else {
        return Ok(composed(key.lq, false));
    };
    if block_intermediate(key, key.lq) <= cap {
        return Ok(composed(key.lq, false));
    }
    // The most rows per block within the cap (the intermediate grows with the rows).
    let (mut lo, mut hi) = (0usize, key.lq);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if block_intermediate(key, mid) <= cap {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let rows_max = lo;
    if rows_max == 0 {
        if tiled_ok {
            return Ok(tiled(true));
        }
        return Err(Error::InvalidArgument(format!(
            "scaled_dot_product_attention: max_score_bytes {cap} is below one query row: \
             needs at least {} bytes",
            block_intermediate(key, 1)
        )));
    }
    let blocks = key.lq.div_ceil(rows_max);
    let rows = key.lq.div_ceil(blocks);
    if tiled_ok && capped_tiled(key, rows) {
        return Ok(tiled(true));
    }
    Ok(composed(rows, true))
}
