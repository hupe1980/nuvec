//! Acceptance for speculative decoding, on the device: from a verification's logits (or
//! probabilities) and the draft, one `[3]` uint32 array `[accepted, next, eos_hit]`, so a
//! verification step costs one small read.
//!
//! - `accepted`: the draft tokens kept (a prefix), cut after the first accepted
//!   end-of-sequence token;
//! - `next`: the token after them (the model's own choice at that position, or a sample);
//! - `eos_hit`: 1 if an accepted draft token or `next` is an end-of-sequence id.
//!
//! Both are compositions of array operations (CPU and GPU, any strides; `vmap` over a
//! leading batch of verification steps; no gradient, as `argmax`).
//!
//! ```
//! use nuvec_core::{Array, array, speculative};
//!
//! # fn main() -> nuvec_core::Result<()> {
//! // The model's choices at three positions are 2, 0, 1: the draft [2, 0] is accepted,
//! // and 1 comes next.
//! let logits = array![[0.0, 0.0, 9.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]];
//! let no_eos = Array::from_slice(&[0u32; 0], [0]);
//! let out = speculative::accept_greedy(&logits, &array![2u32, 0], &no_eos);
//! assert_eq!(speculative::read(&out)?, (2, 1, false)); // one read per verification
//! # Ok(())
//! # }
//! ```

use std::panic::Location;

use crate::{Array, DType, OpErrorKind, Result, ops};

/// The rows and draft length, or the error naming the wrong input.
fn check(op: &str, rows: &Array, draft: &Array, eos: &Array) -> std::result::Result<i32, String> {
    if rows.ndim() != 2 || !matches!(rows.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(format!(
            "{op}: the verification rows must be [N + 1, vocab] floats, got {:?} {}",
            rows.shape(),
            rows.dtype().name()
        ));
    }
    for (name, a) in [("draft", draft), ("eos", eos)] {
        if a.ndim() != 1 || !matches!(a.dtype(), DType::U32 | DType::I32) {
            return Err(format!(
                "{op}: `{name}` must be a [n] uint32 or int32 array, got {:?} {}",
                a.shape(),
                a.dtype().name()
            ));
        }
    }
    let n = draft.shape()[0];
    if rows.shape()[0] != n + 1 {
        return Err(format!(
            "{op}: {} rows for a draft of {n} tokens (N + 1 needed)",
            rows.shape()[0]
        ));
    }
    Ok(n)
}

/// Whether each of `tokens` (`[n]`) is one of `eos` (`[e]`; none when empty).
fn is_eos(tokens: &Array, eos: &Array) -> Array {
    let t = tokens.astype(DType::U32).reshape([-1, 1]);
    let e = eos.astype(DType::U32).reshape([1, -1]);
    t.equal(&e).any(-1)
}

/// `[accepted, next, eos_hit]` from the draft's per-token acceptance `ok` (`[n]` bool), the
/// next-token candidates per position (`next`, `[n + 1]`), the draft and the EOS ids.
fn finish(ok: &Array, next: &Array, draft: &Array, eos: &Array) -> Array {
    // The accepted prefix: positions before the first rejection.
    let prefix = ok.astype(DType::U32).cumprod(0);
    // Cut after the first accepted end-of-sequence token.
    let eos_in = prefix.multiply(is_eos(draft, eos).astype(DType::U32));
    let before = eos_in.cumsum(0).subtract(&eos_in); // EOS strictly before each position
    let keep = prefix.multiply(before.equal(0u32).astype(DType::U32));
    let accepted = keep.sum(..).astype(DType::U32);
    let chosen = next
        .take(&accepted.reshape([1]), 0)
        .squeeze(..)
        .astype(DType::U32);
    let hit = eos_in
        .multiply(&keep)
        .sum(..)
        .greater(0u32)
        .add(is_eos(&chosen.reshape([1]), eos).squeeze(..)) // or, for booleans
        .astype(DType::U32);
    ops::stack(&[accepted, chosen, hit], 0)
}

/// Greedy acceptance: `logits` `[N + 1, vocab]` (float32/16, bfloat16; any strides, such as
/// a slice of `[1, L, vocab]`), `draft` `[N]` and `eos` `[E]` (uint32 or int32, maybe still
/// on the device). The accepted prefix is the draft tokens equal to the logits' argmax at
/// their position; `next` is the argmax after it. Invalid inputs give a failed array.
#[track_caller]
pub fn accept_greedy(logits: &Array, draft: &Array, eos: &Array) -> Array {
    let at = Location::caller();
    Array::sticky("accept_greedy", at, &[logits, draft, eos], || {
        let n = check("accept_greedy", logits, draft, eos).map_err(invalid)?;
        let choice = logits.argmax(-1).astype(DType::U32);
        let ok = draft.astype(DType::U32).equal(choice.i(0..n));
        Ok(finish(&ok, &choice, draft, eos))
    })
}

fn invalid(m: String) -> crate::Error {
    crate::Error::Op {
        op: "speculative",
        at: Location::caller(),
        kind: OpErrorKind::InvalidArgument(m),
    }
}

/// Lossless sampled acceptance (Leviathan et al., Chen et al.): `probs` `[N + 1, vocab]` is
/// the target's sampling distribution at each position (after temperature and filters),
/// `draft_probs` `[N, vocab]` the drafter's (`None`: deterministic drafts, a point mass on
/// each draft token). Draft token `i` is accepted when `u_i < min(1, p_i(d_i) / q_i(d_i))`;
/// after the first rejection at position `a`, `next` is sampled from
/// `norm(max(p_a − q_a, 0))` (`p_a` itself when that has no mass), and after a fully
/// accepted draft from `p_N`. The output's distribution is the target's.
///
/// `key` (a threefry key, [`crate::random`]) is split into two: the first draws the
/// uniforms `u` (`[N]`), the second `next`; the same key gives the same result.
#[track_caller]
pub fn accept_sampled(
    probs: &Array,
    draft: &Array,
    draft_probs: Option<&Array>,
    eos: &Array,
    key: &Array,
) -> Array {
    let at = Location::caller();
    let mut inputs = vec![probs, draft, eos, key];
    inputs.extend(draft_probs);
    Array::sticky("accept_sampled", at, &inputs, || {
        let n = check("accept_sampled", probs, draft, eos).map_err(invalid)?;
        let v = probs.shape()[1];
        let p = probs.astype(DType::F32);
        let d = draft.astype(DType::U32).reshape([n, 1]);
        let q = match draft_probs {
            Some(q) => {
                if q.shape() != [n, v] {
                    return Err(invalid(format!(
                        "accept_sampled: `draft_probs` must be [{n}, {v}], got {:?}",
                        q.shape()
                    )));
                }
                q.astype(DType::F32)
            }
            // A point mass on each draft token.
            None => ops::arange(0, v, 1)
                .reshape([1, v])
                .equal(&d)
                .astype(DType::F32),
        };
        let [uniform_key, next_key] = crate::random::split_n::<2>(key);
        let p_draft = p.i(0..n).take_along_axis(&d, -1).reshape([n]);
        let q_draft = q.take_along_axis(&d, -1).reshape([n]);
        let u = crate::random::uniform(0.0, 1.0, [n], DType::F32, &uniform_key);
        // u < p / q, without dividing: u · q < p (q > 0 for a draft token the drafter drew).
        let ok = u.multiply(&q_draft).less(&p_draft);
        // Per position: the residual (draft positions) or the target (the last row).
        let residual = p.i(0..n).subtract(&q).maximum(0.0);
        let mass = residual.sum_keepdims(-1);
        let residual = mass
            .greater(0.0)
            .select(residual.divide(mass.maximum(f32::MIN_POSITIVE)), p.i(0..n));
        let rows = ops::concatenate(&[residual, p.i(n..n + 1)], 0);
        // `next` for every possible accepted count; `finish` picks the one at `accepted`.
        let keys = crate::random::split(&next_key, (n + 1) as usize);
        let next = ops::stack(
            &(0..=n)
                .map(|i| {
                    crate::random::categorical(&rows.i(i).log(), -1, &keys.i(i)).astype(DType::U32)
                })
                .collect::<Vec<_>>(),
            0,
        );
        Ok(finish(&ok, &next, draft, eos))
    })
}

/// The accepted count, the next token and whether generation ended, read from an
/// acceptance result in one host read.
pub fn read(result: &Array) -> Result<(usize, u32, bool)> {
    match result.to_vec::<u32>()?.as_slice() {
        &[accepted, next, eos] => Ok((accepted as usize, next, eos != 0)),
        v => Err(crate::Error::InvalidArgument(format!(
            "speculative::read: an acceptance result has 3 values, got {}",
            v.len()
        ))),
    }
}
