//! The chunked linear cross-entropy: per-token losses of an output head over row chunks, so
//! the full logits never exist (Liger-Kernel's fused linear cross-entropy).
//!
//! Each chunk's logits are evaluated once. Under a reverse transform, that evaluation also
//! gives the chunk's hidden-state gradient (one token's loss depends only on its own row),
//! which the VJP scales by the cotangent. A trainable head's gradient is a second chunk
//! chain, built only when its cotangent is used. `depends` orders the chunks, so one
//! chunk's logits are alive at a time.

use std::sync::Arc;

use crate::custom::custom_function_flat;
use crate::{Array, DType, Error, Result, ops};

/// The head: logits `[R, V]` of a row chunk `[R, D]` and the head's weights.
pub(crate) type Head = dyn Fn(&Array, &[Array]) -> Array + Send + Sync;

/// The logits chunk's budget: rows per chunk are the largest power of two whose logits fit.
const CHUNK_BYTES: usize = 64 << 20;

pub(crate) fn linear_cross_entropy(
    h: &Array,
    targets: &Array,
    weights: &[Array],
    head: Arc<Head>,
    rows: Option<usize>,
) -> Result<Array> {
    let invalid = |m: String| Err(Error::InvalidArgument(format!("linear_cross_entropy: {m}")));
    if h.ndim() == 0 || targets.shape() != &h.shape()[..h.ndim() - 1] {
        return invalid(format!(
            "targets {:?} must be the hidden states' shape {:?} without the last axis",
            targets.shape(),
            h.shape()
        ));
    }
    if matches!(
        targets.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::Complex64
    ) {
        return invalid(format!(
            "targets must be integers, not {:?}",
            targets.dtype()
        ));
    }
    if rows == Some(0) {
        return invalid("rows per chunk must be at least 1".into());
    }
    let d = h.shape()[h.ndim() - 1];
    let n: i32 = targets.shape().iter().product();
    let h2 = h.reshape([n, d]);
    let t1 = targets.reshape([n]);
    // The logits' dtype and vocabulary, from the head's shape inference on one row (never
    // evaluated).
    let probe = head(&h2.i((0..n.min(1), ..)), weights).check()?;
    let (dtype, vocab) = (probe.dtype(), probe.shape()[probe.ndim() - 1] as usize);
    let rows = rows.unwrap_or_else(|| {
        let fit = (CHUNK_BYTES / (vocab * dtype.size()).max(1)).max(1);
        (1usize << fit.ilog2()).clamp(64, 4096)
    });
    let rows = rows.min(n.max(1) as usize) as i32;
    let save = crate::transforms::in_reverse() && crate::graph::is_tracer(h);
    let body_head = head.clone();
    let f = custom_function_flat(move |xs: &[Array]| {
        let (h, t, ws) = (&xs[0], &xs[1], &xs[2..]);
        if save {
            let (losses, grads) = chain(h, t, ws, &*body_head, rows, dtype, None, [true, false])?;
            Ok(vec![losses, grads.0.expect("asked for")])
        } else {
            Ok(vec![losses_only(h, t, ws, &*body_head, rows, dtype)?])
        }
    })
    .with_masked_vjp(move |primals, outputs, cotangents, wanted| {
        let (h, t, ws) = (&primals[0], &primals[1], &primals[2..]);
        let g = &cotangents[0];
        let want_w = wanted[2..].iter().any(|&w| w);
        let dh = match outputs.get(1) {
            Some(saved) if wanted[0] => {
                Some(saved.multiply(g.expand_dims(-1).astype(saved.dtype())))
            }
            _ => None,
        };
        let need_h = wanted[0] && dh.is_none();
        let (dh2, dw) = if need_h || want_w {
            let (_, (dh2, dw)) = chain(h, t, ws, &*head, rows, dtype, Some(g), [need_h, want_w])?;
            (dh2, dw)
        } else {
            (None, None)
        };
        let dh = dh.or(dh2).unwrap_or_else(|| h.zeros_like());
        let mut out = vec![dh, t.zeros_like()];
        match dw {
            Some(dw) => out.extend(dw.into_iter().zip(ws).map(|(g, w)| g.astype(w.dtype()))),
            None => out.extend(ws.iter().map(Array::zeros_like)),
        }
        Ok(out)
    });
    let mut inputs = vec![h2, t1];
    inputs.extend(weights.iter().cloned());
    let out = f.call(&inputs)?;
    Ok(out[0].reshape(targets.shape()))
}

/// The losses alone (no gradient work), one chunk after another.
fn losses_only(
    h: &Array,
    t: &Array,
    ws: &[Array],
    head: &Head,
    rows: i32,
    dtype: DType,
) -> Result<Array> {
    let n = h.shape()[0];
    let mut parts = Vec::new();
    let mut prev: Option<Array> = None;
    for start in (0..n).step_by(rows as usize) {
        let end = (start + rows).min(n);
        let mut hc = h.i((start..end, ..));
        if let Some(p) = &prev {
            hc = ops::depends(&hc, std::slice::from_ref(p));
        }
        let loss = super::cross_entropy(&head(&hc, ws), &t.i(start..end)).astype(dtype);
        prev = Some(loss.clone());
        parts.push(loss);
    }
    Ok(match parts.len() {
        0 => ops::zeros([0], dtype),
        _ => ops::concatenate(&parts, 0),
    })
}

type Grads = (Option<Array>, Option<Vec<Array>>);

/// One pass over the chunks giving the losses and, as `want` asks, the hidden-state
/// gradient `[N, D]` and the weights' gradients (float32 sums), for per-token cotangents
/// `cot` (ones without them).
#[allow(clippy::too_many_arguments)]
fn chain(
    h: &Array,
    t: &Array,
    ws: &[Array],
    head: &Head,
    rows: i32,
    dtype: DType,
    cot: Option<&Array>,
    want: [bool; 2],
) -> Result<(Array, Grads)> {
    let n = h.shape()[0];
    let (mut losses, mut dh) = (Vec::new(), Vec::new());
    let mut dw: Option<Vec<Array>> = None;
    let mut prev: Vec<Array> = Vec::new();
    for start in (0..n).step_by(rows as usize) {
        let end = (start + rows).min(n);
        let mut hc = h.i((start..end, ..));
        if !prev.is_empty() {
            hc = ops::depends(&hc, &prev);
        }
        let tc = t.i(start..end);
        let gc = match cot {
            Some(g) => g.i(start..end).astype(dtype),
            None => ops::ones([end - start], dtype),
        };
        // The hidden-state gradient is also the ordering link when no weight is wanted.
        let with_h = want[0] || !want[1];
        let f = |xs: &[Array]| {
            let (x, rest) = if with_h {
                (&xs[0], &xs[1..])
            } else {
                (&hc, xs)
            };
            let w = if want[1] { rest } else { ws };
            Ok(vec![super::cross_entropy(&head(x, w), &tc).astype(dtype)])
        };
        let mut primals: Vec<Array> = Vec::new();
        if with_h {
            primals.push(hc.clone());
        }
        if want[1] {
            primals.extend(ws.iter().cloned());
        }
        let (out, grads) = crate::vjp_flat(f, &primals, std::slice::from_ref(&gc))?;
        prev = vec![out[0].clone()];
        losses.push(out[0].clone());
        let mut grads = grads.into_iter();
        if with_h {
            let g = grads.next().expect("one per primal");
            prev.push(g.clone());
            dh.push(g);
        }
        if want[1] {
            let g: Vec<Array> = grads.map(|g| g.astype(DType::F32)).collect();
            let sum = match dw.take() {
                Some(acc) => acc.iter().zip(&g).map(|(a, b)| a.add(b)).collect(),
                None => g,
            };
            prev.extend(sum.iter().cloned());
            dw = Some(sum);
        }
    }
    let losses = match losses.len() {
        0 => ops::zeros([0], dtype),
        _ => ops::concatenate(&losses, 0),
    };
    let dh = (want[0] || !want[1]).then(|| match dh.len() {
        0 => h.zeros_like(),
        _ => ops::concatenate(&dh, 0),
    });
    let dw = want[1].then(|| {
        dw.unwrap_or_else(|| {
            ws.iter()
                .map(|w| w.zeros_like().astype(DType::F32))
                .collect()
        })
    });
    Ok((losses, (dh, dw)))
}
