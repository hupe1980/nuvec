//! Loss functions, as in MLX's `nn.losses`.

use nuvec::{Array, DType, fast, ops};

/// How per-example losses are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// One loss per example.
    None,
    /// The mean over examples.
    Mean,
    /// The sum over examples.
    Sum,
}

/// Cross entropy of `logits` (classes on the last axis) against integer class indices
/// `targets` (the logits' shape without the last axis), as MLX's `nn.losses.cross_entropy`:
/// `logsumexp(z) − z[target]` with `z = logits − max(logits)` (the maximum without
/// gradient); on the GPU, the fused [`nuvec::fast::cross_entropy`] in the logits' dtype.
///
/// Mismatched shapes or non-integer targets give a failed array (with the error of
/// [`nuvec::fast::cross_entropy`]).
#[track_caller]
pub fn cross_entropy(logits: &Array, targets: &Array, reduction: Reduction) -> Array {
    let valid = logits.ndim() > 0
        && targets.shape() == &logits.shape()[..logits.ndim() - 1]
        && is_index_dtype(targets.dtype());
    let loss = if !valid || nuvec::default_device().kind == nuvec::DeviceKind::Gpu {
        // One fused kernel (and one for the gradient), as MLX on the GPU; it also reports
        // invalid arguments as a failed array.
        fast::cross_entropy(logits, targets).astype(logits.dtype())
    } else {
        let classes = logits.shape()[logits.ndim() - 1];
        let z = logits.subtract(logits.max_keepdims(-1).stop_gradient());
        let positions = ops::arange(0, classes, 1).astype(targets.dtype());
        let hit = positions.equal(targets.expand_dims(-1));
        let score = hit.select(&z, 0.0).sum(-1);
        z.logsumexp(-1).subtract(&score)
    };
    match reduction {
        Reduction::None => loss,
        Reduction::Mean => loss.mean(..),
        Reduction::Sum => loss.sum(..),
    }
}

/// Class indices are integers (not one-hot floats).
fn is_index_dtype(d: DType) -> bool {
    !matches!(
        d,
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::Complex64
    )
}
