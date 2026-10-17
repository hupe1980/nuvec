//! Loss functions, as in MLX's `nn.losses`.

use nuvec::{
    Array, DType, Error, Result, arange, expand_dims, logsumexp, max, mean, stop_gradient,
    subtract, sum, where_,
};

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
/// gradient).
pub fn cross_entropy(logits: &Array, targets: &Array, reduction: Reduction) -> Result<Array> {
    let classes = *logits
        .shape()
        .last()
        .ok_or_else(|| Error::InvalidArgument("cross_entropy: logits need a class axis".into()))?;
    if targets.shape() != &logits.shape()[..logits.ndim() - 1] {
        return Err(Error::InvalidArgument(format!(
            "cross_entropy: targets shape {:?} does not match logits shape {:?}",
            targets.shape(),
            logits.shape()
        )));
    }
    if !is_index_dtype(targets.dtype()) {
        return Err(Error::InvalidArgument(format!(
            "cross_entropy: targets must be integer class indices, got {}",
            targets.dtype()
        )));
    }
    let z = subtract(logits, stop_gradient(&max(logits, &[-1], true)?)?)?;
    let positions = arange(0.0, classes as f64, 1.0, targets.dtype())?;
    let hit = nuvec::equal(&positions, expand_dims(targets, -1)?)?;
    let score = sum(&where_(&hit, &z, 0.0)?, &[-1], false)?;
    let loss = subtract(logsumexp(&z, &[-1], false)?, &score)?;
    match reduction {
        Reduction::None => Ok(loss),
        Reduction::Mean => mean(&loss, &[], false),
        Reduction::Sum => sum(&loss, &[], false),
    }
}

/// Class indices are integers (not one-hot floats).
fn is_index_dtype(d: DType) -> bool {
    !matches!(
        d,
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::Complex64
    )
}
