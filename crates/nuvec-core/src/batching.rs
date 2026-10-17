//! Batching rules: `Primitive::vmap`.
//!
//! Each rule receives the real batched inputs with their batch axes (`None` for unbatched
//! inputs) and records the batched computation. Rules that change axes first move the batch
//! axis to the front; element-wise rules without axis changes keep it in place. Every
//! primitive has a rule (D-015), so `vmap` needs no loop fallback.

use crate::array::Array;
use crate::ops::{broadcast_to, expand_dims, matmul, reshape, transpose};
use crate::primitive::Primitive;
use crate::{Result, view};

fn record(primitive: Primitive, inputs: Vec<Array>) -> Result<Vec<Array>> {
    Array::from_op(primitive, inputs)
}

/// `a` with axis `from` moved to position `to`.
pub(crate) fn move_axis(a: &Array, from: usize, to: usize) -> Result<Array> {
    if from == to {
        return Ok(a.clone());
    }
    let mut perm: Vec<usize> = (0..a.ndim()).filter(|&i| i != from).collect();
    perm.insert(to, from);
    transpose(a, &perm)
}

/// `a` with its batch axis first; unbatched inputs get a batch axis of size 1.
fn front(a: &Array, axis: Option<usize>) -> Result<Array> {
    match axis {
        Some(axis) => move_axis(a, axis, 0),
        None => expand_dims(a, 0),
    }
}

/// [`front`], with unbatched inputs broadcast to the full batch size `b`.
fn front_full(a: &Array, axis: Option<usize>, b: usize) -> Result<Array> {
    let x = front(a, axis)?;
    let mut shape = x.shape().to_vec();
    shape[0] = b;
    broadcast_to(&x, &shape)
}

/// Prepends a batch dimension to slice parameters.
fn prepend<T: Copy>(first: T, rest: &[T]) -> Vec<T> {
    std::iter::once(first).chain(rest.iter().copied()).collect()
}

impl Primitive {
    /// The batched outputs of this primitive for batched `inputs` (batch `axes`, `None`
    /// for unbatched inputs; at least one is batched), and their common batch axis.
    pub(crate) fn vmap(
        &self,
        inputs: &[Array],
        axes: &[Option<usize>],
    ) -> Result<(Vec<Array>, usize)> {
        let b = inputs
            .iter()
            .zip(axes)
            .find_map(|(a, ax)| ax.map(|ax| a.shape()[ax]))
            .expect("at least one batched input");
        let x = &inputs[0];
        let at_front = |p: Primitive| Ok((record(p, vec![front(x, axes[0])?])?, 0));
        match self {
            Self::Unary(_) | Self::AsType(_) | Self::StopGradient => {
                Ok((record(self.clone(), vec![x.clone()])?, axes[0].unwrap_or(0)))
            }
            Self::Binary(_) | Self::Select => self.elementwise(inputs, axes),
            Self::Reshape(shape) => {
                let shape: Vec<isize> = prepend(b, shape).iter().map(|&n| n as isize).collect();
                Ok((vec![reshape(&front(x, axes[0])?, &shape)?], 0))
            }
            Self::Transpose(perm) => {
                let perm: Vec<usize> = prepend(0, &perm.iter().map(|a| a + 1).collect::<Vec<_>>());
                Ok((vec![transpose(&front(x, axes[0])?, &perm)?], 0))
            }
            Self::Slice { start, step, len } => at_front(Self::Slice {
                start: prepend(0, start),
                step: prepend(1, step),
                len: prepend(b, len),
            }),
            Self::SliceUpdate { start, step, len } => {
                let batched = Self::SliceUpdate {
                    start: prepend(0, start),
                    step: prepend(1, step),
                    len: prepend(b, len),
                };
                let base = front_full(&inputs[0], axes[0], b)?;
                let update = front_full(&inputs[1], axes[1], b)?;
                Ok((record(batched, vec![base, update])?, 0))
            }
            Self::Broadcast(shape) => {
                let x = front(x, axes[0])?;
                let missing = shape.len() - (x.ndim() - 1);
                let aligned: Vec<isize> = std::iter::once(b)
                    .chain(std::iter::repeat_n(1, missing))
                    .chain(x.shape()[1..].iter().copied())
                    .map(|n| n as isize)
                    .collect();
                let out = broadcast_to(&reshape(&x, &aligned)?, &prepend(b, shape))?;
                Ok((vec![out], 0))
            }
            Self::Reduce { op, axes: reduced } => at_front(Self::Reduce {
                op: *op,
                axes: reduced.iter().map(|a| a + 1).collect(),
            }),
            // Keys `[..., 2]`: a batch of keys draws one sample per key.
            Self::RandomBits { .. } => {
                if axes[0] == Some(x.ndim() - 1) {
                    return Err(crate::Error::InvalidArgument(
                        "random_bits: the key pair axis cannot be batched".into(),
                    ));
                }
                at_front(self.clone())
            }
            Self::ArgReduce { op, axis } => at_front(Self::ArgReduce {
                op: *op,
                axis: axis + 1,
            }),
            // `matmul` broadcasts the size-1 batch axis of an unbatched operand.
            Self::Matmul => Ok((
                vec![matmul(&front(x, axes[0])?, &front(&inputs[1], axes[1])?)?],
                0,
            )),
            // `addmm` broadcasts `c` and size-1 batch axes, as `matmul` does.
            Self::AddMM { .. } => {
                let (alpha, beta) = self.addmm_factors();
                let a = front(x, axes[0])?;
                let b = front(&inputs[1], axes[1])?;
                let c = front(&inputs[2], axes[2])?;
                Ok((vec![crate::ops::addmm(&c, &a, &b, alpha, beta)?], 0))
            }
            Self::Custom(call) => {
                let (primals, axes) = (&inputs[..call.primals], &axes[..call.primals]);
                match &call.rules.vmap {
                    Some(rule) => rule(primals, axes),
                    None => Ok((
                        crate::transforms::batch(&*call.rules.body, primals, axes)?,
                        0,
                    )),
                }
            }
            #[cfg(test)]
            Self::AddMultiply => self.elementwise(inputs, axes),
        }
    }

    /// Element-wise primitives: in place when every input is batched along the same axis
    /// with equal shapes; otherwise batch axes go first and inputs are broadcast together.
    fn elementwise(&self, inputs: &[Array], axes: &[Option<usize>]) -> Result<(Vec<Array>, usize)> {
        if let Some(axis) = axes[0]
            && axes.iter().all(|&a| a == Some(axis))
            && inputs.iter().all(|a| a.shape() == inputs[0].shape())
        {
            return Ok((record(self.clone(), inputs.to_vec())?, axis));
        }
        let fronted = inputs
            .iter()
            .zip(axes)
            .map(|(a, &ax)| front(a, ax))
            .collect::<Result<Vec<_>>>()?;
        let mut shape = fronted[0].shape().to_vec();
        for a in &fronted[1..] {
            shape = view::broadcast_shapes(self.name(), &shape, a.shape())?.to_vec();
        }
        let operands = fronted
            .iter()
            .map(|a| broadcast_to(a, &shape))
            .collect::<Result<Vec<_>>>()?;
        Ok((record(self.clone(), operands)?, 0))
    }
}
