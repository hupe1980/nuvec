//! Tensor-parallel linear layers over a distributed group, as `mlx.nn`'s
//! `AllToShardedLinear` and `ShardedToAllLinear` (and their quantized forms).
//!
//! A pair of them splits a block across the ranks with one collective: the first layer
//! keeps each rank's slice of the output features (its input is the full activation), the
//! second each rank's slice of the input features, and sums the partial products over the
//! group. Every rank holds `1/size` of both weights.
//!
//! ```
//! use nuvec::distributed::Group;
//! use nuvec::{array, random};
//! use nuvec_nn::Linear;
//! use nuvec_nn::sharded::{AllToShardedLinear, ShardedToAllLinear};
//!
//! # fn main() -> nuvec::Result<()> {
//! random::seed(0);
//! let (up, down) = (Linear::new(4, 8), Linear::new(8, 4));
//! let group = Group::single(); // `nuvec::distributed::init(false)?` across processes
//! let up_r = AllToShardedLinear::from_linear(&up, &group)?;
//! let down_r = ShardedToAllLinear::from_linear(&down, &group)?;
//! let x = array![[1.0, 2.0, 3.0, 4.0]];
//! let y = down_r.forward(&up_r.forward(&x));
//! assert_eq!(y.shape(), [1, 4]);
//! # Ok(())
//! # }
//! ```

use nuvec::distributed::{self, Group};
use nuvec::{Array, Error, Param, Result};

use crate::{Linear, Module, QuantizedLinear};

/// The projection a sharded layer holds: this rank's slice of a full-precision or a
/// quantized layer.
#[derive(Debug, Clone, Module)]
pub enum Shard {
    /// A full-precision slice.
    Linear(Linear),
    /// A quantized slice.
    Quantized(QuantizedLinear),
}

impl Shard {
    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

/// `[start, end)` of rank `rank`'s slice of `n` features, or an error naming `layer`.
fn slice(layer: &str, what: &str, n: i32, unit: i32, group: &Group) -> Result<(i32, i32)> {
    let size = group.size() as i32;
    if n % (size * unit) != 0 {
        return Err(Error::InvalidArgument(format!(
            "{layer}: {n} {what} do not split into {size} shards{}",
            if unit > 1 {
                format!(" of whole groups of {unit}")
            } else {
                String::new()
            }
        )));
    }
    let per = n / size;
    let r = group.rank() as i32;
    Ok((r * per, (r + 1) * per))
}

fn dims(a: &Array) -> (i32, i32) {
    let s = a.shape();
    (
        s.first().copied().unwrap_or(0),
        s.get(1).copied().unwrap_or(0),
    )
}

/// A linear layer whose output features are split across `group` (MLX's
/// `AllToShardedLinear`): each rank computes its slice of the output from the full input.
/// The input's gradient is summed over the group (`sum_gradients`), so training sees every
/// rank's slice.
#[derive(Debug, Clone, Module)]
pub struct AllToShardedLinear {
    /// This rank's rows of the weight (and its slice of the bias).
    pub shard: Shard,
    /// The group the features are split across.
    #[tree(skip)]
    pub group: Group,
}

impl AllToShardedLinear {
    /// This rank's slice of `linear` (`[output / size, input]`); an error if the output
    /// features do not split evenly.
    pub fn from_linear(linear: &Linear, group: &Group) -> Result<Self> {
        let (out, _) = dims(&linear.weight);
        let (a, b) = slice("AllToShardedLinear", "output features", out, 1, group)?;
        Ok(Self {
            shard: Shard::Linear(Linear {
                weight: Param::new(linear.weight.i((a..b, ..))),
                bias: linear.bias.as_ref().map(|p| Param::new(p.i(a..b))),
            }),
            group: group.clone(),
        })
    }

    /// This rank's rows of a quantized layer (packed weight, scales and biases).
    pub fn from_quantized(layer: &QuantizedLinear, group: &Group) -> Result<Self> {
        let (out, _) = dims(&layer.weight);
        let (a, b) = slice("AllToShardedLinear", "output features", out, 1, group)?;
        let rows = |x: &Array| x.i((a..b, ..));
        Ok(Self {
            shard: Shard::Quantized(QuantizedLinear {
                weight: rows(&layer.weight),
                scales: rows(&layer.scales),
                biases: layer.biases.as_ref().map(rows),
                bias: layer.bias.as_ref().map(|p| Param::new(p.i(a..b))),
                quant: layer.quant,
            }),
            group: group.clone(),
        })
    }

    /// This rank's output features, `[..., output / size]`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        self.shard
            .forward(&distributed::sum_gradients(x, Some(&self.group)))
    }
}

/// A linear layer whose input features are split across `group` (MLX's
/// `ShardedToAllLinear`): each rank multiplies its slice of the input by its columns of
/// the weight, and the partial products are summed over the group (`all_sum`) before the
/// bias is added once.
#[derive(Debug, Clone, Module)]
pub struct ShardedToAllLinear {
    /// This rank's columns of the weight (no bias).
    pub shard: Shard,
    /// The full bias, `[output]`.
    pub bias: Option<Param>,
    /// The group the features are split across.
    #[tree(skip)]
    pub group: Group,
}

impl ShardedToAllLinear {
    /// This rank's columns of `linear` (`[output, input / size]`); an error if the input
    /// features do not split evenly.
    pub fn from_linear(linear: &Linear, group: &Group) -> Result<Self> {
        let (_, input) = dims(&linear.weight);
        let (a, b) = slice("ShardedToAllLinear", "input features", input, 1, group)?;
        Ok(Self {
            shard: Shard::Linear(Linear {
                weight: Param::new(linear.weight.i((.., a..b))),
                bias: None,
            }),
            bias: linear.bias.clone(),
            group: group.clone(),
        })
    }

    /// This rank's columns of a quantized layer: the input features split in whole
    /// quantization groups.
    pub fn from_quantized(layer: &QuantizedLinear, group: &Group) -> Result<Self> {
        let (_, packed) = dims(&layer.weight);
        let bits = layer.quant.bits.max(1);
        let input = packed * 32 / bits;
        let group_size = layer.quant.group_size.max(1);
        let (a, b) = slice(
            "ShardedToAllLinear",
            "input features",
            input,
            group_size,
            group,
        )?;
        let (pa, pb) = (a * bits / 32, b * bits / 32);
        let (sa, sb) = (a / group_size, b / group_size);
        let cols = |x: &Array| x.i((.., sa..sb));
        Ok(Self {
            shard: Shard::Quantized(QuantizedLinear {
                weight: layer.weight.i((.., pa..pb)),
                scales: cols(&layer.scales),
                biases: layer.biases.as_ref().map(cols),
                bias: None,
                quant: layer.quant,
            }),
            bias: layer.bias.clone(),
            group: group.clone(),
        })
    }

    /// The full output from this rank's slice of the input features, `[..., input / size]`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        let y = distributed::all_sum(&self.shard.forward(x), Some(&self.group));
        match &self.bias {
            Some(b) => y.add(b),
            None => y,
        }
    }
}
