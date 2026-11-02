//! Pipeline parallelism, as mlx-lm's `model.pipeline(group, split)`: each rank of a group
//! runs a contiguous range of the layers (rank 0 the last), receiving the previous stage's
//! activations and sending its own on. A model too large for one Mac runs across several,
//! over any distributed backend.
//!
//! Each rank reads only its stage's tensors ([`load_model_pipelined`]): its layers, the
//! embeddings on the first stage (the highest rank) and the final norm and output head on
//! the last (rank 0). Only rank 0 computes the head and samples; every rank receives its
//! tokens ([`Model::share_tokens`]), so `generate`, `decode_batch` and the
//! [`Batcher`](crate::Batcher) yield the same events on every rank. Every rank runs the
//! same program with the same requests.

use std::ops::Range;
use std::path::Path;

use nuvec::distributed::{self, Group};
use nuvec::{Array, DType, Error, Result};
use nuvec_nn::Module;

use crate::model::Model;

/// How a pipeline's layers are split across the ranks.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum Split {
    /// mlx-lm's default: the same number of layers per rank, the extras to the low ranks.
    #[default]
    Even,
    /// Layers per rank, rank 0 first (mlx-lm's `split`): one positive entry per rank,
    /// summing to the layer count.
    Layers(Vec<usize>),
    /// By memory: each rank offers its device's recommended working set minus its active
    /// memory, keeping `reserve` (a fraction, for the KV cache and activations) free.
    Memory {
        /// The fraction of each budget left for the KV cache and activations.
        reserve: f64,
    },
    /// By memory, with each rank's budget in bytes given (rank 0 first).
    Budgets(Vec<u64>, f64),
}

impl Split {
    /// [`Split::Memory`] with 15% of each budget kept free.
    pub fn memory() -> Self {
        Self::Memory { reserve: 0.15 }
    }
}

/// One rank's part of a pipeline.
#[derive(Debug, Clone)]
pub struct Stage {
    group: Group,
    rank: usize,
    size: usize,
    layers: Range<usize>,
    /// The dtype of the residual stream between stages (the first stage's embeddings'
    /// output, shared with every rank when the stage is set up).
    activations: DType,
}

impl Stage {
    /// The layers this rank runs.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Whether this rank embeds the tokens (the first stage: the highest rank).
    pub fn is_first(&self) -> bool {
        self.rank + 1 == self.size
    }

    /// Whether this rank computes the output head and samples (the last stage: rank 0).
    pub fn is_last(&self) -> bool {
        self.rank == 0
    }

    /// The group.
    pub fn group(&self) -> &Group {
        &self.group
    }

    /// The rank this stage receives its activations from (the stage before).
    pub(crate) fn previous(&self) -> usize {
        self.rank + 1
    }

    /// The rank this stage sends its activations to (the stage after).
    pub(crate) fn next(&self) -> usize {
        self.rank - 1
    }

    /// The dtype of the activations passed between stages.
    pub(crate) fn activations(&self) -> DType {
        self.activations
    }
}

/// The dtypes the residual stream can have, by code (for sharing it between ranks).
const ACTIVATION_DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];

/// Sets `stage`'s activation dtype from the first stage's embeddings: that rank measures
/// it (a lazy embedding of one token; nothing is evaluated) and every rank receives it.
pub(crate) fn share_activations(model: &Model, stage: &mut Stage) -> Result<()> {
    let code = if stage.is_first() {
        let probe = model.embed(&Array::from_slice(&[0u32], [1, 1]));
        let d = probe.dtype();
        ACTIVATION_DTYPES
            .iter()
            .position(|&x| x == d)
            .ok_or_else(|| Error::InvalidArgument(format!("pipeline: activations of {d:?}")))?
            as u32
            + 1
    } else {
        0
    };
    let all = distributed::all_sum(&Array::from_slice(&[code], [1]), Some(&stage.group));
    let code = all.to_vec::<u32>()?[0];
    stage.activations = *ACTIVATION_DTYPES
        .get(code.wrapping_sub(1) as usize)
        .ok_or_else(|| Error::InvalidArgument("pipeline: no activation dtype".into()))?;
    Ok(())
}

/// The layer ranges per rank (rank 0 first) for `counts` layers per rank: rank `r` runs the
/// layers after those of ranks `r + 1 .. size` (mlx-lm's order).
fn ranges(counts: &[usize]) -> Vec<Range<usize>> {
    (0..counts.len())
        .map(|r| {
            let start: usize = counts[r + 1..].iter().sum();
            start..start + counts[r]
        })
        .collect()
}

/// What the split needs to know about a model: each layer's bytes and the fixed bytes of
/// the first stage (embeddings) and the last (norm, head, and the embeddings when tied).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Sizes {
    pub(crate) layers: Vec<u64>,
    pub(crate) first: u64,
    pub(crate) last: u64,
}

/// Layers per rank (rank 0 first).
pub(crate) fn split_counts(split: &Split, sizes: &Sizes, size: usize) -> Result<Vec<usize>> {
    let n = sizes.layers.len();
    if size > n {
        return Err(Error::InvalidArgument(format!(
            "pipeline: {size} ranks for {n} layers (every rank needs a layer)"
        )));
    }
    match split {
        Split::Even => {
            let (base, extra) = (n / size, n % size);
            Ok((0..size).map(|r| base + usize::from(r < extra)).collect())
        }
        Split::Layers(counts) => {
            if counts.len() != size || counts.contains(&0) || counts.iter().sum::<usize>() != n {
                return Err(Error::InvalidArgument(format!(
                    "pipeline: split {counts:?} must have one positive entry per rank ({size}) \
                     summing to {n} layers"
                )));
            }
            Ok(counts.clone())
        }
        Split::Memory { .. } => Err(Error::InvalidArgument(
            "pipeline: the memory split needs the ranks' budgets".into(),
        )),
        Split::Budgets(budgets, reserve) => by_budget(budgets, *reserve, sizes, size),
    }
}

/// The contiguous split that fits each rank's usable budget: the first stage (the highest
/// rank) takes layers from the front until its budget is spent, then the next, and so on;
/// a shortfall is an error naming the bytes. When the budgets have room to spare, the
/// layers are spread in proportion to them.
fn by_budget(budgets: &[u64], reserve: f64, sizes: &Sizes, size: usize) -> Result<Vec<usize>> {
    if budgets.len() != size {
        return Err(Error::InvalidArgument(format!(
            "pipeline: {} budgets for {size} ranks",
            budgets.len()
        )));
    }
    if !(0.0..1.0).contains(&reserve) {
        return Err(Error::InvalidArgument(format!(
            "pipeline: reserve {reserve} must be in [0, 1)"
        )));
    }
    let usable: Vec<u64> = budgets
        .iter()
        .map(|&b| (b as f64 * (1.0 - reserve)) as u64)
        .collect();
    let fixed = |r: usize| {
        let mut f = 0;
        if r + 1 == size {
            f += sizes.first;
        }
        if r == 0 {
            f += sizes.last;
        }
        f
    };
    let need: u64 = sizes.layers.iter().sum::<u64>() + sizes.first + sizes.last;
    let have: u64 = usable.iter().sum();
    // What each rank has for layers beyond its fixed costs.
    let room: Vec<u64> = (0..size)
        .map(|r| usable[r].saturating_sub(fixed(r)))
        .collect();
    let total_room: u64 = room.iter().sum();
    let layer_bytes: u64 = sizes.layers.iter().sum();
    if have < need || total_room < layer_bytes {
        return Err(Error::InvalidArgument(format!(
            "pipeline: the group's budgets hold {have} of {need} bytes"
        )));
    }
    // First in proportion to the room, then (if that leaves rank 0 short) as many layers
    // per rank as fit, from the first stage on.
    fill(sizes, &room, total_room, true)
        .or_else(|| fill(sizes, &room, total_room, false))
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "pipeline: the group's budgets hold {have} of {need} bytes (no contiguous split \
                 fits each rank)"
            ))
        })
}

/// Layers per rank (rank 0 first), filling from the first stage (the highest rank): each
/// takes layers up to its share of `room` (`proportional`) or its whole room, leaving one
/// layer for every later rank; rank 0 takes the rest. `None` when something does not fit.
fn fill(sizes: &Sizes, room: &[u64], total_room: u64, proportional: bool) -> Option<Vec<usize>> {
    let (size, n) = (room.len(), sizes.layers.len());
    let layer_bytes: u64 = sizes.layers.iter().sum();
    let mut counts = vec![0usize; size];
    let mut at = 0usize;
    for r in (1..size).rev() {
        let target = if proportional {
            (layer_bytes as f64 * room[r] as f64 / total_room as f64) as u64
        } else {
            room[r]
        };
        let (mut used, mut k) = (0u64, 0usize);
        // At least one layer, and one left for each of ranks r − 1 .. 0.
        while at + k < n - r {
            let next = sizes.layers[at + k];
            if used + next > room[r] || (k > 0 && used + next > target) {
                break;
            }
            used += next;
            k += 1;
        }
        if k == 0 {
            return None;
        }
        counts[r] = k;
        at += k;
    }
    let rest: u64 = sizes.layers[at..].iter().sum();
    if rest > room[0] {
        return None;
    }
    counts[0] = n - at;
    Some(counts)
}

/// The decoder layer a tensor belongs to (`…layers.N.…`, as mlx-lm's `sharded_load` matches
/// names: checkpoints nest the text model differently); the multi-token-prediction head's
/// layers are not decoder layers.
fn layer_of(name: &str) -> Option<usize> {
    if name.starts_with("mtp.") {
        return None;
    }
    let at = name
        .match_indices("layers.")
        .find(|(i, _)| *i == 0 || name.as_bytes()[i - 1] == b'.')?
        .0;
    let rest = &name[at + "layers.".len()..];
    let digits = rest.split('.').next()?;
    digits.parse().ok()
}

/// Whether a tensor is the token embedding.
fn is_embedding(name: &str) -> bool {
    name.contains("embed_tokens.")
}

/// [`Sizes`] of `n` layers from `(name, bytes)` pairs.
fn sizes_of(n: usize, tied: bool, tensors: impl Iterator<Item = (String, u64)>) -> Sizes {
    let mut s = Sizes {
        layers: vec![0; n],
        first: 0,
        last: 0,
    };
    for (name, bytes) in tensors {
        if let Some(i) = layer_of(&name) {
            if let Some(l) = s.layers.get_mut(i) {
                *l += bytes;
            }
        } else if is_embedding(&name) {
            s.first += bytes;
            if tied {
                s.last += bytes;
            }
        } else {
            s.last += bytes;
        }
    }
    s
}

impl Model {
    /// Each layer's bytes and the stages' fixed bytes, from the parameters' shapes and
    /// dtypes (loaded or not).
    pub(crate) fn sizes(&self) -> Sizes {
        let params = self
            .parameters()
            .into_iter()
            .map(|(name, a)| (name, a.size() as u64 * a.dtype().size() as u64));
        sizes_of(
            self.config.num_hidden_layers,
            self.config.tie_word_embeddings,
            params,
        )
    }

    /// Makes this model rank `group.rank()`'s stage of a pipeline over `group`, as
    /// mlx-lm's `model.pipeline(group, split)`: the weights outside the stage are dropped.
    /// A group of size 1 leaves the model whole.
    pub fn pipeline(&mut self, group: &Group, split: Split) -> Result<()> {
        let stage = plan(self, group, split)?;
        if let Some(mut stage) = stage {
            share_activations(self, &mut stage)?;
            self.drop_outside(&stage)?;
            self.set_stage(stage);
        }
        Ok(())
    }

    /// The stage this model runs, when pipelined.
    pub fn stage(&self) -> Option<&Stage> {
        self.stage_ref()
    }

    /// [`share_tokens`](Self::share_tokens) for sampled tokens: a tensor-parallel model
    /// whose sampling is `deterministic` (greedy, or seeded) yields its own tokens, as
    /// mlx-lm: every rank samples the same ones from bit-identical logits, one collective
    /// per step fewer.
    #[track_caller]
    pub fn share_sampled(&self, tokens: &Array, deterministic: bool) -> Array {
        if deterministic && self.sharding().is_some() {
            return tokens.clone();
        }
        self.share_tokens(tokens)
    }

    /// Every rank's copy of rank 0's `tokens` (integers) on a pipelined model, in the order
    /// of the stage's other collectives; the tokens unchanged otherwise.
    #[track_caller]
    pub fn share_tokens(&self, tokens: &Array) -> Array {
        if let Some(group) = self.sharding() {
            // Every rank ran the same collectives to get here; rank 0's tokens win.
            let mine = tokens.astype(DType::U32);
            let mine = if group.rank() == 0 {
                mine
            } else {
                mine.multiply(0u32)
            };
            return distributed::all_sum(&mine, Some(group));
        }
        let Some(stage) = self.stage_ref() else {
            return tokens.clone();
        };
        // The other ranks' tokens were sampled from logits evaluated after their send: zero
        // them by arithmetic, so the reduction stays ordered after the send.
        let mine = tokens.astype(DType::U32);
        let mine = if stage.is_last() {
            mine
        } else {
            mine.multiply(0u32)
        };
        distributed::all_sum(&mine, Some(stage.group()))
    }
}

/// The stage of `group.rank()` for `model` under `split` (`None` for a group of one).
fn plan(model: &Model, group: &Group, split: Split) -> Result<Option<Stage>> {
    if group.size() == 1 {
        return Ok(None);
    }
    decide(&model.config, &model.sizes(), group, split).map(Some)
}

/// The stage of `group.rank()` (a group of more than one) for a model of `config` and `sizes`.
fn decide(config: &crate::Config, sizes: &Sizes, group: &Group, split: Split) -> Result<Stage> {
    if config.gemma4.is_some() {
        return Err(Error::InvalidArgument(
            "pipeline: Gemma 4 is not supported: its per-layer inputs and shared KV layers \
             would cross stages (as mlx-lm, which does not pipeline it either)"
                .into(),
        ));
    }
    let size = group.size();
    let split = match split {
        Split::Memory { reserve } => Split::Budgets(gather_budgets(group)?, reserve),
        s => s,
    };
    let counts = split_counts(&split, sizes, size)?;
    let rank = group.rank();
    Ok(Stage {
        group: group.clone(),
        rank,
        size,
        layers: ranges(&counts)[rank].clone(),
        activations: DType::F32,
    })
}

/// Every rank's budget: its device's recommended working set minus its active memory.
fn gather_budgets(group: &Group) -> Result<Vec<u64>> {
    let own = nuvec::metal::device_info().map_or(0, |d| {
        d.recommended_working_set
            .saturating_sub(nuvec::memory::active_memory() as u64)
    });
    let all = distributed::all_gather(&Array::from_slice(&[own], [1]), Some(group));
    all.to_vec::<u64>()
}

/// Whether `name` belongs to `stage` (its layers, and the embeddings, norm and heads its
/// position needs).
pub(crate) fn in_stage(name: &str, stage: &Stage, tied: bool) -> bool {
    if let Some(i) = layer_of(name) {
        return stage.layers.contains(&i);
    }
    if is_embedding(name) {
        return stage.is_first() || (stage.is_last() && tied);
    }
    stage.is_last()
}

/// [`crate::load_model`] for rank `group.rank()`'s stage of a pipeline over `group`: reads
/// only the stage's tensors (its layers; the embeddings on the first stage; the norm, the
/// head and any multi-token-prediction head on the last), deciding the split from the
/// checkpoint's headers before reading. A group of size 1 loads the whole model.
pub fn load_model_pipelined(dir: impl AsRef<Path>, group: &Group, split: Split) -> Result<Model> {
    let dir = dir.as_ref();
    if group.size() == 1 {
        return crate::load_model(dir);
    }
    let plan = |config: &crate::Config, index: &[nuvec::io::TensorInfo]| {
        let tensors = index.iter().map(|t| (t.name.clone(), t.bytes as u64));
        let sizes = sizes_of(
            config.num_hidden_layers,
            config.tie_word_embeddings,
            tensors,
        );
        decide(config, &sizes, group, split.clone())
    };
    crate::load_model_with(dir, Some(&plan))
}

/// [`crate::load_model`] then [`Model::shard`] over `group` (mlx-lm's `sharded_load` for
/// tensor parallelism): each rank keeps its slice of every sharded weight, copied, so the
/// full weights' memory is released (on memory-mapped checkpoints only the slices' pages
/// are read; misaligned ones are read whole once, as mlx-lm reads them).
pub fn load_model_sharded(dir: impl AsRef<Path>, group: &Group) -> Result<Model> {
    let mut model = crate::load_model(dir)?;
    model.shard(group)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(layers: &[u64], first: u64, last: u64) -> Sizes {
        Sizes {
            layers: layers.to_vec(),
            first,
            last,
        }
    }

    #[test]
    fn splits_as_mlx_lm() {
        let s = sizes(&[1; 10], 0, 0);
        assert_eq!(split_counts(&Split::Even, &s, 3).unwrap(), [4, 3, 3]);
        assert_eq!(ranges(&[4, 3, 3]), [6..10, 3..6, 0..3]);
        assert_eq!(
            split_counts(&Split::Layers(vec![2, 3, 5]), &s, 3).unwrap(),
            [2, 3, 5]
        );
        for bad in [vec![2, 8], vec![0, 5, 5], vec![2, 3, 4]] {
            let e = split_counts(&Split::Layers(bad), &s, 3)
                .unwrap_err()
                .to_string();
            assert!(e.contains("summing to 10 layers"), "{e}");
        }
        let e = split_counts(&Split::Even, &sizes(&[1; 2], 0, 0), 3)
            .unwrap_err()
            .to_string();
        assert!(e.contains("3 ranks for 2 layers"), "{e}");
    }

    #[test]
    fn names_map_to_stages() {
        assert_eq!(layer_of("model.layers.12.mlp.gate_proj.weight"), Some(12));
        assert_eq!(
            layer_of("model.language_model.layers.3.linear_attn.A_log"),
            Some(3)
        );
        assert_eq!(layer_of("mtp.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(layer_of("model.norm.weight"), None);
        assert_eq!(layer_of("model.sublayers.2.x"), None);
        let s = sizes_of(
            2,
            true,
            [
                ("model.embed_tokens.weight".to_string(), 10),
                ("model.layers.0.w".to_string(), 3),
                ("model.layers.1.w".to_string(), 4),
                ("model.norm.weight".to_string(), 1),
            ]
            .into_iter(),
        );
        assert_eq!((s.layers, s.first, s.last), (vec![3, 4], 10, 11));
    }

    #[test]
    fn splits_by_memory() {
        // Ten layers of 10 bytes, 30 bytes of embeddings (first stage) and of head (last).
        let s = sizes(&[10; 10], 30, 30);
        // Equal budgets: an even-ish split with the fixed costs where they belong.
        let c = split_counts(&Split::Budgets(vec![100, 100], 0.0), &s, 2).unwrap();
        assert_eq!(c.iter().sum::<usize>(), 10);
        assert!(c[0] * 10 + 30 <= 100 && c[1] * 10 + 30 <= 100, "{c:?}");
        // Uneven Macs: rank 1 (the first stage) has room for more.
        let c = split_counts(&Split::Budgets(vec![60, 200], 0.0), &s, 2).unwrap();
        assert!(c[0] * 10 + 30 <= 60, "{c:?}");
        assert_eq!(c.iter().sum::<usize>(), 10);
        // The reserve counts.
        let c = split_counts(&Split::Budgets(vec![200, 200], 0.5), &s, 2).unwrap();
        assert!(c[0] * 10 + 30 <= 100 && c[1] * 10 + 30 <= 100, "{c:?}");
        // Too small: an error naming the bytes, before any loading.
        let e = split_counts(&Split::Budgets(vec![50, 50], 0.0), &s, 2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("hold 100 of 160 bytes"), "{e}");
        // The same inputs give the same split (every rank computes it).
        let a = split_counts(&Split::Budgets(vec![90, 150, 70], 0.1), &s, 3).unwrap();
        let b = split_counts(&Split::Budgets(vec![90, 150, 70], 0.1), &s, 3).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.iter().sum::<usize>(), 10);
        assert!(a.iter().all(|&k| k > 0));
    }
}
