//! Batched decoding over a paged KV cache: several sequences, each with its own length,
//! produce their next tokens in one forward pass. Keys and values live in blocks of a
//! fixed number of positions ([`PagedCache`]); each sequence lists its blocks in a block
//! table, read by [`nuvec::fast::paged_attention`].
//!
//! Sequences are prefilled as usual (into [`KvCache`]s) and then moved into the paged
//! cache with [`PagedCache::insert`]. Gated DeltaNet layers keep each sequence's states at
//! its slot of a state pool ([`nuvec::fast::gated_delta_slots`]).

use nuvec::{Array, DType, Result, SliceArg, fast, slice, slice_update, squeeze, transpose, zeros};

use crate::model::{KvCache, Model, PagedLayer, PagedStep};

/// A sequence of a [`PagedCache`].
pub type SeqId = usize;

#[derive(Debug, Clone)]
struct Sequence {
    blocks: Vec<u32>,
    len: usize,
}

/// Keys and values of many sequences in blocks of `block_size` positions, one pool of
/// `blocks` blocks per attention layer (`[blocks, kv_heads, block_size, head_dim]`), with
/// each sequence's blocks and length; Gated DeltaNet layers keep one state slot per
/// sequence (up to `sequences`).
#[derive(Debug, Clone)]
pub struct PagedCache {
    layers: Vec<PagedLayer>,
    blocks: usize,
    block_size: usize,
    max_sequences: usize,
    kv_heads: usize,
    head_dim: usize,
    free: Vec<u32>,
    sequences: Vec<Option<Sequence>>,
}

fn invalid(msg: impl std::fmt::Display) -> nuvec::Error {
    nuvec::Error::InvalidArgument(format!("paged cache: {msg}"))
}

impl PagedCache {
    /// An empty cache of `blocks` blocks of `block_size` positions (a power of two) per
    /// attention layer of `model`, for up to `sequences` sequences at a time. The pools
    /// are allocated with the first sequence, in its states' dtypes.
    pub fn new(model: &Model, blocks: usize, block_size: usize, sequences: usize) -> Result<Self> {
        if !block_size.is_power_of_two() || blocks == 0 || sequences == 0 {
            return Err(invalid(
                "blocks of a power-of-two size, at least one block and one sequence",
            ));
        }
        let c = &model.config;
        Ok(Self {
            layers: vec![PagedLayer::default(); c.num_hidden_layers],
            blocks,
            block_size,
            max_sequences: sequences,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            free: (0..blocks as u32).rev().collect(),
            sequences: Vec::new(),
        })
    }

    /// Blocks not used by any sequence.
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// The positions sequence `id` holds.
    pub fn len(&self, id: SeqId) -> Option<usize> {
        self.sequence(id).ok().map(|s| s.len)
    }

    /// Whether no sequence is held.
    pub fn is_empty(&self) -> bool {
        self.sequences.iter().all(Option::is_none)
    }

    fn sequence(&self, id: SeqId) -> Result<&Sequence> {
        self.sequences
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid(format!("no sequence {id}")))
    }

    /// Blocks to hold `positions` more positions after `len`.
    fn blocks_needed(&self, len: usize, positions: usize) -> usize {
        (len + positions).div_ceil(self.block_size) - len.div_ceil(self.block_size)
    }

    fn allocate(&mut self, n: usize) -> Result<Vec<u32>> {
        if n > self.free.len() {
            return Err(invalid(format!(
                "{n} more blocks needed, {} free",
                self.free.len()
            )));
        }
        Ok(self.free.split_off(self.free.len() - n))
    }

    /// Slots (`block · block_size + position`) of positions `start..end` of `sequence`.
    fn slots(&self, sequence: &Sequence, start: usize, end: usize) -> Vec<i32> {
        (start..end)
            .map(|p| {
                (sequence.blocks[p / self.block_size] as usize * self.block_size
                    + p % self.block_size) as i32
            })
            .collect()
    }

    /// Moves a prefilled sequence (`cache`, one [`KvCache`] per layer, batch 1) into the
    /// paged cache and returns its id.
    pub fn insert(&mut self, cache: &[KvCache]) -> Result<SeqId> {
        if cache.len() != self.layers.len() {
            return Err(invalid(format!(
                "{} layer caches for {} layers",
                cache.len(),
                self.layers.len()
            )));
        }
        let len = cache.first().map_or(0, KvCache::offset);
        // Checked before anything is allocated.
        for c in cache {
            let batch = match (c.linear_state(), c.keys_values()?) {
                (Some((conv, _)), _) => conv.shape()[0],
                (None, Some((k, _))) => k.shape()[0],
                (None, None) => return Err(invalid("a layer without keys, values or states")),
            };
            if batch != 1 || c.offset() != len {
                return Err(invalid(
                    "prefilled caches of one sequence, all of its length",
                ));
            }
        }
        let id = match self.sequences.iter().position(Option::is_none) {
            Some(free) => free,
            None if self.sequences.len() < self.max_sequences => self.sequences.len(),
            None => return Err(invalid(format!("{} sequences already", self.max_sequences))),
        };
        let blocks = self.allocate(self.blocks_needed(0, len))?;
        let sequence = Sequence { blocks, len };
        let slots = self.slots(&sequence, 0, len);
        let slots = Array::from_slice(&slots, &[len])?;
        let (kv, d) = (self.kv_heads, self.head_dim);
        let max_sequences = self.max_sequences;
        for (layer, c) in self.layers.iter_mut().zip(cache) {
            if let Some((conv, recurrent)) = c.linear_state() {
                // A Gated DeltaNet layer: its states into slot `id` of the pools.
                for (pool_slot, state) in
                    [(&mut layer.conv, conv), (&mut layer.recurrent, recurrent)]
                {
                    let pool_array = match pool_slot.take() {
                        Some(p) => p,
                        None => {
                            let mut shape = state.shape().to_vec();
                            shape[0] = max_sequences;
                            zeros(&shape, state.dtype())?
                        }
                    };
                    let mut region = vec![SliceArg::full(); state.ndim()];
                    region[0] = SliceArg::new(Some(id as isize), Some(id as isize + 1), 1);
                    *pool_slot = Some(slice_update(&pool_array, &state, &region)?);
                }
                continue;
            }
            let (k, v) = c.keys_values()?.expect("checked above");
            let pool = |dtype: DType| zeros(&[self.blocks, kv, self.block_size, d], dtype);
            for (pool_slot, new) in [(&mut layer.keys, k), (&mut layer.values, v)] {
                let pool_array = match pool_slot.take() {
                    Some(p) => p,
                    None => pool(new.dtype())?,
                };
                // `[1, Hkv, len, D]` as rows `[len, Hkv, D]`.
                let rows = transpose(&squeeze(&new, &[0])?, &[1, 0, 2])?;
                *pool_slot = Some(fast::paged_write(&pool_array, &rows, &slots)?);
            }
        }
        if id == self.sequences.len() {
            self.sequences.push(None);
        }
        self.sequences[id] = Some(sequence);
        Ok(id)
    }

    /// Frees sequence `id`'s blocks.
    pub fn remove(&mut self, id: SeqId) -> Result<()> {
        self.sequence(id)?;
        let sequence = self.sequences[id].take().expect("checked");
        self.free.extend(sequence.blocks);
        Ok(())
    }

    /// Reserves one new position for each of `ids` and describes the step.
    fn step(&mut self, ids: &[SeqId]) -> Result<PagedStep> {
        for (i, &id) in ids.iter().enumerate() {
            self.sequence(id)?;
            if ids[..i].contains(&id) {
                return Err(invalid(format!("sequence {id} twice in one step")));
            }
        }
        // All or nothing: enough free blocks for every sequence's new position.
        let needed: usize = ids
            .iter()
            .map(|&id| self.blocks_needed(self.sequences[id].as_ref().expect("checked").len, 1))
            .sum();
        if needed > self.free.len() {
            return Err(invalid(format!(
                "{needed} more blocks needed, {} free",
                self.free.len()
            )));
        }
        let mut slots = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        for &id in ids {
            let len = self.sequence(id)?.len;
            let more = self.blocks_needed(len, 1);
            let new = self.allocate(more)?;
            let sequence = self.sequences[id].as_mut().expect("checked");
            sequence.blocks.extend(new);
            sequence.len += 1;
            let sequence = self.sequences[id].as_ref().expect("checked");
            slots.push(self.slots(sequence, len, len + 1)[0]);
            positions.push(len as i32);
        }
        let width = ids
            .iter()
            .map(|&id| self.sequences[id].as_ref().expect("checked").blocks.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut tables = vec![0i32; ids.len() * width];
        let mut lengths = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            let s = self.sequences[id].as_ref().expect("checked");
            for (j, &block) in s.blocks.iter().enumerate() {
                tables[i * width + j] = block as i32;
            }
            lengths.push(s.len as i32);
        }
        let b = ids.len();
        let state_slots: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        Ok(PagedStep {
            slots: Array::from_slice(&slots, &[b])?,
            positions: Array::from_slice(&positions, &[b])?,
            lengths: Array::from_slice(&lengths, &[b])?,
            tables: Array::from_slice(&tables, &[b, width])?,
            state_slots: Array::from_slice(&state_slots, &[b])?,
        })
    }

    /// The cached arrays (to evaluate them).
    pub fn state(&self) -> Vec<Array> {
        self.layers
            .iter()
            .flat_map(|l| [&l.keys, &l.values, &l.conv, &l.recurrent])
            .flatten()
            .cloned()
            .collect()
    }
}

/// The logits `[B, vocab]` of the next token of each sequence in `ids` after `tokens` (one
/// per sequence, the token at each one's next position), extending each sequence by it.
pub fn decode_batch(
    model: &Model,
    tokens: &[u32],
    ids: &[SeqId],
    cache: &mut PagedCache,
) -> Result<Array> {
    if tokens.len() != ids.len() || ids.is_empty() {
        return Err(invalid(format!(
            "{} tokens for {} sequences",
            tokens.len(),
            ids.len()
        )));
    }
    let step = cache.step(ids)?;
    let b = ids.len();
    let input = Array::from_slice(tokens, &[b, 1])?;
    let logits = model.forward_paged(&input, &mut cache.layers, &step)?;
    squeeze(
        &slice(
            &logits,
            &[
                SliceArg::full(),
                SliceArg::new(Some(0), Some(1), 1),
                SliceArg::full(),
            ],
        )?,
        &[1],
    )
}
