//! Batched decoding over a paged KV cache: several sequences, each with its own length,
//! produce their next tokens in one forward pass. Keys and values live in blocks of a
//! fixed number of positions ([`PagedCache`]); each sequence lists its blocks in a block
//! table, read by [`nuvec::fast::paged_attention`].
//!
//! Sequences are prefilled as usual (into [`KvCache`]s) and then moved into the paged
//! cache with [`PagedCache::insert`]. Gated DeltaNet layers keep each sequence's states at
//! its slot of a state pool ([`nuvec::fast::gated_delta_slots`]).
//!
//! A step is atomic: its new positions are reserved, the forward pass runs, and only then
//! are the lengths and blocks committed; a failed step frees what it reserved and leaves
//! the pools as they were.

use nuvec::{Array, Result, fast, ops};

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
    pub(crate) layers: Vec<PagedLayer>,
    blocks: usize,
    block_size: usize,
    max_sequences: usize,
    kv_heads: usize,
    head_dim: usize,
    free: Vec<u32>,
    sequences: Vec<Option<Sequence>>,
}

/// One new position reserved for each of a step's sequences: the blocks it allocated
/// (lengths unchanged), and the step's description. [`PagedCache::commit`] makes it part
/// of the sequences after a successful forward pass, [`PagedCache::rollback`] frees it.
#[derive(Debug)]
pub(crate) struct Reservation {
    ids: Vec<SeqId>,
    /// The block each sequence's new position needed, if any.
    blocks: Vec<Option<u32>>,
    pub(crate) step: PagedStep,
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
        let int = |v: usize| i32::try_from(v).map_err(|_| invalid(format!("{v} is too large")));
        Ok(Self {
            layers: vec![
                PagedLayer {
                    pool_size: (int(blocks)?, int(block_size)?),
                    state_slots: int(sequences)?,
                    ..Default::default()
                };
                c.num_hidden_layers
            ],
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

    /// Blocks in all (used or free).
    pub(crate) fn total_blocks(&self) -> usize {
        self.blocks
    }

    /// Positions per block.
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
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

    /// The slot (`block · block_size + position`) of position `p` in `blocks`.
    fn slot(&self, blocks: &[u32], p: usize) -> i32 {
        (blocks[p / self.block_size] as usize * self.block_size + p % self.block_size) as i32
    }

    /// Moves a prefilled sequence (`cache`, one [`KvCache`] per layer, batch 1) into the
    /// paged cache and returns its id. An empty cache (nothing prefilled, as for a
    /// one-token prompt) is a sequence of length 0: no blocks, zeroed states. When a write
    /// fails, the blocks are returned and the pools left as they were.
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
            let batch = match (c.linear_state(), c.keys_values()) {
                (Some((conv, _)), _) => conv.shape().first().copied(),
                (None, Some((k, _))) => k.shape().first().copied(),
                (None, None) if len == 0 => Some(1),
                (None, None) => return Err(invalid("a layer without keys, values or states")),
            };
            if batch != Some(1) || c.offset() != len {
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
        let saved = self.layers.clone();
        if let Err(e) = self.write(cache, id, &blocks, len) {
            self.layers = saved;
            self.free.extend(blocks);
            return Err(e);
        }
        drop(saved);
        if id == self.sequences.len() {
            self.sequences.push(None);
        }
        self.sequences[id] = Some(Sequence { blocks, len });
        Ok(id)
    }

    /// Writes a prefilled sequence's keys and values (positions `0..len`) into `blocks`
    /// and its states into slot `id` (zeros for an empty sequence).
    fn write(&mut self, cache: &[KvCache], id: SeqId, blocks: &[u32], len: usize) -> Result<()> {
        let slots: Vec<i32> = (0..len).map(|p| self.slot(blocks, p)).collect();
        let slots = (len > 0).then(|| Array::from(slots));
        let (kv, d, block_size, pool_blocks) =
            (self.kv_heads, self.head_dim, self.block_size, self.blocks);
        for (layer, c) in self.layers.iter_mut().zip(cache) {
            let state_slots = layer.state_slots;
            if let Some((conv, recurrent)) = c.linear_state() {
                // A Gated DeltaNet layer: its states into slot `id` of the pools.
                for (pool_slot, state) in
                    [(&mut layer.conv, conv), (&mut layer.recurrent, recurrent)]
                {
                    let pool_array = match pool_slot.take() {
                        Some(p) => p,
                        None => {
                            let mut shape = state.shape().to_vec();
                            shape[0] = state_slots;
                            ops::zeros(shape, state.dtype())
                        }
                    };
                    *pool_slot = Some(pool_array.at(id..id + 1).set(&state).check()?);
                }
                continue;
            }
            let Some(slots) = &slots else {
                // An empty sequence: a previous sequence's states at its slot are zeroed
                // (pools not allocated yet start as zeros).
                for pool_slot in [&mut layer.conv, &mut layer.recurrent] {
                    if let Some(pool_array) = pool_slot.take() {
                        *pool_slot = Some(pool_array.at(id..id + 1).set(0.0).check()?);
                    }
                }
                continue;
            };
            let (k, v) = c.keys_values().expect("checked by insert");
            for (pool_slot, new) in [(&mut layer.keys, k), (&mut layer.values, v)] {
                let pool_array = match pool_slot.take() {
                    Some(p) => p,
                    None => ops::zeros([pool_blocks, kv, block_size, d], new.dtype()),
                };
                // `[1, Hkv, len, D]` as rows `[len, Hkv, D]`.
                let rows = new.squeeze(0).transpose([1, 0, 2]);
                *pool_slot = Some(fast::paged_write(&pool_array, &rows, slots).check()?);
            }
        }
        Ok(())
    }

    /// Frees sequence `id`'s blocks.
    pub fn remove(&mut self, id: SeqId) -> Result<()> {
        self.sequence(id)?;
        let sequence = self.sequences[id].take().expect("checked");
        self.free.extend(sequence.blocks);
        Ok(())
    }

    /// Reserves one new position for each of `ids` (all or nothing: the blocks they need
    /// are allocated, their lengths unchanged) and describes the step.
    pub(crate) fn reserve(&mut self, ids: &[SeqId]) -> Result<Reservation> {
        for (i, &id) in ids.iter().enumerate() {
            self.sequence(id)?;
            if ids[..i].contains(&id) {
                return Err(invalid(format!("sequence {id} twice in one step")));
            }
        }
        let lens: Vec<usize> = ids
            .iter()
            .map(|&id| self.sequences[id].as_ref().expect("checked").len)
            .collect();
        let needed: usize = lens.iter().map(|&len| self.blocks_needed(len, 1)).sum();
        let mut new = self.allocate(needed)?.into_iter();
        let mut reserved = Vec::with_capacity(ids.len());
        let mut tables_of = Vec::with_capacity(ids.len());
        let mut slots = Vec::with_capacity(ids.len());
        let mut positions = Vec::with_capacity(ids.len());
        for (&id, &len) in ids.iter().zip(&lens) {
            let block = (self.blocks_needed(len, 1) == 1).then(|| new.next().expect("allocated"));
            let mut table = self.sequences[id].as_ref().expect("checked").blocks.clone();
            table.extend(block);
            slots.push(self.slot(&table, len));
            positions.push(len as i32);
            reserved.push(block);
            tables_of.push(table);
        }
        let width = tables_of.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut tables = vec![0i32; ids.len() * width];
        for (i, table) in tables_of.iter().enumerate() {
            for (j, &block) in table.iter().enumerate() {
                tables[i * width + j] = block as i32;
            }
        }
        let lengths: Vec<i32> = lens.iter().map(|&len| len as i32 + 1).collect();
        let b = ids.len();
        let state_slots: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let step = PagedStep {
            slots: Array::from(slots),
            positions: Array::from(positions),
            lengths: Array::from(lengths),
            tables: Array::from_slice(&tables, [b, width]),
            state_slots: Array::from(state_slots),
        };
        Ok(Reservation {
            ids: ids.to_vec(),
            blocks: reserved,
            step,
        })
    }

    /// Extends each sequence of a reservation by its new position (after the forward
    /// pass succeeded). A sequence removed meanwhile gets its block freed.
    pub(crate) fn commit(&mut self, reservation: Reservation) {
        for (id, block) in reservation.ids.into_iter().zip(reservation.blocks) {
            match self.sequences.get_mut(id).and_then(Option::as_mut) {
                Some(sequence) => {
                    sequence.blocks.extend(block);
                    sequence.len += 1;
                }
                None => self.free.extend(block),
            }
        }
    }

    /// Frees a reservation's blocks (after the forward pass failed).
    pub(crate) fn rollback(&mut self, reservation: Reservation) {
        self.free.extend(reservation.blocks.into_iter().flatten());
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
/// On an error the cache is left as it was.
pub fn decode_batch(
    model: &Model,
    tokens: &[u32],
    ids: &[SeqId],
    cache: &mut PagedCache,
) -> Result<Array> {
    let reservation = cache.reserve(ids)?;
    match forward_reserved(model, &Array::from(tokens.to_vec()), &reservation, cache) {
        Ok(logits) => {
            cache.commit(reservation);
            Ok(logits)
        }
        Err(e) => {
            cache.rollback(reservation);
            Err(e)
        }
    }
}

/// The logits `[B, vocab]` of a reserved step (not committed), after `tokens` (`[B]`
/// token ids, maybe not computed yet: a pipelined step feeds the previous step's tokens).
pub(crate) fn forward_reserved(
    model: &Model,
    tokens: &Array,
    reservation: &Reservation,
    cache: &mut PagedCache,
) -> Result<Array> {
    let b = tokens.shape().first().copied().unwrap_or(0);
    if tokens.ndim() != 1 || b as usize != reservation.ids.len() || b == 0 {
        return Err(invalid(format!(
            "{:?} tokens for {} sequences",
            tokens.shape(),
            reservation.ids.len()
        )));
    }
    let input = tokens.reshape([b, 1]);
    let logits = model.forward_paged(&input, &mut cache.layers, &reservation.step)?;
    logits.i((.., 0)).check()
}
