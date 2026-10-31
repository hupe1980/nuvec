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

use crate::model::{GroupStep, KvCache, Model, PagedKind, PagedLayer, PagedStep};

/// A sequence of a [`PagedCache`].
pub type SeqId = usize;

/// The layers that keep the same positions: all of them (full attention), or the last
/// `W − 1` before the next query (a sliding window of `W` keys).
#[derive(Debug, Clone)]
struct Group {
    window: Option<usize>,
    /// Blocks of this group's pools.
    blocks: usize,
    free: Vec<u32>,
}

#[derive(Debug, Clone)]
struct Sequence {
    /// Per group: the block of each logical block (`len / block_size` of them; blocks a
    /// window no longer reads are freed and their entries are 0, below `live[g]`).
    tables: Vec<Vec<u32>>,
    /// Per group: the first logical block still held.
    live: Vec<usize>,
    len: usize,
}

/// Keys and values of many sequences in blocks of `block_size` positions, with each
/// sequence's block tables and length; Gated DeltaNet layers keep one state slot per
/// sequence (up to `sequences`).
///
/// Layers are grouped by what they keep: full-attention layers share `blocks` blocks per
/// layer; sliding-window layers (window `W`) keep only the blocks their window reads, so
/// their pools hold `sequences · (⌈(W − 1) / block_size⌉ + 2)` blocks whatever the
/// sequences' lengths, and blocks before the window are freed after each step. KV-shared
/// layers (Gemma 4) have no pools: they read their source layer's.
#[derive(Debug, Clone)]
pub struct PagedCache {
    pub(crate) layers: Vec<PagedLayer>,
    groups: Vec<Group>,
    /// Each layer's group (`None`: no keys and values of its own).
    layer_group: Vec<Option<usize>>,
    /// The Gated DeltaNet layers (states at each sequence's slot).
    states: Vec<bool>,
    /// The group the batcher budgets (the full-attention one, if any).
    primary: usize,
    block_size: usize,
    max_sequences: usize,
    sequences: Vec<Option<Sequence>>,
}

/// One new position reserved for each of a step's sequences: the blocks it allocated per
/// group (lengths unchanged), and the step's description. [`PagedCache::commit`] makes it
/// part of the sequences after a successful forward pass, [`PagedCache::rollback`] frees
/// it.
#[derive(Debug)]
pub(crate) struct Reservation {
    ids: Vec<SeqId>,
    /// Per sequence and group: the block the new position needed, if any.
    blocks: Vec<Vec<Option<u32>>>,
    pub(crate) step: PagedStep,
}

fn invalid(msg: impl std::fmt::Display) -> nuvec::Error {
    nuvec::Error::InvalidArgument(format!("paged cache: {msg}"))
}

impl PagedCache {
    /// An empty cache of `blocks` blocks of `block_size` positions (a power of two) per
    /// full-attention layer of `model` (sliding-window layers hold what their window needs),
    /// for up to `sequences` sequences at a time. The pools are allocated with the first
    /// sequence, in its states' dtypes.
    pub fn new(model: &Model, blocks: usize, block_size: usize, sequences: usize) -> Result<Self> {
        if !block_size.is_power_of_two() || blocks == 0 || sequences == 0 {
            return Err(invalid(
                "blocks of a power-of-two size, at least one block and one sequence",
            ));
        }
        let int = |v: usize| i32::try_from(v).map_err(|_| invalid(format!("{v} is too large")));
        let mut groups: Vec<Group> = Vec::new();
        let mut layer_group = Vec::new();
        let kinds = model.paged_layers();
        for &kind in &kinds {
            let PagedKind::Keys(window) = kind else {
                layer_group.push(None);
                continue;
            };
            let g = match groups.iter().position(|g| g.window == window) {
                Some(g) => g,
                None => {
                    let n = match window {
                        None => blocks,
                        Some(w) => sequences * ((w.max(1) - 1).div_ceil(block_size) + 2),
                    };
                    groups.push(Group {
                        window,
                        blocks: n,
                        free: (0..n as u32).rev().collect(),
                    });
                    groups.len() - 1
                }
            };
            layer_group.push(Some(g));
        }
        if !groups.iter().any(|g| g.window.is_none()) {
            // No full-attention layer: a full group without layers still carries the
            // `blocks` budget the batcher admits by (window groups never run out).
            groups.push(Group {
                window: None,
                blocks,
                free: (0..blocks as u32).rev().collect(),
            });
        }
        let primary = groups.iter().position(|g| g.window.is_none()).unwrap_or(0);
        let layers = layer_group
            .iter()
            .map(|g| {
                Ok(PagedLayer {
                    pool_size: (
                        int(g.map_or(blocks, |g| groups[g].blocks))?,
                        int(block_size)?,
                    ),
                    state_slots: int(sequences)?,
                    group: g.unwrap_or(0),
                    ..Default::default()
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            layers,
            groups,
            layer_group,
            states: kinds.iter().map(|&k| k == PagedKind::States).collect(),
            primary,
            block_size,
            max_sequences: sequences,
            sequences: Vec::new(),
        })
    }

    /// Blocks not used by any sequence (of the full-attention layers).
    pub fn free_blocks(&self) -> usize {
        self.groups[self.primary].free.len()
    }

    /// Free blocks of group `group` (0 for the first group of layers, in layer order) and
    /// its window (`None`: full attention); `None` past the last group.
    pub fn free_blocks_in(&self, group: usize) -> Option<(usize, Option<usize>)> {
        self.groups.get(group).map(|g| (g.free.len(), g.window))
    }

    /// Blocks in all (used or free, of the full-attention layers).
    pub(crate) fn total_blocks(&self) -> usize {
        self.groups[self.primary].blocks
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

    /// The first position group `g` keeps of a sequence of `len` positions (the next
    /// query, at `len`, reads from `len − W + 1`).
    fn first_kept(&self, g: usize, len: usize) -> usize {
        self.groups[g]
            .window
            .map_or(0, |w| (len + 1).saturating_sub(w))
    }

    fn allocate(&mut self, g: usize, n: usize) -> Result<Vec<u32>> {
        let free = &mut self.groups[g].free;
        if n > free.len() {
            return Err(invalid(format!(
                "{n} more blocks needed, {} free",
                free.len()
            )));
        }
        Ok(free.split_off(free.len() - n))
    }

    /// The slot of position `p` in `table`.
    fn slot(&self, table: &[u32], p: usize) -> i32 {
        (table[p / self.block_size] as usize * self.block_size + p % self.block_size) as i32
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
        for ((c, g), &states) in cache.iter().zip(&self.layer_group).zip(&self.states) {
            let batch = match (c.linear_state(), c.keys_values()) {
                (Some((conv, _)), _) => conv.shape().first().copied(),
                (None, Some((k, _))) => k.shape().first().copied(),
                // An empty sequence, or a KV-shared layer (nothing of its own).
                (None, None) if len == 0 || (g.is_none() && !states) => Some(1),
                (None, None) => return Err(invalid("a layer without keys, values or states")),
            };
            if batch != Some(1) || c.offset() != len {
                return Err(invalid(
                    "prefilled caches of one sequence, all of its length",
                ));
            }
            if let Some(g) = g
                && c.start() > self.first_kept(*g, len)
            {
                return Err(invalid("a sliding-window cache without its window"));
            }
        }
        let id = match self.sequences.iter().position(Option::is_none) {
            Some(free) => free,
            None if self.sequences.len() < self.max_sequences => self.sequences.len(),
            None => return Err(invalid(format!("{} sequences already", self.max_sequences))),
        };
        // Per group: the blocks of the kept positions (dead entries 0 before them).
        let mut tables = Vec::with_capacity(self.groups.len());
        let mut live = Vec::with_capacity(self.groups.len());
        for g in 0..self.groups.len() {
            let first = self.first_kept(g, len) / self.block_size;
            let n = len.div_ceil(self.block_size).saturating_sub(first);
            match self.allocate(g, n) {
                Ok(blocks) => {
                    let mut table = vec![0; first];
                    table.extend(blocks);
                    tables.push(table);
                    live.push(first);
                }
                Err(e) => {
                    self.release(&tables, &live);
                    return Err(e);
                }
            }
        }
        let saved = self.layers.clone();
        if let Err(e) = self.write(cache, id, &tables, len) {
            self.layers = saved;
            self.release(&tables, &live);
            return Err(e);
        }
        drop(saved);
        if id == self.sequences.len() {
            self.sequences.push(None);
        }
        self.sequences[id] = Some(Sequence { tables, live, len });
        Ok(id)
    }

    /// Returns the live blocks of `tables` to their groups.
    fn release(&mut self, tables: &[Vec<u32>], live: &[usize]) {
        for (g, (table, &first)) in tables.iter().zip(live).enumerate() {
            self.groups[g].free.extend(&table[first..]);
        }
    }

    /// Writes a prefilled sequence's keys and values (the positions each group keeps) into
    /// its blocks and its states into slot `id` (zeros for an empty sequence).
    fn write(
        &mut self,
        cache: &[KvCache],
        id: SeqId,
        tables: &[Vec<u32>],
        len: usize,
    ) -> Result<()> {
        let firsts: Vec<usize> = (0..self.groups.len())
            .map(|g| self.first_kept(g, len))
            .collect();
        let slots: Vec<Option<Array>> = tables
            .iter()
            .zip(&firsts)
            .map(|(table, &first)| {
                (first < len).then(|| {
                    Array::from(
                        (first..len)
                            .map(|p| self.slot(table, p))
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .collect();
        let block_size = self.block_size;
        for ((layer, c), g) in self.layers.iter_mut().zip(cache).zip(&self.layer_group) {
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
            let Some(g) = *g else {
                // A Gated DeltaNet layer of an empty sequence: a previous sequence's states
                // at its slot are zeroed (pools not allocated yet start as zeros). A
                // KV-shared layer has no pools.
                for pool_slot in [&mut layer.conv, &mut layer.recurrent] {
                    if let Some(pool_array) = pool_slot.take() {
                        *pool_slot = Some(pool_array.at(id..id + 1).set(0.0).check()?);
                    }
                }
                continue;
            };
            let Some(slots) = &slots[g] else {
                continue; // an empty sequence: no positions to write
            };
            let (k, v) = c.keys_values().expect("checked by insert");
            // The kept positions of the cache's (which starts at `c.start()`).
            let from = (firsts[g] - c.start()) as i32;
            let pool_blocks = layer.pool_size.0;
            for (pool_slot, new) in [(&mut layer.keys, k), (&mut layer.values, v)] {
                let [_, heads, _, dims] = new.shape_n::<4>().unwrap_or_default();
                let pool_array = match pool_slot.take() {
                    Some(p) => p,
                    None => ops::zeros([pool_blocks, heads, block_size as i32, dims], new.dtype()),
                };
                // `[1, Hkv, len, D]` as rows `[len, Hkv, D]`.
                let rows = new.i((.., .., from..)).squeeze(0).transpose([1, 0, 2]);
                *pool_slot = Some(fast::paged_write(&pool_array, &rows, slots).check()?);
            }
        }
        Ok(())
    }

    /// Frees sequence `id`'s blocks.
    pub fn remove(&mut self, id: SeqId) -> Result<()> {
        self.sequence(id)?;
        let sequence = self.sequences[id].take().expect("checked");
        self.release(&sequence.tables, &sequence.live);
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
        let block_size = self.block_size;
        let new_block = |len: usize| len.is_multiple_of(block_size);
        let needed = lens.iter().filter(|&&len| new_block(len)).count();
        let mut allocated: Vec<std::vec::IntoIter<u32>> = Vec::with_capacity(self.groups.len());
        for g in 0..self.groups.len() {
            match self.allocate(g, needed) {
                Ok(blocks) => allocated.push(blocks.into_iter()),
                Err(e) => {
                    for (g, blocks) in allocated.into_iter().enumerate() {
                        self.groups[g].free.extend(blocks);
                    }
                    return Err(e);
                }
            }
        }
        let mut reserved = Vec::with_capacity(ids.len());
        let mut groups = Vec::with_capacity(self.groups.len());
        let mut per_group: Vec<(Vec<i32>, Vec<Vec<u32>>)> =
            vec![(Vec::new(), Vec::new()); self.groups.len()];
        for (&id, &len) in ids.iter().zip(&lens) {
            let sequence = self.sequences[id].as_ref().expect("checked");
            let mut blocks = Vec::with_capacity(self.groups.len());
            for (g, (slots, tables)) in per_group.iter_mut().enumerate() {
                let block = new_block(len).then(|| allocated[g].next().expect("allocated"));
                let mut table = sequence.tables[g].clone();
                table.extend(block);
                slots.push(self.slot(&table, len));
                tables.push(table);
                blocks.push(block);
            }
            reserved.push(blocks);
        }
        let b = ids.len();
        for (slots, tables_of) in per_group {
            let width = tables_of.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let mut tables = vec![0i32; b * width];
            for (i, table) in tables_of.iter().enumerate() {
                for (j, &block) in table.iter().enumerate() {
                    tables[i * width + j] = block as i32;
                }
            }
            groups.push(GroupStep {
                slots: Array::from(slots),
                tables: Array::from_slice(&tables, [b, width]),
                host_tables: tables_of,
            });
        }
        let positions: Vec<i32> = lens.iter().map(|&len| len as i32).collect();
        let lengths: Vec<i32> = lens.iter().map(|&len| len as i32 + 1).collect();
        let state_slots: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let step = PagedStep {
            positions: Array::from(positions),
            lengths: Array::from(lengths),
            state_slots: Array::from(state_slots),
            groups,
            host_lengths: lens.iter().map(|&len| len + 1).collect(),
            block_size: self.block_size,
        };
        Ok(Reservation {
            ids: ids.to_vec(),
            blocks: reserved,
            step,
        })
    }

    /// Extends each sequence of a reservation by its new position (after the forward
    /// pass succeeded), and frees the blocks windows no longer read. A sequence removed
    /// meanwhile gets its blocks freed.
    pub(crate) fn commit(&mut self, reservation: Reservation) {
        let block_size = self.block_size;
        for (id, blocks) in reservation.ids.into_iter().zip(reservation.blocks) {
            let Some(mut sequence) = self.sequences.get_mut(id).and_then(Option::take) else {
                for (g, block) in blocks.into_iter().enumerate() {
                    self.groups[g].free.extend(block);
                }
                continue;
            };
            sequence.len += 1;
            for (g, block) in blocks.into_iter().enumerate() {
                sequence.tables[g].extend(block);
                // Blocks wholly before the next query's window.
                let dead = self.first_kept(g, sequence.len) / block_size;
                let table = &mut sequence.tables[g];
                for logical in sequence.live[g]..dead.min(table.len()) {
                    self.groups[g].free.push(table[logical]);
                    table[logical] = 0;
                }
                sequence.live[g] = sequence.live[g].max(dead.min(table.len()));
            }
            self.sequences[id] = Some(sequence);
        }
    }

    /// Frees a reservation's blocks (after the forward pass failed).
    pub(crate) fn rollback(&mut self, reservation: Reservation) {
        for blocks in reservation.blocks {
            for (g, block) in blocks.into_iter().enumerate() {
                self.groups[g].free.extend(block);
            }
        }
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
