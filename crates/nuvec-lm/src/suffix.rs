//! SuffixDecoding (arXiv 2411.04975), chain drafts: a model-free drafter over the current
//! context and a bounded store of earlier sequences (prompts and outputs). It matches the
//! longest suffix of the context that occurred before, and drafts the most frequent
//! continuation among those occurrences, as long as the next token's share stays above a
//! threshold, for at most `alpha` times the match length.

use std::collections::{HashMap, VecDeque};

use nuvec::Result;

use crate::model::Model;
use crate::speculative::{Draft, Drafter};

/// The hash of `tokens`.
fn key(tokens: &[u32]) -> u64 {
    // FNV-1a over the tokens' bytes.
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |h, &t| {
        t.to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    })
}

/// SuffixDecoding's drafter (`Send`, not shared): the store of earlier sequences persists
/// across generations, so pass `&mut` to [`crate::speculative_generate`] and
/// [`SuffixDecoding::remember`] each finished request.
#[derive(Debug, Clone)]
pub struct SuffixDecoding {
    /// The shortest suffix matched (default 2; [`SuffixDecoding::with_min_match`]): the
    /// index is keyed by it.
    min_match: usize,
    /// The longest suffix compared (default 32).
    pub max_depth: usize,
    /// At most `alpha · match length` draft tokens (default 1.0).
    pub alpha: f32,
    /// A draft stops when its next token's share of the matching occurrences falls below
    /// this (default 0.1).
    pub min_share: f32,
    /// The most tokens the store keeps; the oldest sequences go first (default 1M).
    pub capacity: usize,
    /// Stored sequences, oldest first, with the id of the first.
    store: VecDeque<Vec<u32>>,
    first: usize,
    stored: usize,
    /// `min_match`-gram → (sequence id, end position) of each occurrence in the store.
    index: HashMap<u64, Vec<(usize, usize)>>,
    /// The current context's own occurrences, rebuilt when a new context starts.
    current: Vec<u32>,
    current_index: HashMap<u64, Vec<usize>>,
}

impl Default for SuffixDecoding {
    fn default() -> Self {
        Self {
            min_match: 2,
            max_depth: 32,
            alpha: 1.0,
            min_share: 0.1,
            capacity: 1 << 20,
            store: VecDeque::new(),
            first: 0,
            stored: 0,
            index: HashMap::new(),
            current: Vec::new(),
            current_index: HashMap::new(),
        }
    }
}

/// The most occurrences compared per draft (the most recent).
const CANDIDATES: usize = 256;

impl SuffixDecoding {
    /// The drafter matching suffixes of at least `min_match` tokens (at least 1), with an
    /// empty store.
    pub fn with_min_match(self, min_match: usize) -> Self {
        Self {
            min_match: min_match.max(1),
            store: VecDeque::new(),
            first: 0,
            stored: 0,
            index: HashMap::new(),
            current: Vec::new(),
            current_index: HashMap::new(),
            ..self
        }
    }

    /// Adds a finished sequence (a prompt and its output) to the store, evicting the oldest
    /// beyond the capacity.
    pub fn remember(&mut self, sequence: &[u32]) {
        let id = self.first + self.store.len();
        let m = self.min_match.max(1);
        for end in m..=sequence.len() {
            self.index
                .entry(key(&sequence[end - m..end]))
                .or_default()
                .push((id, end));
        }
        self.stored += sequence.len();
        self.store.push_back(sequence.to_vec());
        while self.stored > self.capacity && self.store.len() > 1 {
            let old = self.store.pop_front().expect("non-empty");
            self.stored -= old.len();
            let gone = self.first;
            self.first += 1;
            // Occurrences are listed in the order sequences were stored, so the evicted
            // sequence's are at the front of the lists of its own n-grams.
            for end in m..=old.len() {
                let k = key(&old[end - m..end]);
                if let Some(list) = self.index.get_mut(&k) {
                    let n = list.partition_point(|&(s, _)| s <= gone);
                    list.drain(..n);
                    if list.is_empty() {
                        self.index.remove(&k);
                    }
                }
            }
        }
    }

    /// Indexes `context`, continuing the current one when it extends it.
    fn track(&mut self, context: &[u32]) {
        let m = self.min_match.max(1);
        if !context.starts_with(&self.current) {
            self.current.clear();
            self.current_index.clear();
        }
        let from = self.current.len().max(m - 1) + 1;
        self.current
            .extend_from_slice(&context[self.current.len()..]);
        for end in from..=self.current.len() {
            self.current_index
                .entry(key(&self.current[end - m..end]))
                .or_default()
                .push(end);
        }
    }

    /// The draft after `context`: up to `max` host tokens.
    pub fn suggest(&mut self, context: &[u32], max: usize) -> Vec<u32> {
        self.track(context);
        let (m, n) = (self.min_match.max(1), context.len());
        if n < m || max == 0 {
            return Vec::new();
        }
        let k = key(&context[n - m..]);
        // Occurrences: sequences and end positions, with the length of the suffix they
        // match (at most `max_depth`).
        let suffix_len = |seq: &[u32], end: usize| {
            (0..self.max_depth.min(end).min(n))
                .take_while(|&i| seq[end - 1 - i] == context[n - 1 - i])
                .count()
        };
        let mut found: Vec<(&[u32], usize, usize)> = Vec::new();
        if let Some(ends) = self.current_index.get(&k) {
            for &end in ends.iter().rev().filter(|&&e| e < n).take(CANDIDATES) {
                found.push((&self.current, end, suffix_len(&self.current, end)));
            }
        }
        if let Some(hits) = self.index.get(&k) {
            for &(id, end) in hits.iter().rev().take(CANDIDATES) {
                let seq = &self.store[id - self.first];
                found.push((seq, end, suffix_len(seq, end)));
            }
        }
        let Some(best) = found.iter().map(|f| f.2).max().filter(|&l| l >= m) else {
            return Vec::new();
        };
        let mut live: Vec<(&[u32], usize)> = found
            .into_iter()
            .filter(|f| f.2 == best)
            .map(|(s, e, _)| (s, e))
            .collect();
        let limit = max.min(((self.alpha * best as f32) as usize).max(1));
        let mut draft = Vec::with_capacity(limit);
        while draft.len() < limit && !live.is_empty() {
            let j = draft.len();
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for (s, e) in &live {
                if let Some(&t) = s.get(e + j) {
                    *counts.entry(t).or_default() += 1;
                }
            }
            let total: usize = counts.values().sum();
            // The most frequent next token (ties: the smallest id, for determinism).
            let Some((&t, &c)) = counts
                .iter()
                .max_by_key(|(t, c)| (**c, std::cmp::Reverse(**t)))
            else {
                break;
            };
            if (c as f32) < self.min_share * total as f32 {
                break;
            }
            draft.push(t);
            live.retain(|(s, e)| s.get(e + j) == Some(&t));
        }
        draft
    }
}

impl Drafter for SuffixDecoding {
    fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> Result<Draft> {
        Ok(Draft::from_tokens(&self.suggest(context, max)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_the_most_frequent_continuation_of_the_longest_suffix() {
        let mut s = SuffixDecoding::default();
        // From the context itself: "1 2 3 4 … 1 2 3" drafts what followed (bounded by
        // alpha · match length = 3).
        assert_eq!(s.suggest(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3], 8), [4, 5, 6]);
        // From the store: an earlier output.
        let mut s = SuffixDecoding::default();
        s.remember(&[7, 8, 9, 10, 11, 12]);
        assert_eq!(s.suggest(&[0, 7, 8, 9], 2), [10, 11]);
        // The longest suffix decides: "5 8 9" occurs once (→ 1), "8 9" twice more (→ 2).
        let mut s = SuffixDecoding::default();
        s.remember(&[5, 8, 9, 1, 0]);
        s.remember(&[6, 8, 9, 2, 0]);
        s.remember(&[4, 8, 9, 2, 0]);
        assert_eq!(s.suggest(&[5, 8, 9], 1), [1]);
        assert_eq!(s.suggest(&[3, 8, 9], 1), [2]);
        // Below the minimum match, nothing.
        let mut s = SuffixDecoding::default();
        s.remember(&[1, 2, 3]);
        assert!(s.suggest(&[9, 1], 3).is_empty());
        assert!(s.suggest(&[], 3).is_empty());
    }

    #[test]
    fn the_store_is_bounded_and_forgets_the_oldest() {
        let mut s = SuffixDecoding {
            capacity: 8,
            ..Default::default()
        };
        s.remember(&[1, 2, 3, 4]);
        s.remember(&[5, 6, 7, 8]);
        assert_eq!(s.suggest(&[0, 1, 2], 2), [3, 4]);
        s.remember(&[9, 10, 11, 12]); // evicts the first
        assert!(s.suggest(&[0, 1, 2], 2).is_empty());
        assert_eq!(s.suggest(&[0, 9, 10], 2), [11, 12]);
    }
}
