//! Draft trees (DDTree, arXiv 2604.12989): the prefixes a block drafter finds most likely,
//! verified in one pass. The size kept per step maximises the expected tokens per
//! measured drafting and verification time (EVICT, arXiv 2605.00342).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use nuvec::Array;

/// A drafter's best tokens per draft position with their log-probabilities, best first:
/// `tokens` `[depth, k]` (uint32) and `logprobs` `[depth, k]` (float32).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Candidates {
    /// The tokens, `[depth, k]`.
    pub tokens: Array,
    /// Their log-probabilities, `[depth, k]`, descending along each row.
    pub logprobs: Array,
}

impl Candidates {
    /// Candidates from `tokens` and `logprobs` (`[depth, k]`, each row sorted descending).
    pub fn new(tokens: Array, logprobs: Array) -> Self {
        Self { tokens, logprobs }
    }
}

/// The `k` largest of each row of `x` (`[rows, n]`, float32), largest first, and their
/// indices (uint32): the best blocks of 128 by their maxima, then the best of those blocks
/// (every one of the `k` largest lies in a block among the `k` with the largest maxima), so
/// no full sort of `n`.
pub(crate) fn top_k(x: &Array, k: usize) -> (Array, Array) {
    use nuvec::ops;
    const BLOCK: i32 = 128;
    let [rows, n] = x.shape_n::<2>().unwrap_or_default();
    let k = (k as i32).clamp(1, n.max(1));
    let blocks = (n + BLOCK - 1) / BLOCK;
    let padded = ops::pad(
        x,
        &[(0, 0), (0, blocks * BLOCK - n)],
        ops::PadMode::Constant,
        f32::NEG_INFINITY,
    )
    .reshape([rows, blocks, BLOCK]);
    let kb = k.min(blocks);
    let best = ops::argpartition(&padded.max(-1), -kb, -1).i((.., -kb..));
    let index = ops::broadcast_to(&best.reshape([rows, kb, 1]), [rows, kb, BLOCK]);
    let within = ops::take_along_axis(&padded, &index, 1).reshape([rows, kb * BLOCK]);
    let top = ops::argpartition(&within, -k, -1).i((.., -k..));
    let values = ops::take_along_axis(&within, &top, -1);
    let order = ops::argsort(&values.negative(), -1);
    let (top, values) = (
        ops::take_along_axis(&top, &order, -1),
        ops::take_along_axis(&values, &order, -1),
    );
    let block = ops::take_along_axis(&best, &top.floor_divide(BLOCK), -1);
    let ids = &block * BLOCK + top.remainder(BLOCK);
    (values, ids.astype(nuvec::DType::U32))
}

/// A tree node: its parent (0 is the root, the last verified token; nodes are 1-based in
/// the order built), token, depth (1 below the root) and the log-probability of its path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Node {
    pub(crate) parent: usize,
    pub(crate) token: u32,
    pub(crate) depth: usize,
    pub(crate) score: f32,
}

/// A heap entry: the node `(parent, depth, rank)` with its path score; ties go to the
/// earlier entry, so the order is deterministic.
struct Entry {
    score: f32,
    seq: usize,
    parent: usize,
    depth: usize,
    rank: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// The best `budget` prefixes of the candidates (`[depth, k]`, row-major, each row sorted
/// descending), best first (DDTree's Algorithm 1). Every node's parent comes before it, so
/// any leading run of the result is a tree.
pub(crate) fn build(
    tokens: &[u32],
    logprobs: &[f32],
    depth: usize,
    k: usize,
    budget: usize,
) -> Vec<Node> {
    let mut nodes = Vec::with_capacity(budget);
    if depth == 0 || k == 0 || tokens.len() < depth * k || logprobs.len() < depth * k {
        return nodes;
    }
    let lp = |d: usize, r: usize| logprobs[(d - 1) * k + r];
    let mut heap = BinaryHeap::new();
    let mut seq = 0;
    let mut push = |heap: &mut BinaryHeap<Entry>, score: f32, parent, depth, rank| {
        if score.is_finite() {
            heap.push(Entry {
                score,
                seq,
                parent,
                depth,
                rank,
            });
            seq += 1;
        }
    };
    push(&mut heap, lp(1, 0), 0, 1, 0);
    while nodes.len() < budget {
        let Some(e) = heap.pop() else { break };
        nodes.push(Node {
            parent: e.parent,
            token: tokens[(e.depth - 1) * k + e.rank],
            depth: e.depth,
            score: e.score,
        });
        let id = nodes.len();
        if e.rank + 1 < k {
            let sibling = e.score - lp(e.depth, e.rank) + lp(e.depth, e.rank + 1);
            push(&mut heap, sibling, e.parent, e.depth, e.rank + 1);
        }
        if e.depth < depth {
            push(&mut heap, e.score + lp(e.depth + 1, 0), id, e.depth + 1, 0);
        }
    }
    nodes
}

/// Verification times per row count, measured, with optimistic estimates for row counts
/// not measured yet (so that the size choice tries them), and the drafting time.
#[derive(Debug, Clone, Default)]
pub(crate) struct Costs {
    seconds: Vec<Option<f64>>,
    draft: Option<f64>,
}

/// A moving average.
fn average(old: Option<f64>, new: f64) -> f64 {
    old.map_or(new, |old| 0.7 * old + 0.3 * new)
}

impl Costs {
    /// Records a verification of `rows` rows that took `seconds`.
    pub(crate) fn record(&mut self, rows: usize, seconds: f64) {
        if self.seconds.len() <= rows {
            self.seconds.resize(rows + 1, None);
        }
        self.seconds[rows] = Some(average(self.seconds[rows], seconds));
    }

    /// Records a draft that took `seconds`.
    pub(crate) fn record_draft(&mut self, seconds: f64) {
        self.draft = Some(average(self.draft, seconds));
    }

    /// The drafting time (0 before the first).
    pub(crate) fn draft(&self) -> f64 {
        self.draft.unwrap_or(0.0)
    }

    /// The expected time of `rows` rows: measured; between two measured counts,
    /// interpolated; else the nearest measured count's, scaled by the square root of the
    /// ratio of rows (optimistic both ways: costs grow with rows, at most linearly, and
    /// often in flat tiles); else `rows^½` (relative units: then only the shape matters).
    pub(crate) fn get(&self, rows: usize) -> f64 {
        let measured = |r: usize| self.seconds.get(r).copied().flatten().map(|s| (r, s));
        if let Some((_, s)) = measured(rows) {
            return s;
        }
        let below = (1..rows.min(self.seconds.len())).rev().find_map(measured);
        let above = (rows + 1..self.seconds.len()).find_map(measured);
        let scale = |(r, s): (usize, f64)| s * (rows as f64 / r as f64).sqrt();
        match (below, above) {
            (Some((r0, s0)), Some((r1, s1))) => {
                s0 + (s1 - s0) * (rows - r0) as f64 / (r1 - r0) as f64
            }
            (Some(m), None) | (None, Some(m)) => scale(m),
            (None, None) => (rows as f64).sqrt(),
        }
    }
}

/// The number of leading nodes to verify: the most expected tokens (the model's own token
/// plus `calibration` times the nodes' path probabilities) per expected time of drafting
/// and verifying the root and those nodes.
pub(crate) fn size(nodes: &[Node], calibration: f64, costs: &Costs) -> usize {
    let draft = costs.draft();
    let mut best = (0, 1.0 / (draft + costs.get(1)));
    let mut expected = 1.0;
    for (i, n) in nodes.iter().enumerate() {
        expected += calibration * f64::from(n.score).exp();
        let rate = expected / (draft + costs.get(i + 2));
        if rate > best.1 {
            best = (i + 1, rate);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every prefix of up to `depth` tokens with its score, by brute force.
    fn prefixes(lp: &[f32], depth: usize, k: usize) -> Vec<(Vec<usize>, f32)> {
        let mut all = vec![(Vec::new(), 0.0f32)];
        let mut level = all.clone();
        for d in 0..depth {
            let mut next = Vec::new();
            for (p, s) in &level {
                for r in 0..k {
                    let mut q = p.clone();
                    q.push(r);
                    next.push((q, s + lp[d * k + r]));
                }
            }
            all.extend(next.iter().cloned());
            level = next;
        }
        all.remove(0);
        all
    }

    #[test]
    fn the_tree_is_the_best_prefixes() {
        let (depth, k) = (4, 3);
        let lp: Vec<f32> = [
            [-0.1, -2.5, -3.0],
            [-0.7, -0.9, -4.0],
            [-0.05, -3.2, -3.5],
            [-1.5, -1.6, -1.7],
        ]
        .concat();
        let tokens: Vec<u32> = (0..(depth * k) as u32).collect();
        for budget in [1, 2, 5, 9, 20] {
            let nodes = build(&tokens, &lp, depth, k, budget);
            let mut want: Vec<f32> = prefixes(&lp, depth, k).iter().map(|p| p.1).collect();
            want.sort_by(|a, b| b.total_cmp(a));
            want.truncate(budget);
            let got: Vec<f32> = nodes.iter().map(|n| n.score).collect();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "{budget}: {got:?} vs {want:?}");
            }
            for (i, n) in nodes.iter().enumerate() {
                assert!(n.parent <= i, "parent before child");
                let parent_depth = match n.parent {
                    0 => 0,
                    p => nodes[p - 1].depth,
                };
                assert_eq!(n.depth, parent_depth + 1);
                let rank = n.token as usize - (n.depth - 1) * k;
                let parent_score = match n.parent {
                    0 => 0.0,
                    p => nodes[p - 1].score,
                };
                assert!((n.score - parent_score - lp[(n.depth - 1) * k + rank]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn top_k_equals_a_sort() {
        let n = 1000;
        let x: Vec<f32> = (0..2 * n)
            .map(|i| ((i * 7919) % 2003) as f32 * 0.37 - 300.0)
            .collect();
        let (values, ids) = top_k(&Array::from_slice(&x, [2, n as i32]), 9);
        let (values, ids) = (
            values.to_vec::<f32>().unwrap(),
            ids.to_vec::<u32>().unwrap(),
        );
        for r in 0..2 {
            let mut want: Vec<(f32, usize)> =
                x[r * n..(r + 1) * n].iter().copied().zip(0..).collect();
            want.sort_by(|a, b| b.0.total_cmp(&a.0));
            for j in 0..9 {
                assert_eq!(values[r * 9 + j], want[j].0, "row {r} rank {j}");
                assert_eq!(
                    x[r * n + ids[r * 9 + j] as usize],
                    want[j].0,
                    "row {r} rank {j}"
                );
            }
        }
    }

    #[test]
    fn the_size_follows_the_cost() {
        let lp: Vec<f32> = [[-0.01, -5.0], [-0.02, -5.0], [-3.0, -3.1]].concat();
        let tokens: Vec<u32> = (0..6).collect();
        let nodes = build(&tokens, &lp, 3, 2, 6);
        // Flat to 8 rows: every node pays for itself.
        let mut costs = Costs::default();
        for r in 1..=8 {
            costs.record(r, 0.025);
        }
        assert_eq!(size(&nodes, 1.0, &costs), 6);
        // Each row costs as much as a step: only the near-certain nodes pay.
        let mut costs = Costs::default();
        for r in 1..=8 {
            costs.record(r, 0.025 * r as f64);
        }
        assert_eq!(size(&nodes, 1.0, &costs), 0);
        // Drafting is paid either way: then the near-certain nodes pay.
        costs.record_draft(0.006);
        assert_eq!(size(&nodes, 1.0, &costs), 2);
        let mut costs = Costs::default();
        for r in 1..=8 {
            costs.record(r, 0.025 * (1.0 + 0.1 * (r - 1) as f64));
        }
        assert_eq!(size(&nodes, 1.0, &costs), 2);
    }

    #[test]
    fn unmeasured_costs_are_interpolated_or_optimistic() {
        let mut costs = Costs::default();
        assert_eq!(costs.get(16), 4.0);
        costs.record(8, 0.025);
        assert!((costs.get(2) - 0.0125).abs() < 1e-12);
        assert!((costs.get(32) - 0.05).abs() < 1e-12);
        costs.record(16, 0.041);
        assert!((costs.get(12) - 0.033).abs() < 1e-12);
        costs.record(16, 0.041);
        assert!((costs.get(16) - 0.041).abs() < 1e-12);
    }
}
