//! A deterministic, in-tree mutational fuzz harness: no nightly toolchain,
//! no libFuzzer, no new dependency, so it runs in `cargo test` on stable.
//!
//! A target starts from a seed corpus (valid files and every crash input found so far) and
//! applies byte-level mutations: bit flips, insertions, deletions, splices with another
//! corpus entry, and little-endian length overwrites. Every case runs under `catch_unwind`;
//! a panic prints the target, seed, case and the input in hex, then fails the test.
//!
//! The number of cases defaults to 10 000 per target and can be raised (or lowered) with
//! `NUVEC_FUZZ_CASES`. Included from `nuvec-core`'s tests with `#[path]`, so each test
//! crate uses only part of it.

#![allow(dead_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};

/// The default number of cases per target.
pub(crate) const DEFAULT_CASES: usize = 10_000;

/// The number of cases per target: `NUVEC_FUZZ_CASES`, or [`DEFAULT_CASES`].
pub(crate) fn cases() -> usize {
    std::env::var("NUVEC_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

/// SplitMix64: a small, fast, deterministic generator.
#[derive(Debug, Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `0..n` (`0` if `n` is `0`).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// A uniform value in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// `true` with probability `1 / n`.
    pub(crate) fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    /// A uniform element of `items`.
    pub(crate) fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    /// A uniform value in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Lengths a length field is overwritten with: boundaries and overflow bait.
const LENGTHS: [u64; 10] = [
    0,
    1,
    7,
    8,
    255,
    65_535,
    1 << 31,
    u32::MAX as u64,
    i64::MAX as u64,
    u64::MAX,
];

/// Applies one random mutation to `input`; `corpus` provides splice material.
pub(crate) fn mutate(rng: &mut Rng, input: &mut Vec<u8>, corpus: &[Vec<u8>]) {
    match rng.below(5) {
        // Bit flip.
        0 if !input.is_empty() => {
            let at = rng.below(input.len());
            input[at] ^= 1 << rng.below(8);
        }
        // Insert a few random or interesting bytes.
        1 => {
            let at = rng.below(input.len() + 1);
            let n = rng.range(1, 8);
            let bytes: Vec<u8> = (0..n)
                .map(|_| match rng.below(3) {
                    0 => *rng.pick(b"[]{}\",:0-9eE.\\u'()"),
                    1 => *rng.pick(&[0u8, 0x7F, 0x80, 0xFF]),
                    _ => rng.next_u64() as u8,
                })
                .collect();
            input.splice(at..at, bytes);
        }
        // Delete a range.
        2 if !input.is_empty() => {
            let at = rng.below(input.len());
            let n = rng.range(1, (input.len() - at).min(16));
            input.drain(at..at + n);
        }
        // Splice: replace the tail with another entry's tail.
        3 => {
            let other = rng.pick(corpus);
            let at = rng.below(input.len() + 1);
            let from = rng.below(other.len() + 1);
            input.truncate(at);
            input.extend_from_slice(&other[from..]);
        }
        // Overwrite 1, 2, 4 or 8 bytes with a little-endian length.
        _ => {
            let width = *rng.pick(&[1usize, 2, 4, 8]);
            if input.len() >= width {
                let at = rng.below(input.len() - width + 1);
                let value = rng.pick(&LENGTHS).wrapping_add(rng.below(3) as u64);
                input[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
            }
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Runs `target` on [`cases`] mutated inputs derived from `corpus` (each corpus entry is
/// also run unmodified first). Panics (failing the test) on the first input whose run
/// panics, after printing what reproduces it.
pub(crate) fn fuzz(name: &str, seed: u64, corpus: &[Vec<u8>], target: impl Fn(&[u8])) {
    assert!(!corpus.is_empty(), "{name}: empty corpus");
    let run = |case: &str, input: &[u8]| {
        if catch_unwind(AssertUnwindSafe(|| target(input))).is_err() {
            panic!(
                "fuzz target `{name}` panicked: seed {seed:#x}, {case}, {} bytes: {}",
                input.len(),
                hex(input)
            );
        }
    };
    for (i, entry) in corpus.iter().enumerate() {
        run(&format!("corpus entry {i}"), entry);
    }
    let mut rng = Rng::new(seed);
    for case in 0..cases() {
        let mut input = rng.pick(corpus).clone();
        for _ in 0..rng.range(1, 4) {
            mutate(&mut rng, &mut input, corpus);
        }
        run(&format!("case {case}"), &input);
    }
}
