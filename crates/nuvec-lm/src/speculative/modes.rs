//! [`DraftPolicy::Auto`](super::DraftPolicy::Auto)'s timed choice for model drafters: plain
//! decoding, chains or trees, whichever gives the most tokens per second on this machine
//! and text. Each is timed in short probes and the fastest kept; the others are probed
//! again now and then (less often the more they lose), as the text changes.

/// A way to produce the next tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Plain decoding (pipelined).
    Plain,
    /// A chain draft and its verification.
    Chain,
    /// A draft tree and its verification.
    Tree,
}

const MODES: [Mode; 3] = [Mode::Chain, Mode::Tree, Mode::Plain];

/// Actions (steps or verifications) per probe, and per window of the mode kept. Plain
/// steps vary little: a few time them.
const PROBE: usize = 16;
const PROBE_PLAIN: usize = 4;
const KEEP: usize = 128;
/// The most windows a losing mode waits before its next probe.
const MAX_WAIT: usize = 16;

#[derive(Debug, Clone)]
pub(crate) struct Modes {
    mode: Mode,
    /// Actions left in the window, and whether the window is a probe.
    left: usize,
    probe: bool,
    /// Tokens and seconds of the window.
    window: (f64, f64),
    /// Per mode: whether it applies, its last rate (tokens per second), and the windows
    /// to wait before probing it again (doubled each time it loses).
    usable: [bool; 3],
    rates: [Option<f64>; 3],
    wait: [usize; 3],
    due: [usize; 3],
}

fn index(mode: Mode) -> usize {
    MODES.iter().position(|&m| m == mode).expect("listed")
}

impl Modes {
    /// Timing among chains, plain decoding and, if `tree`, trees; chains first.
    pub(crate) fn new(tree: bool) -> Self {
        Self {
            mode: Mode::Chain,
            left: PROBE,
            probe: true,
            window: (0.0, 0.0),
            usable: [true, tree, true],
            rates: [None; 3],
            wait: [1; 3],
            due: [0; 3],
        }
    }

    /// The mode of the next action.
    pub(crate) fn mode(&self) -> Mode {
        self.mode
    }

    /// Trees do not apply after all (the drafter gives no candidates): their window ends.
    pub(crate) fn without_trees(&mut self) {
        self.usable[index(Mode::Tree)] = false;
        if self.mode == Mode::Tree {
            self.window = (0.0, 0.0);
            self.next();
        }
    }

    /// Records an action of the current mode that gave `tokens` in `seconds`.
    pub(crate) fn record(&mut self, tokens: usize, seconds: f64) {
        self.window.0 += tokens as f64;
        self.window.1 += seconds;
        self.left = self.left.saturating_sub(1);
        if self.left == 0 {
            self.rates[index(self.mode)] = Some(self.window.0 / self.window.1.max(1e-9));
            self.window = (0.0, 0.0);
            self.next();
        }
    }

    /// The best measured usable mode.
    fn best(&self) -> Mode {
        let rate = |m: &Mode| self.rates[index(*m)].unwrap_or(0.0);
        MODES
            .iter()
            .filter(|m| self.usable[index(**m)])
            .max_by(|a, b| rate(a).total_cmp(&rate(b)))
            .copied()
            .unwrap_or(Mode::Plain)
    }

    /// The next window: an unmeasured mode's probe; after a probe, the best mode kept (a
    /// probed mode that lost waits twice as long); after a kept window, the probe of the
    /// mode most due, if any.
    fn next(&mut self) {
        let usable = |m: &Mode| self.usable[index(*m)];
        if let Some(&m) = MODES
            .iter()
            .find(|m| usable(m) && self.rates[index(**m)].is_none())
        {
            self.start_probe(m);
            return;
        }
        let best = self.best();
        if self.probe {
            let i = index(self.mode);
            if self.mode != best {
                self.wait[i] = (self.wait[i] * 2).min(MAX_WAIT);
            } else {
                self.wait[i] = 1;
            }
            self.due[i] = self.wait[i];
            (self.mode, self.left, self.probe) = (best, KEEP, false);
            return;
        }
        for m in MODES {
            let i = index(m);
            if m != best && usable(&m) {
                self.due[i] = self.due[i].saturating_sub(1);
            }
        }
        let due = MODES
            .iter()
            .filter(|m| **m != best && usable(m) && self.due[index(**m)] == 0)
            .min_by_key(|m| self.wait[index(**m)]);
        match due {
            Some(&m) => self.start_probe(m),
            None => (self.mode, self.left, self.probe) = (best, KEEP, false),
        }
    }

    fn start_probe(&mut self, mode: Mode) {
        let left = match mode {
            Mode::Plain => PROBE_PLAIN,
            _ => PROBE,
        };
        (self.mode, self.left, self.probe) = (mode, left, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(m: &mut Modes, rates: [f64; 3], actions: usize) {
        for _ in 0..actions {
            let r = rates[index(m.mode())];
            m.record(4, 4.0 / r);
        }
    }

    #[test]
    fn the_fastest_mode_is_kept_and_losers_are_probed_less_often() {
        let mut m = Modes::new(true);
        // Chains 400, trees 500, plain 100 tokens per second: each probed once, trees kept.
        run(&mut m, [400.0, 500.0, 100.0], 2 * PROBE + PROBE_PLAIN);
        assert_eq!(m.mode(), Mode::Tree);
        // Over many windows the losers are probed with growing gaps: under 5% of actions.
        let mut actions = [0usize; 3];
        for _ in 0..40 * KEEP {
            actions[index(m.mode())] += 1;
            run(&mut m, [400.0, 500.0, 100.0], 1);
        }
        let losing = actions[index(Mode::Plain)] + actions[index(Mode::Chain)];
        assert!(losing * 20 < 40 * KEEP, "{actions:?}");
        // The text changes: plain decoding is now the fastest; it wins its next probe and
        // then holds most actions.
        run(&mut m, [50.0, 60.0, 100.0], 20 * KEEP);
        let mut plain = 0;
        for _ in 0..20 * KEEP {
            plain += usize::from(m.mode() == Mode::Plain);
            run(&mut m, [50.0, 60.0, 100.0], 1);
        }
        assert!(plain * 10 > 9 * 20 * KEEP, "{plain}");
    }

    #[test]
    fn without_trees_the_choice_is_chains_or_plain() {
        let mut m = Modes::new(true);
        run(&mut m, [300.0, 0.0, 100.0], PROBE);
        assert_eq!(m.mode(), Mode::Tree);
        m.without_trees();
        assert_eq!(m.mode(), Mode::Plain);
        run(&mut m, [300.0, 0.0, 100.0], PROBE_PLAIN);
        assert_eq!(m.mode(), Mode::Chain);
    }
}
