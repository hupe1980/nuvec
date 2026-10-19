//! Per-kernel GPU time, for finding where time goes (`NUVEC_PROFILE=1`).
//!
//! While enabled, array evaluation commits every dispatch in a command buffer of its own
//! and waits for it, and each completed command buffer's GPU time (Metal's GPU start and
//! end timestamps) is added to its kernel's total. Commands then never overlap (nor wait
//! for one another on the GPU), so totals show each kernel's cost alone, not the
//! overlapped wall time. `NUVEC_PROFILE=shapes` keeps each
//! grid size (threads per dimension) apart. The GPU idles between command buffers, so its
//! clocks drop: absolute times are inflated (most for short kernels); compare shares and
//! orders of magnitude, and measure throughput with the benchmarks.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("NUVEC_PROFILE").is_some_and(|v| v != "0"));

/// Per kernel: dispatches and GPU seconds.
type Table = HashMap<Arc<str>, (u64, f64)>;

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

/// Whether profiling is on (`NUVEC_PROFILE` set to anything but `0`, read once).
pub fn enabled() -> bool {
    *ENABLED
}

/// Whether grid sizes are kept apart (`NUVEC_PROFILE=shapes`).
pub(crate) fn shapes() -> bool {
    static SHAPES: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("NUVEC_PROFILE").is_some_and(|v| v == "shapes"));
    *SHAPES
}

/// Adds `seconds` of GPU time, shared equally by `kernels` (one per dispatch).
pub(crate) fn record(kernels: &[Arc<str>], seconds: f64) {
    if kernels.is_empty() {
        return;
    }
    let share = seconds / kernels.len() as f64;
    let mut table = TABLE.lock().unwrap_or_else(PoisonError::into_inner);
    let table = table.get_or_insert_with(HashMap::new);
    for k in kernels {
        let entry = table.entry(k.clone()).or_default();
        entry.0 += 1;
        entry.1 += share;
    }
}

/// `(kernel, dispatches, GPU seconds)` recorded so far, most expensive first.
pub fn report() -> Vec<(String, u64, f64)> {
    let table = TABLE.lock().unwrap_or_else(PoisonError::into_inner);
    let mut rows: Vec<_> = table
        .iter()
        .flatten()
        .map(|(k, &(n, s))| (k.to_string(), n, s))
        .collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    rows
}

/// Forgets everything recorded so far.
pub fn reset() {
    *TABLE.lock().unwrap_or_else(PoisonError::into_inner) = None;
}
