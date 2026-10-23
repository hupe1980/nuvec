//! Attention routing: the kernel each attention call runs, observed with the GPU profiler
//! (`NUVEC_PROFILE=shapes`, in a child process so the profiler's global table and its
//! serialised dispatches stay out of the other tests).
//!
//! `routing_is_unchanged_at_one_and_many_queries` characterises the routing of one query
//! (decode) and of more than eight (prefill) with an independent copy of the 0.3.0 rules,
//! so the router is provably behaviour-preserving there. The other tests check the plan
//! (`fast::plan_attention`): it is pure, validates as the op, honours the score cap, names
//! the kernel that runs, and bounds the scratch memory the call holds.

#![cfg(target_os = "macos")]

mod common;

use std::process::Command;

use nuvec_core::ops;
use nuvec_core::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mask {
    None,
    Causal,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `blocks` is 0 for one pass.
    Vector {
        passes: u8,
        blocks: usize,
    },
    Tiled,
    Composed,
}

#[derive(Debug, Clone, Copy)]
struct Case {
    b: usize,
    hq: usize,
    hkv: usize,
    lq: usize,
    lk: usize,
    d: usize,
    dv: usize,
    dtype: DType,
    mask: Mask,
    sinks: bool,
    cap: Option<usize>,
    window: Option<i32>,
}

/// The characterisation grid: one query and more than eight, over the head sizes with and
/// without kernels, key counts on both sides of the vector and tiled thresholds.
fn grid() -> Vec<Case> {
    let mut cases = Vec::new();
    let heads = [(16, 2), (8, 8), (32, 8)];
    for &(d, dv) in &[
        (64, 64),
        (80, 80),
        (128, 128),
        (192, 192),
        (256, 256),
        (512, 512),
        (96, 64),
    ] {
        for &lq in &[1usize, 9, 33, 342, 1024] {
            for lk in [lq, lq + 777, 8193] {
                for &(hq, hkv) in &heads {
                    // Keep the composition's scores small enough for a test.
                    if hq * lq * lk * 2 > 300 << 20 {
                        continue;
                    }
                    for (i, &dtype) in [DType::F16, DType::BF16, DType::F32].iter().enumerate() {
                        let mask = [Mask::Causal, Mask::None, Mask::Bool][(lq + lk + i) % 3];
                        cases.push(Case {
                            b: 1 + (lk % 2),
                            hq,
                            hkv,
                            lq,
                            lk,
                            d,
                            dv,
                            dtype,
                            mask,
                            sinks: (d + lq) % 3 == 0,
                            cap: None,
                            window: None,
                        });
                    }
                }
            }
        }
    }
    // Decode at long keys: the GQA kernel and the larger key-block buckets.
    for lk in [1023, 1024, 4096, 8192, 8193, 32769, 65537] {
        for &(hq, hkv, d) in &[(32, 4, 128), (16, 2, 256), (24, 2, 128), (8, 1, 64)] {
            for dtype in [DType::F16, DType::BF16] {
                cases.push(Case {
                    b: 1,
                    hq,
                    hkv,
                    lq: 1,
                    lk,
                    d,
                    dv: d,
                    dtype,
                    mask: Mask::Causal,
                    sinks: false,
                    cap: None,
                    window: None,
                });
            }
        }
    }
    cases
}

/// The 0.3.0 routing rules, written out independently of the router.
fn expected(c: &Case, class: char) -> Route {
    let (lq, lk, d, dv) = (c.lq, c.lk, c.d, c.dv);
    let gqa = c.hq / c.hkv;
    let has_mask = c.mask == Mask::Bool;
    let causal = c.mask == Mask::Causal;
    if lq > 8 {
        let head =
            (d == dv && matches!(d, 64 | 72 | 80 | 96 | 128 | 192 | 256)) || (d, dv) == (96, 64);
        let causal_ok = !causal || lq <= lk;
        let scores = c.b * c.hq * lq * lk * c.dtype.size();
        let huge = scores > 4 << 30;
        let slow = match d {
            192 => !huge,
            256 => !(huge || (causal && lq >= 1024 && lk <= 8192)),
            _ => false,
        };
        return if head && causal_ok && !slow {
            Route::Tiled
        } else {
            Route::Composed
        };
    }
    let head = (d == dv && matches!(d, 64 | 96 | 128 | 256 | 512))
        || (d, dv) == (192, 128)
        || (d, dv) == (96, 64);
    let d512 = d != 512 || (lq == 1 && gqa == 8 && !has_mask && lk >= 1024);
    if !(head && lq <= lk && lq * gqa <= 32 && d512) {
        return Route::Composed;
    }
    let grouped = c.hkv < c.hq;
    let two_pass = (matches!(class, 'd' | 's') && lk >= 1024) || (grouped && lk >= 4096);
    if !two_pass {
        return Route::Vector {
            passes: 1,
            blocks: 0,
        };
    }
    let n_simds = gqa * lq;
    let blocks = match class {
        's' if lk > 1024 && n_simds > 4 => match lk {
            ..=8192 => 128,
            8193..=32768 => 256,
            32769..=65536 => 512,
            _ => 1024,
        },
        's' => 64,
        'd' if n_simds <= 2 && lk > 8192 => 256,
        'd' if n_simds >= 6 && lk >= 65536 => 1024,
        'd' if n_simds >= 6 && lk >= 16384 => 512,
        'd' => 128,
        _ if n_simds >= 4 => 64,
        _ => 32,
    };
    Route::Vector { passes: 2, blocks }
}

/// The route of one profiled call, from its kernels (`name [x, y, z]` per dispatch).
fn observed(kernels: &[String], lq: usize) -> Route {
    let extent = |k: &str| -> Vec<usize> {
        let inner = k
            .split_once('[')
            .map_or("", |(_, r)| r.trim_end_matches(']'));
        inner
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect()
    };
    if let Some(k) = kernels
        .iter()
        .find(|k| k.starts_with("nuvec_sdpa_vector_2pass_1"))
    {
        let e = extent(k);
        return Route::Vector {
            passes: 2,
            blocks: e[2] / lq,
        };
    }
    if kernels.iter().any(|k| k.starts_with("nuvec_sdpa_vector_")) {
        return Route::Vector {
            passes: 1,
            blocks: 0,
        };
    }
    if kernels
        .iter()
        .any(|k| k.starts_with("nuvec_steel_attention"))
    {
        return Route::Tiled;
    }
    Route::Composed
}

fn options<'a>(
    c: &Case,
    mask: fast::AttentionMask<'a>,
    sinks: &'a Array,
) -> fast::AttentionOptions<'a> {
    let mut o = fast::AttentionOptions::from(mask).sinks(c.sinks.then_some(sinks));
    if let Some(w) = c.window {
        o = o.window(w);
    }
    if let Some(cap) = c.cap {
        o = o.max_score_bytes(cap);
    }
    o
}

/// The attention call of `c` (unevaluated) and its evaluated inputs, which the caller
/// keeps alive so that measurements see only the call's own memory.
fn attention(c: &Case) -> (Array, Vec<Array>) {
    let z = |l: usize, h: usize, e: usize| ops::zeros([c.b, h, l, e], c.dtype);
    let (q, k, v) = (
        z(c.lq, c.hq, c.d),
        z(c.lk, c.hkv, c.d),
        z(c.lk, c.hkv, c.dv),
    );
    let bool_mask = ops::ones([c.lq, c.lk], DType::Bool);
    let sinks = ops::zeros([c.hq], c.dtype);
    let mask = match c.mask {
        Mask::None => fast::AttentionMask::None,
        Mask::Causal => fast::AttentionMask::Causal,
        Mask::Bool => fast::AttentionMask::Array(&bool_mask),
    };
    let out = fast::scaled_dot_product_attention(&q, &k, &v, 0.125, &options(c, mask, &sinks));
    eval(&[&q, &k, &v, &bool_mask, &sinks]).unwrap();
    (out, vec![q, k, v, bool_mask, sinks])
}

/// Runs this test binary's `child` test (with the profiler on when `profile`), returning
/// its stdout.
fn run_child(child: &str, profile: bool) -> String {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        child,
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ])
    .env("NUVEC_PLAN_CHILD", "1")
    .env_remove("NUVEC_PROFILE");
    if profile {
        cmd.env("NUVEC_PROFILE", "shapes");
    }
    let out = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "child {child} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// Profiled kernels per grid case, as `@@case <i>` followed by `@@kernel <name [extent]>`
/// lines.
#[test]
#[ignore = "run by routing_is_unchanged_at_one_and_many_queries in a profiled child"]
fn child_profile_grid() {
    if std::env::var_os("NUVEC_PLAN_CHILD").is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (i, c) in grid().iter().enumerate() {
        let (out, _inputs) = attention(c);
        nuvec_gpu::profile::reset();
        out.eval().unwrap();
        println!("\n@@case {i}");
        for (kernel, _, _) in nuvec_gpu::profile::report() {
            println!("@@kernel {kernel}");
        }
    }
}

/// The kernels of each grid case, parsed from a profiled child's output.
fn profiled_grid() -> Vec<Vec<String>> {
    let stdout = run_child("child_profile_grid", true);
    let mut cases: Vec<Vec<String>> = Vec::new();
    for line in stdout.lines() {
        if line.starts_with("@@case ") {
            cases.push(Vec::new());
        } else if let Some(k) = line.strip_prefix("@@kernel ") {
            cases.last_mut().expect("a case first").push(k.to_string());
        }
    }
    cases
}

fn class() -> char {
    nuvec_gpu::Device::shared()
        .and_then(|d| d.info().class)
        .unwrap_or('g')
}

#[test]
fn routing_is_unchanged_at_one_and_many_queries() {
    if common::require_gpu().is_none() {
        return;
    }
    let cases = grid();
    let profiled = profiled_grid();
    assert_eq!(profiled.len(), cases.len(), "one profile per case");
    let class = class();
    for (c, kernels) in cases.iter().zip(&profiled) {
        assert_eq!(
            observed(kernels, c.lq),
            expected(c, class),
            "{c:?}: kernels {kernels:?}"
        );
    }
}

// ── The plan ───────────────────────────────────────────────────────────────────────────

/// Bytes per element of the largest score-sized intermediate.
fn score_size(c: &Case) -> usize {
    if c.sinks || c.dtype == DType::F32 {
        4
    } else {
        c.dtype.size()
    }
}

/// The plan grid: vector, tiled and composed shapes, uncapped and with caps that force 2,
/// 3 and ⌈Lq/9⌉ blocks and a cap below one row, a sliding window, on head sizes with and
/// without kernels.
fn plan_grid() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut i = 0usize;
    for &(b, hq, hkv) in &[(1, 16, 2), (2, 8, 8), (1, 32, 8)] {
        for d in [64, 80, 128, 256, 512] {
            for lq in [1usize, 4, 9, 33, 342, 1024] {
                for lk in [lq, lq + 777, 8193] {
                    if b * hq * lq * lk * 4 > 96 << 20 {
                        continue;
                    }
                    i += 1;
                    let dtype = [DType::F16, DType::BF16, DType::F32][i % 3];
                    let mask = [Mask::Causal, Mask::None, Mask::Bool][(i / 3) % 3];
                    let sinks = i.is_multiple_of(5);
                    let base = Case {
                        b,
                        hq,
                        hkv,
                        lq,
                        lk,
                        d,
                        dv: d,
                        dtype,
                        mask,
                        sinks,
                        cap: None,
                        window: None,
                    };
                    cases.push(base);
                    if lq <= 8 {
                        continue;
                    }
                    let row = b * hq * lk * score_size(&base);
                    for blocks in [2, 3, lq.div_ceil(9)] {
                        cases.push(Case {
                            cap: Some(row * lq.div_ceil(blocks)),
                            ..base
                        });
                    }
                    cases.push(Case {
                        cap: Some(row - 1),
                        ..base
                    });
                    if mask == Mask::Causal {
                        cases.push(Case {
                            window: Some(100),
                            ..base
                        });
                        cases.push(Case {
                            window: Some(100),
                            cap: Some(b * hq * 120 * score_size(&base) * 3),
                            ..base
                        });
                    }
                }
            }
        }
    }
    cases
}

fn plan_of(c: &Case) -> Result<fast::AttentionPlan> {
    let bool_mask = ops::ones([c.lq, c.lk], DType::Bool);
    let sinks = ops::zeros([c.hq], c.dtype);
    let mask = match c.mask {
        Mask::None => fast::AttentionMask::None,
        Mask::Causal => fast::AttentionMask::Causal,
        Mask::Bool => fast::AttentionMask::Array(&bool_mask),
    };
    let s = |l: usize, h: usize, e: usize| [c.b as i32, h as i32, l as i32, e as i32];
    fast::plan_attention(
        &s(c.lq, c.hq, c.d),
        &s(c.lk, c.hkv, c.d),
        &s(c.lk, c.hkv, c.dv),
        c.dtype,
        &options(c, mask, &sinks),
    )
}

/// Whether the plan of `c` can run (a cap below one row without a tiled kernel cannot).
fn runnable(c: &Case) -> bool {
    plan_of(c).is_ok()
}

/// Profiled kernels with dispatch counts per plan-grid case.
#[test]
#[ignore = "run by plans_name_the_kernels_that_run in a profiled child"]
fn child_profile_plans() {
    if std::env::var_os("NUVEC_PLAN_CHILD").is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (i, c) in plan_grid().iter().enumerate() {
        println!("\n@@case {i}");
        if !runnable(c) {
            continue;
        }
        let (out, _inputs) = attention(c);
        nuvec_gpu::profile::reset();
        out.eval().unwrap();
        for (kernel, count, _) in nuvec_gpu::profile::report() {
            println!("@@kernel {count} {kernel}");
        }
    }
}

/// Peak bytes above inputs and output per plan-grid case (no profiler, one at a time).
#[test]
#[ignore = "run by scratch_is_within_the_plan in a child"]
fn child_measure_plans() {
    if std::env::var_os("NUVEC_PLAN_CHILD").is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let dev = nuvec_gpu::Device::shared().expect("Metal");
    for (i, c) in plan_grid().iter().enumerate() {
        if !runnable(c) {
            continue;
        }
        let (out, _inputs) = attention(c);
        dev.reset_peak();
        out.eval().unwrap();
        let m = dev.memory();
        println!("\n@@mem {i} {}", m.peak - m.active);
        drop(out);
    }
}

/// Plans of the plan grid, printed by a child process.
#[test]
#[ignore = "run by plans_are_pure in a child"]
fn child_print_plans() {
    if std::env::var_os("NUVEC_PLAN_CHILD").is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (i, c) in plan_grid().iter().enumerate() {
        println!("\n@@plan {i} {:?}", plan_of(c).map_err(|e| e.to_string()));
    }
}

#[test]
fn plans_are_pure() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let cases = plan_grid();
    let plans: Vec<String> = cases
        .iter()
        .map(|c| format!("{:?}", plan_of(c).map_err(|e| e.to_string())))
        .collect();
    let again = |label: &str| {
        for (c, p) in cases.iter().zip(&plans) {
            let q = format!("{:?}", plan_of(c).map_err(|e| e.to_string()));
            assert_eq!(&q, p, "{label}: {c:?}");
        }
    };
    for _ in 0..100 {
        let c = &cases[cases.len() / 2];
        assert_eq!(
            format!("{:?}", plan_of(c).map_err(|e| e.to_string())),
            plans[cases.len() / 2]
        );
    }
    again("again");
    // Another thread.
    let theirs = std::thread::spawn(|| {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        plan_grid()
            .iter()
            .map(|c| format!("{:?}", plan_of(c).map_err(|e| e.to_string())))
            .collect::<Vec<_>>()
    })
    .join()
    .unwrap();
    assert_eq!(theirs, plans, "another thread");
    // Under memory pressure.
    let big = ops::zeros([256 << 20], DType::F32);
    big.eval().unwrap();
    again("after allocating 1 GiB");
    drop(big);
    // Another process.
    let stdout = run_child("child_print_plans", false);
    let child: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("@@plan "))
        .map(|l| l.split_once(' ').expect("index and plan").1)
        .collect();
    assert_eq!(child.len(), plans.len());
    for (i, (a, b)) in child.iter().zip(&plans).enumerate() {
        assert_eq!(a, b, "another process: {:?}", cases[i]);
    }
}

/// Lines `@@<tag> ...` per case index of a child's output.
fn per_case(stdout: &str, tag: &str) -> Vec<Vec<String>> {
    let mut cases: Vec<Vec<String>> = Vec::new();
    for line in stdout.lines() {
        if line.starts_with("@@case ") {
            cases.push(Vec::new());
        } else if let Some(k) = line.strip_prefix(tag) {
            cases.last_mut().expect("a case first").push(k.to_string());
        }
    }
    cases
}

#[test]
fn plans_name_the_kernels_that_run() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let cases = plan_grid();
    let profiled = per_case(&run_child("child_profile_plans", true), "@@kernel ");
    assert_eq!(profiled.len(), cases.len());
    let mut kinds = std::collections::BTreeSet::new();
    for (c, kernels) in cases.iter().zip(&profiled) {
        let Ok(plan) = plan_of(c) else { continue };
        let count = |pred: &dyn Fn(&str) -> bool| -> u64 {
            kernels
                .iter()
                .map(|k| k.split_once(' ').unwrap())
                .filter(|(_, name)| pred(name))
                .map(|(n, _)| n.parse::<u64>().unwrap())
                .sum()
        };
        let first_pass = count(&|n| n.starts_with("nuvec_sdpa_vector_") && !n.contains("2pass_2"));
        let two_pass = count(&|n| n.starts_with("nuvec_sdpa_vector_2pass_1"));
        let tiled = count(&|n| n.starts_with("nuvec_steel_attention"));
        let softmax = count(&|n| n.contains("softmax"));
        let what = format!("{c:?} → {plan:?}: {kernels:?}");
        match plan.kernel {
            fast::AttentionKernel::Vector {
                passes, dispatches, ..
            } => {
                kinds.insert("vector");
                assert_eq!(first_pass, dispatches as u64, "{what}");
                assert_eq!(two_pass > 0, passes == 2, "{what}");
                assert_eq!(tiled, 0, "{what}");
            }
            fast::AttentionKernel::Tiled => {
                kinds.insert(if plan.capped {
                    "tiled (capped)"
                } else {
                    "tiled"
                });
                assert_eq!(tiled, 1, "{what}");
                assert_eq!(first_pass, 0, "{what}");
            }
            fast::AttentionKernel::Composed => {
                kinds.insert("composed");
                assert_eq!(tiled + first_pass, 0, "{what}");
                if !c.sinks {
                    assert_eq!(softmax, 1, "{what}");
                }
            }
            fast::AttentionKernel::ComposedBlocks { rows } => {
                kinds.insert("blocks");
                assert_eq!(tiled + first_pass, 0, "{what}");
                if !c.sinks {
                    assert_eq!(softmax, c.lq.div_ceil(rows as usize) as u64, "{what}");
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    for kind in ["vector", "tiled", "composed", "blocks"] {
        assert!(
            kinds.contains(kind),
            "the grid never planned {kind}: {kinds:?}"
        );
    }
}

#[test]
fn scratch_is_within_the_plan() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let cases = plan_grid();
    let stdout = run_child("child_measure_plans", false);
    let mut checked = 0;
    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("@@mem ") else {
            continue;
        };
        let (i, bytes) = rest.split_once(' ').unwrap();
        let (c, measured) = (
            &cases[i.parse::<usize>().unwrap()],
            bytes.parse::<usize>().unwrap(),
        );
        let plan = plan_of(c).unwrap();
        let scratch = plan.scratch_bytes;
        // Allocator rounding: up to a page per buffer (a few per block), and the size
        // classes.
        let blocks = match plan.kernel {
            fast::AttentionKernel::ComposedBlocks { rows } => c.lq.div_ceil(rows as usize),
            _ => 1,
        };
        let slack = (256 << 10) + (64 << 10) * blocks + scratch / 32;
        assert!(
            measured <= scratch + slack,
            "{c:?} {plan:?}: measured {measured} above the plan's {scratch}"
        );
        // The scaled queries are an upper-bound term (the scores' peak may come after
        // they are released), so the lower bound holds where the scores dominate.
        let rows = match plan.kernel {
            fast::AttentionKernel::ComposedBlocks { rows } => rows as usize,
            _ => c.lq,
        };
        let scaled_q = c.b * c.hq * rows * c.d * c.dtype.size();
        let blocked = matches!(plan.kernel, fast::AttentionKernel::ComposedBlocks { .. });
        if scratch >= 8 << 20 && scaled_q * 4 <= scratch && !blocked {
            checked += 1;
            assert!(
                measured * 10 >= scratch * 8,
                "{c:?} {plan:?}: measured {measured} below 80% of the plan's {scratch}"
            );
        }
        if let Some(cap) = c.cap {
            assert!(plan.max_intermediate_bytes <= cap, "{c:?} {plan:?}");
        }
    }
    assert!(checked > 10, "too few large cases ({checked})");
}

#[test]
fn invalid_options_are_rejected_by_plan_and_op() {
    let q = ops::zeros([1, 4, 16, 64], DType::F16);
    let k = ops::zeros([1, 2, 32, 64], DType::F16);
    let (qs, ks) = ([1, 4, 16, 64], [1, 2, 32, 64]);
    let invalid = |o: fast::AttentionOptions<'_>| {
        let plan = fast::plan_attention(&qs, &ks, &ks, DType::F16, &o);
        assert!(
            matches!(plan, Err(Error::InvalidArgument(_))),
            "{o:?}: {plan:?}"
        );
        let op = fast::scaled_dot_product_attention(&q, &k, &k, 1.0, &o).check();
        assert!(op.is_err(), "{o:?}");
    };
    invalid(fast::AttentionOptions::causal().max_score_bytes(0));
    invalid(fast::AttentionOptions::default().window(8));
    invalid(fast::AttentionOptions::causal().window(0));
    let bad_mask = ops::ones([3, 5], DType::Bool);
    invalid(fast::AttentionOptions::default().mask(&bad_mask));
    let bad_sinks = ops::zeros([3], DType::F16);
    invalid(fast::AttentionOptions::causal().sinks(&bad_sinks));
    assert!(fast::plan_attention(&[1, 4, 16], &ks, &ks, DType::F16, &Default::default()).is_err());
    assert!(fast::plan_attention(&qs, &ks, &ks, DType::I32, &Default::default()).is_err());
}

#[test]
fn a_cap_below_one_row_runs_tiled_or_fails_naming_the_minimum() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    // One row of scores: 8 heads × 777 keys × 2 bytes. Head size 256 composes uncapped at
    // these shapes; under the cap the tiled kernel runs.
    let row = 8 * 777 * 2;
    let tiled = fast::plan_attention(
        &[1, 8, 33, 256],
        &[1, 8, 777, 256],
        &[1, 8, 777, 256],
        DType::F16,
        &fast::AttentionOptions::causal().max_score_bytes(row - 1),
    )
    .unwrap();
    assert_eq!(tiled.kernel, fast::AttentionKernel::Tiled);
    assert!(tiled.capped);
    // No tiled kernel for head size 512: an error naming the smallest admissible cap.
    let err = fast::plan_attention(
        &[1, 8, 33, 512],
        &[1, 8, 777, 512],
        &[1, 8, 777, 512],
        DType::F16,
        &fast::AttentionOptions::causal().max_score_bytes(row - 1),
    )
    .unwrap_err();
    assert!(err.to_string().contains(&format!("{row}")), "{err}");
    // The CPU has no fused kernel: the cap blocks the composition, or fails below a row.
    with_stream(Device::cpu(), || {
        let p = fast::plan_attention(
            &[1, 8, 33, 128],
            &[1, 8, 777, 128],
            &[1, 8, 777, 128],
            DType::F16,
            &fast::AttentionOptions::causal().max_score_bytes(row * 4),
        )
        .unwrap();
        assert_eq!(p.kernel, fast::AttentionKernel::ComposedBlocks { rows: 4 });
        assert!(
            fast::plan_attention(
                &[1, 8, 33, 128],
                &[1, 8, 777, 128],
                &[1, 8, 777, 128],
                DType::F16,
                &fast::AttentionOptions::causal().max_score_bytes(row - 1),
            )
            .is_err()
        );
    });
}

/// A real long-prefill schedule (Qwen3.6: 16 query heads, 2 key heads, head size 256,
/// bf16) under a 512 MiB score budget: the plan holds the budget and its scratch, and
/// the kernel it names runs. Benchmark tier (large; run in the release checklist).
#[test]
#[ignore = "benchmark tier: a long-prefill chunk schedule up to 262,144 keys"]
fn long_prefill_schedule_plans_hold_the_budget() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let dev = nuvec_gpu::Device::shared().expect("Metal");
    let cap = 512usize << 20;
    for &(lq, lk) in &[
        (2048, 8193),
        (1024, 33_792),
        (1024, 131_072),
        (1024, 262_144),
    ] {
        let opts = fast::AttentionOptions::causal().max_score_bytes(cap);
        let plan = fast::plan_attention(
            &[1, 16, lq, 256],
            &[1, 2, lk, 256],
            &[1, 2, lk, 256],
            DType::BF16,
            &opts,
        )
        .unwrap();
        assert!(plan.max_intermediate_bytes <= cap, "{lq}×{lk}: {plan:?}");
        let q = ops::zeros([1, 16, lq, 256], DType::BF16);
        let k = ops::zeros([1, 2, lk, 256], DType::BF16);
        eval(&[&q, &k]).unwrap();
        let out = fast::scaled_dot_product_attention(&q, &k, &k, 0.0625, &opts);
        dev.reset_peak();
        out.eval().unwrap();
        let m = dev.memory();
        let measured = m.peak - m.active;
        eprintln!("{lq}×{lk}: {plan:?}, measured scratch {measured}");
        assert!(measured <= plan.scratch_bytes + (1 << 20) + plan.scratch_bytes / 32);
    }
}
