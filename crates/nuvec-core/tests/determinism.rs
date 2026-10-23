//! Run-to-run determinism on the GPU: the same inputs give bit-identical results across
//! runs, whatever memory the allocator hands out and whatever is still in flight.
//!
//! The workload's shapes are chosen to reach the kernel variants that combine partial
//! results or run in several passes: split-K GEMM, quantized products, the two-pass decode
//! attention, multi-pass reductions, multi-block sorts and scans, the sorted MoE gather and
//! Gated DeltaNet. Each run is submitted before the previous one is read, and runs are
//! interleaved with allocations of other sizes, so buffers are reused while GPU work that
//! used them is still pending.
//!
//! Not covered, because they are nondeterministic by design (as in MLX): scatter-add with
//! repeated indices (float atomics) and `put_along_axis` with repeated indices (the last
//! write wins in no fixed order).

#![cfg(target_os = "macos")]

mod common;

use nuvec_core::ops;
use nuvec_core::*;

/// Makes the GPU this thread's default stream; false (and skip) if there is none.
fn gpu() -> bool {
    let ok = common::require_gpu().is_some();
    set_default_stream(Stream::default_for(Device::gpu(0)));
    ok
}

/// Fixed inputs in about [-1, 1], one pattern per seed.
fn normal(shape: &[usize], dtype: DType, seed: u64) -> Array {
    let n: usize = shape.iter().product();
    let t = ops::arange(0.0, n as f64, 1.0);
    let x = (&t * (0.7381 + seed as f64 * 0.0137)).sin();
    let x = &x * (&t * (0.0071 * seed as f64 + 0.013)).cos();
    x.reshape(shape).astype(dtype)
}

/// Every operation of the workload, from inputs fixed by their seeds.
fn workload() -> Result<Vec<Array>> {
    let (f32_, bf16_) = (DType::F32, DType::BF16);
    let mut out = Vec::new();
    // GEMM: split-K (few output tiles, long K; also the bf16 shape), tiled (short K), and a
    // matrix-vector product.
    out.push(normal(&[32, 16384], f32_, 1).matmul(&normal(&[16384, 32], f32_, 2)));
    out.push(normal(&[512, 1024], bf16_, 3).matmul(&normal(&[1024, 768], bf16_, 4)));
    out.push(normal(&[2048, 256], bf16_, 34).matmul(&normal(&[256, 2048], bf16_, 35)));
    let w = normal(&[4096, 4096], bf16_, 5);
    out.push(normal(&[1, 4096], bf16_, 6).matmul(&w.t()));
    // Quantized: matrix-vector, matrix-matrix, and the transposed-weight product.
    let (wq, scales, biases) = w.quantize(64, 4);
    for (rows, seed) in [(1, 7), (8, 8), (256, 9)] {
        let x = normal(&[rows, 4096], bf16_, seed);
        out.push(x.quantized_matmul(&wq, &scales, &biases, true, 64, 4));
    }
    out.push(normal(&[1, 4096], bf16_, 10).quantized_matmul(&wq, &scales, &biases, false, 64, 4));
    // Mixture of experts: rows sorted by expert through 16 quantized experts.
    let experts = normal(&[16, 512, 1024], bf16_, 11);
    let (eq, es, eb) = experts.quantize(64, 4);
    let ids: Vec<u32> = (0..512u32).map(|i| i * 16 / 512).collect();
    let ids = Array::from(ids);
    out.push(normal(&[512, 1, 1024], bf16_, 12).gather_qmm(
        &eq,
        &es,
        &eb,
        None,
        Some(&ids),
        true,
        64,
        4,
        true,
    ));
    // Attention: decode over many keys (two passes), causal prefill (tiled).
    let (k, v) = (
        normal(&[1, 4, 8192, 128], bf16_, 13),
        normal(&[1, 4, 8192, 128], bf16_, 14),
    );
    out.push(fast::scaled_dot_product_attention(
        &normal(&[1, 16, 1, 128], bf16_, 15),
        &k,
        &v,
        0.088,
        &fast::AttentionOptions::default(),
    ));
    let (k, v) = (
        normal(&[1, 4, 300, 128], bf16_, 16),
        normal(&[1, 4, 300, 128], bf16_, 17),
    );
    out.push(fast::scaled_dot_product_attention(
        &normal(&[1, 16, 300, 128], bf16_, 18),
        &k,
        &v,
        0.088,
        &fast::AttentionOptions::causal(),
    ));
    // Reductions: a full sum in several passes, row reductions, softmax.
    out.push(normal(&[1 << 22], f32_, 19).sum(..));
    let rows = normal(&[64, 32768], bf16_, 20);
    out.push(rows.logsumexp(-1));
    out.push(rows.softmax(-1));
    out.push(rows.max(0));
    out.push(rows.sum(0));
    // Multi-block sort and scan.
    let long = normal(&[4, 50000], f32_, 21);
    out.push(long.sort(-1));
    out.push(long.argsort(-1));
    out.push(normal(&[4, 100000], f32_, 22).cumsum(-1));
    // Fused norms and rotations.
    let (h, n) = fast::add_rms_norm(
        &normal(&[64, 4096], bf16_, 23),
        &normal(&[64, 4096], bf16_, 24),
        Some(&normal(&[4096], bf16_, 25)),
        1e-6,
    );
    out.extend([h, n]);
    let rotation = fast::Rotation {
        dims: 128,
        traditional: false,
        base: Some(1e6),
        scale: 1.0,
        freqs: None,
    };
    out.push(fast::norm_rope(
        &normal(&[1, 64, 8, 128], bf16_, 26).transpose([0, 2, 1, 3]),
        Some((&normal(&[128], bf16_, 27), 1e-6)),
        rotation,
        40,
        Some(&normal(&[1, 8, 256, 128], bf16_, 28)),
    ));
    // Gated DeltaNet.
    let (y, state) = fast::gated_delta(
        &(normal(&[1, 64, 4, 128], bf16_, 29) * 0.05),
        &(normal(&[1, 64, 4, 128], bf16_, 30) * 0.05),
        &normal(&[1, 64, 8, 128], bf16_, 31),
        &normal(&[1, 64, 8], f32_, 32).sigmoid(),
        &normal(&[1, 64, 8], bf16_, 33).sigmoid(),
        &ops::zeros([1, 8, 128, 128], f32_),
    );
    out.extend([y, state]);
    Ok(out)
}

/// The bits of every element (floats widened exactly to float32, integers to int64).
fn bits(a: &Array) -> Vec<u64> {
    match a.dtype().category() {
        Category::Float => a
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        _ => a
            .astype(DType::I64)
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as u64)
            .collect(),
    }
}

#[test]
fn results_are_bit_identical_across_runs() {
    if !gpu() {
        return;
    }
    let first: Vec<Vec<u64>> = workload().unwrap().iter().map(bits).collect();
    let mut pending: Option<Vec<Array>> = None;
    for run in 0..5 {
        // Other allocations, some kept alive, so later runs get different buffers.
        let noise: Vec<Array> = (0..(run + 1) * 3)
            .map(|i| normal(&[1000 + 997 * i, 64], DType::F32, 100 + i as u64) * 2.0)
            .collect();
        async_eval(&noise).unwrap();
        // Submit this run before reading the previous one: its buffers come from memory
        // that pending work still uses.
        let outputs = workload().unwrap();
        async_eval(&outputs).unwrap();
        drop(noise);
        if let Some(previous) = pending.replace(outputs) {
            check(&first, &previous, run);
        }
    }
    check(&first, &pending.unwrap(), 5);
}

fn check(first: &[Vec<u64>], outputs: &[Array], run: usize) {
    assert_eq!(first.len(), outputs.len());
    for (i, (want, got)) in first.iter().zip(outputs).enumerate() {
        let got = bits(got);
        let differ = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ, 0,
            "run {run}: output {i} differs in {differ} elements"
        );
    }
}

/// Uniform noise from this thread's own key sequence (no explicit key).
fn draw() -> Vec<f32> {
    set_default_stream(Stream::default_for(Device::cpu()));
    random::uniform(0.0, 1.0, [8], DType::F32, None)
        .to_vec::<f32>()
        .unwrap()
}

/// Threads draw distinct streams; after `seed` on this thread, threads that start drawing
/// in a fixed order draw the same streams in every run.
#[test]
fn threads_draw_distinct_reproducible_streams() {
    // Unseeded: two concurrent threads differ.
    let (a, b) = (std::thread::spawn(draw), std::thread::spawn(draw));
    assert_ne!(a.join().unwrap(), b.join().unwrap(), "unseeded threads");
    // Seeded: each worker's stream is the same in both runs, and the workers differ.
    let run = || {
        random::seed(7);
        let first = std::thread::spawn(draw).join().unwrap();
        (first, std::thread::spawn(draw).join().unwrap())
    };
    let (first, second) = run();
    assert_ne!(first, second, "workers of one run");
    assert_eq!(run(), (first, second), "the same in every run");
}

/// FNV-1a over every output's shape and bits.
fn hash(outputs: &[Array]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for out in outputs {
        for &d in out.shape() {
            eat(d as u64);
        }
        for v in bits(out) {
            eat(v);
        }
    }
    h
}

/// The workload's hash, printed as `hash=<hex>` for the parent test.
#[test]
#[ignore = "run by results_are_bit_identical_across_processes in a child process"]
fn child_prints_the_workload_hash() {
    if std::env::var_os("NUVEC_DETERMINISM_CHILD").is_none() || !gpu() {
        return;
    }
    println!("\nhash={:016x}", hash(&workload().unwrap()));
}

/// Two fresh processes give the workload's outputs bit for bit (and equal this one's).
/// The hash goes to `target/determinism/<precompiled|runtime>.hash`, so `just
/// determinism-xbuild` can compare the precompiled kernels with the runtime-compiled ones
/// (`NUVEC_METALLIB=off`).
#[test]
fn results_are_bit_identical_across_processes() {
    if !gpu() {
        return;
    }
    let child = || {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child_prints_the_workload_hash",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("NUVEC_DETERMINISM_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("hash="))
            .expect("a hash line")
            .to_string()
    };
    let (a, b) = (child(), child());
    assert_eq!(a, b, "two processes differ");
    let own = format!("{:016x}", hash(&workload().unwrap()));
    assert_eq!(a, own, "a child differs from this process");
    let config = match std::env::var("NUVEC_METALLIB").as_deref() {
        Ok("off") => "runtime",
        _ => "precompiled",
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/determinism");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{config}.hash")), &own).unwrap();
}

/// Every kernel source using float atomics (whose results depend on the order threads
/// commit) is listed among the non-deterministic operations in the `nuvec` crate's
/// "Determinism" documentation.
#[test]
fn float_atomics_are_documented() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let docs = std::fs::read_to_string(root.join("crates/nuvec/src/lib.rs")).unwrap();
    let section = docs
        .split("# Determinism")
        .nth(1)
        .expect("a Determinism section in the nuvec crate docs");
    let listed = section
        .split("Not deterministic")
        .nth(1)
        .expect("a list of non-deterministic operations");
    let dir = root.join("crates/nuvec-kernels/src/metal");
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !(name.ends_with(".metal") || name.ends_with(".h")) {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        // Float atomics: `atomic_float`, or a compare-and-swap loop over 16-bit floats.
        let float_atomics = source.contains("atomic_float") || source.contains("atomic_add_16");
        if float_atomics {
            found.push(name.clone());
            assert!(
                listed.contains(&format!("`{name}`")),
                "{name} uses float atomics but is not in the Determinism section's list"
            );
        }
    }
    assert!(found.contains(&"indexing.metal".to_string()), "{found:?}");
}

/// The operations documented as deterministic despite scattering: unique indices give the
/// same bits in every run (no test needs a racing operation to differ).
#[test]
fn unique_index_scatters_are_deterministic() {
    if !gpu() {
        return;
    }
    let base = normal(&[64, 4096], DType::F32, 40);
    let values = normal(&[64, 16], DType::F32, 41);
    // Unique positions per row (as a top-k sampler's `argpartition` gives).
    let idx: Vec<i32> = (0..64 * 16)
        .map(|i| ((i % 16) * 251 + i / 16) % 4096)
        .collect();
    let idx = Array::from_slice(&idx, [64, 16]);
    let run = || {
        let put = base.put_along_axis(&idx, &values, -1);
        let fill = base.put_along_axis(&idx, -1e9f32, -1);
        let add = base.scatter_add_along_axis(&idx, &values, -1);
        vec![bits(&put), bits(&fill), bits(&add)]
    };
    let first = run();
    for i in 0..20 {
        assert_eq!(run(), first, "run {i}");
    }
}
