//! A server-like soak test: requests (a chunked prefill, up to 256 decode steps, a custom
//! Metal kernel, a compiled sampler, a periodic cache clear, a 1 GiB cache limit) on a
//! thread of the caller's own, for hours, sampling the process's physical footprint and the allocator's
//! statistics every minute.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example soak -- [--hours 1] [--model ID | --fixture]
//! ```
//!
//! Writes `target/soak/soak-<unix seconds>.csv` (`t_seconds, phys_footprint, active,
//! active_with_mapped, peak_with_mapped, cache, wired, requests`) and fails if the
//! footprint's floor grows by 64 MiB or more after the warm-up (10 minutes, or a sixth of a
//! shorter run: the median of the last quarter's floors against the first quarter's), or if the allocator's in-use bytes do not return to their baseline after a
//! request. `--model` defaults to `mlx-community/Qwen3-0.6B-4bit` (from the Hugging Face
//! cache); `--fixture` uses the tiny `qwen3_f32` test model instead.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use nuvec::fast::{KernelConfig, MetalKernel};
use nuvec::{Array, DType, Device, Stream, set_default_stream};
use nuvec_lm::{GenerateOptions, generate, load, load_model};

const GROWTH_LIMIT: u64 = 64 << 20;

/// The process's physical footprint (what the system charges it), in bytes.
fn phys_footprint() -> u64 {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let pid = std::process::id() as libc::pid_t;
    // SAFETY: `proc_pid_rusage` writes one `rusage_info_v4` (the flavour asked for) into
    // the buffer, which is that large and properly aligned; the result is checked before
    // the buffer is read.
    let ok = unsafe {
        libc::proc_pid_rusage(
            pid,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if ok != 0 {
        return 0;
    }
    // SAFETY: the call succeeded, so the buffer holds an initialised `rusage_info_v4`.
    unsafe { info.assume_init() }.ri_phys_footprint
}

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let hours: f64 = value("--hours").map_or(1.0, |v| v.parse().expect("a number"));
    let fixture = args.iter().any(|a| a == "--fixture");
    let model = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-0.6B-4bit".into());
    let run = Duration::from_secs_f64(hours * 3600.0);
    std::thread::spawn(move || soak(run, fixture, &model))
        .join()
        .expect("the soak thread does not panic")
}

fn soak(run: Duration, fixture: bool, model_id: &str) -> nuvec::Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let device = nuvec::gpu::Device::shared().expect("a Metal device");
    // A server bounds the buffer cache (by default it may grow to the memory limit, as
    // MLX's): requests of varying lengths leave buffers of sizes no later request reuses.
    device.set_cache_limit(1 << 30);
    let (model, prompt) = if fixture {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_f32");
        (
            load_model(&dir)?,
            (0..300u32).map(|i| i % 50 + 1).collect::<Vec<_>>(),
        )
    } else {
        let (model, tokenizer) = load(model_id)?;
        let text = "Unified memory lets the CPU and the GPU share one copy of the weights. ";
        (model, tokenizer.encode(&text.repeat(300)))
    };
    // A custom kernel and a compiled sampler, as a server's post-processing.
    let scale = MetalKernel::new(
        "soak_scale",
        &["x"],
        &["y"],
        "uint i = thread_position_in_grid.x; y[i] = x[i] * 2.0f;",
    );
    let sampler = nuvec::compile(|logits: &Array| logits.astype(DType::F32).argmax(-1));
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/soak");
    std::fs::create_dir_all(&dir).map_err(|e| nuvec::Error::Io(e.to_string()))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let path = dir.join(format!("soak-{stamp}.csv"));
    let mut csv = std::fs::File::create(&path).map_err(|e| nuvec::Error::Io(e.to_string()))?;
    let io = |e: std::io::Error| nuvec::Error::Io(e.to_string());
    writeln!(
        csv,
        "t_seconds,phys_footprint,active,active_with_mapped,peak_with_mapped,cache,wired,requests"
    )
    .map_err(io)?;

    let warmup = Duration::from_secs(600).min(run / 6);
    let start = Instant::now();
    let mut next_sample = Duration::ZERO;
    let mut baseline = None::<usize>;
    // The footprint right after each cache clear (its floor), after the warm-up.
    let mut floors: Vec<u64> = Vec::new();
    let mut requests = 0u64;
    let mut failures = Vec::new();
    while start.elapsed() < run {
        // One request: a prompt of varying length in chunks, up to 256 decode steps.
        let len = 256 + (requests as usize * 997) % (prompt.len() - 256).max(1);
        let options = GenerateOptions {
            max_tokens: 256,
            prefill_step: 512,
            ..Default::default()
        };
        let mut last = 0u32;
        for token in generate(&model, &prompt[..len], options)? {
            last = token?;
        }
        let x = Array::from_slice(&[last as f32; 1024], [1024]);
        let y = scale.call(
            &[x],
            &[(&[1024], DType::F32)],
            &KernelConfig {
                grid: [1024, 1, 1],
                threadgroup: [256, 1, 1],
                template: Vec::new(),
                ..Default::default()
            },
        );
        let picked = sampler(&y[0].reshape([1, -1]))?;
        picked.eval()?;
        drop((y, picked));
        requests += 1;
        if requests.is_multiple_of(10) {
            device.clear_cache();
            if start.elapsed() >= warmup {
                floors.push(phys_footprint());
            }
        }
        // Between requests nothing is in use but the weights.
        let active = device.memory().active;
        match baseline {
            None => baseline = Some(active),
            Some(b) if active > b + (1 << 20) => {
                failures.push(format!(
                    "request {requests}: active {active} > baseline {b}"
                ));
            }
            Some(_) => {}
        }
        if start.elapsed() >= next_sample {
            let m = device.memory();
            let foot = phys_footprint();
            let t = start.elapsed();
            writeln!(
                csv,
                "{},{foot},{},{},{},{},{},{requests}",
                t.as_secs(),
                m.active,
                m.active_with_mapped,
                m.peak_with_mapped,
                m.cache,
                m.wired
            )
            .map_err(io)?;
            next_sample += Duration::from_secs(60);
        }
    }
    // Growth: the median floor of the last quarter of the run against the first quarter's
    // (transient buffers in flight make single samples noisy).
    let median = |xs: &[u64]| {
        let mut v = xs.to_vec();
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    };
    let quarter = (floors.len() / 4).max(1);
    let growth = if floors.len() >= 4 {
        median(&floors[floors.len() - quarter..]).saturating_sub(median(&floors[..quarter]))
    } else {
        0
    };
    println!(
        "{requests} requests in {:.1} h; footprint growth after warm-up {:.1} MiB; {}",
        start.elapsed().as_secs_f64() / 3600.0,
        growth as f64 / f64::from(1 << 20),
        path.display()
    );
    if growth >= GROWTH_LIMIT {
        failures.push(format!("footprint grew {growth} bytes after the warm-up"));
    }
    if let Some(f) = failures.first() {
        println!("FAIL: {f} ({} problems)", failures.len());
        return Err(nuvec::Error::InvalidArgument(f.clone()));
    }
    println!("PASS growth={:.1} MiB", growth as f64 / f64::from(1 << 20));
    Ok(())
}
