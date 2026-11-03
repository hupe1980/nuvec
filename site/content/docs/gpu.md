+++
title = "GPU, memory and profiling"
description = "How nuvec runs on the Apple GPU: asynchronous Metal command buffers, unified memory, MLX's allocator, memory limits, wired memory with residency sets, and a per-kernel profiler."
weight = 50
[extra]
group = "GPU"
+++

## How work reaches the GPU

Evaluating a graph encodes one Metal compute dispatch per node into command buffers and commits them without waiting:

- **Growing command buffers.** An evaluation that starts with the GPU idle commits after 8 dispatches, so the GPU starts quickly, then doubles the batch while the GPU is busy (up to 256). A command buffer that makes new buffers commits after 50 dispatches, as MLX's 50 ops, so later work reuses its intermediates.
- **Minimal barriers.** Dispatches that don't depend on each other run concurrently within a command buffer; a memory barrier is inserted only when a dispatch reads what an earlier one wrote.
- **Ordering by fences.** Command buffers are ordered through per-buffer fences, so independent work from different evaluations can overlap on the GPU.
- **Errors.** A GPU failure is reported by the first read of an affected array, and every later GPU result on the device reports it too, as in MLX.

Kernels are compiled ahead of time into `nuvec.metallib` when Apple's Metal toolchain is installed. Otherwise their sources are embedded and compiled, with the same options, when the GPU is first used: about 20 seconds once, then cached by macOS (`NUVEC_METALLIB=off` selects this mode; `NUVEC_REQUIRE_METALLIB=1` makes it a build error).

## Unified memory, safely

Buffers live in shared memory: the CPU reads GPU results in place. Rust's ownership keeps CPU and GPU from racing: CPU access waits until submitted work using a buffer is done, and GPU writes need exclusive access to the buffer.

Three techniques keep memory low:

- **Reuse while the GPU runs.** A buffer dropped while submitted work still uses it goes back to the cache immediately; later GPU work may reuse it, ordered after those uses by its fences (MLX does the same).
- **In-place updates (donation).** When an update's input is owned by nothing else — no other handle, view or pending consumer — the update writes into it instead of copying. That covers inputs computed earlier in the same evaluation and inputs still being computed by earlier, in-flight work, which is what lets a decode loop submit step *n + 1* before reading token *n*. Element-wise operations and softmax do the same with an input of the output's dtype and shape, as in MLX.
- **Early release.** Intermediates of 16 MiB or more return to the allocator as soon as their last consumer is encoded, without waiting for the command buffer to be committed.

## Memory limits

The allocator follows MLX's rules (a size-bucketed cache, least-recently-used eviction, a memory limit at 1.5× the GPU's recommended working set capped at 95% of RAM). `nuvec::memory` has MLX's memory functions, under MLX's names without `get_`:

```rust,no_run
use nuvec::memory;

memory::set_memory_limit(24 << 30); // allocations and mappings beyond fail instead of swapping
memory::set_cache_limit(4 << 30);   // trim the buffer cache
println!("{}", memory::peak_memory()); // MLX's get_peak_memory
let stats = memory::stats();        // every statistic: active, peak, cache, limits, wired and mapped bytes
memory::clear_cache();
```

The memory limit bounds `active + mapped` and refuses with an out-of-memory error (`Error::is_oom`), whereas MLX treats its limit as a guideline. `active_memory` and `peak_memory` include memory-mapped checkpoints, like MLX's figures, which count the weights MLX copies. `MemoryStats` also splits out the allocator's own buffers (`active`, `peak`). `reset_peak_memory` resets both peaks. Without a Metal device, every figure is zero and setting a limit has no effect, as in MLX's CPU build.

## Wired memory

`memory::set_wired_limit` keeps buffers resident while the GPU runs, through Metal **residency sets** attached to the command queue. As in MLX 0.32, the wired buffers are spread over sets of at most 5% of the GPU's working set each (64 MiB at least; `MLX_RESIDENCY_SET_MAX_PCT`), up to 32, so a set that loses residency under memory pressure brings back only its own buffers; `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS` renews the sets' residency periodically. New buffers and memory-mapped checkpoints join the sets up to the limit; raising the limit wires waiting buffers, lowering it unwires them. Mapped checkpoints are not in `active` or `peak` (which count the allocator's buffers): `mapped`, `wired_mapped`, `active_with_mapped` and `peak_with_mapped` include them. It needs macOS 15 and a Metal 3 GPU (`memory::can_wire`), and the default is 0, as in MLX.

```rust,no_run
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let info = nuvec::metal::device_info().expect("a Metal GPU");
nuvec::memory::set_wired_limit(info.recommended_working_set as usize)?;
# Ok(())
# }
```

`nuvec_lm::WiredLimit` does this for the duration of a generation, as mlx-lm does.

## Neural Accelerators (M5)

M5-class GPUs have a Neural Accelerator in every GPU core, reached through Metal 4's TensorOps (`matmul2d` from Metal Performance Primitives). nuvec ships kernels for them in a second kernel library for macOS 26.2 and later; the baseline still runs on macOS 15. They cover dense GEMM (64 × 64 tiles, thin tiles of 16 or 32 rows or columns, and split K), affine 2/4/8-bit quantized products, and the sorted mixture-of-experts gather.

- `NUVEC_NAX=auto` (default), `on` or `off`; `nuvec::metal::nax()` reports whether they run and why not.
- Under `auto` they are off: they have not been measured on an M5, and an eligible M5 reports `Off(Unmeasured)`.
- `NUVEC_NAX=on` forces them on any GPU with TensorOps (every Apple GPU on macOS 26.2). On earlier GPUs they run on the ordinary GPU cores at about the speed of the SIMD kernels.
- `just nax-report` checks their correctness, benchmarks them against nuvec's SIMD kernels and MLX's NAX path, and writes a report.

## Profiling

Set `NUVEC_PROFILE=1` to record the GPU time of every kernel:

```sh
NUVEC_PROFILE=1 cargo run --release -p nuvec-lm --example bench -- \
    --model mlx-community/Qwen3-0.6B-4bit --text "Hello. " --repeats 160 --max-tokens 1
```

```text
GPU time, 961 prompt tokens: 823.0 ms
  498.16 ms  60.5%    382  nuvec_affine_qmm_t_bfloat16_gs64_b4_al1_batch0
  253.77 ms  30.8%     54  nuvec_steel_attention_bfloat16_…
   24.24 ms   2.9%    336  nuvec_copy_into_128_g3
```

In this mode every dispatch runs alone and is waited for, so each total is that kernel's own cost. The GPU idles between dispatches and its clocks drop, so absolute times are inflated: use the shares to find hotspots and the benchmarks to measure speed. `NUVEC_PROFILE=shapes` also splits totals by grid size; `nuvec::metal::profile::report()` returns the table to your program.

## Captures

`nuvec::metal::start_capture("run.gputrace")` and `stop_capture()` record the GPU work in between into a trace for Xcode, as MLX's `mx.metal.start_capture` does. Metal allows captures only with `MTL_CAPTURE_ENABLED=1` in the environment (or under Xcode); otherwise starting returns an error saying so. The file must end in `.gputrace` and not exist yet.
