+++
title = "GPU, memory and profiling"
description = "How nuvec runs on the Apple GPU: asynchronous Metal command buffers, unified memory, MLX's allocator, memory limits, wired memory with residency sets, and a per-kernel profiler."
weight = 50
[extra]
group = "GPU"
+++

## How work reaches the GPU

Evaluating a graph encodes one Metal compute dispatch per node into command buffers and commits them without waiting:

- **Growing command buffers.** An evaluation that starts with the GPU idle commits after 8 dispatches, so the GPU starts quickly, then doubles the batch while the GPU is busy (up to 256).
- **Minimal barriers.** Dispatches that don't depend on each other run concurrently within a command buffer; a memory barrier is inserted only when a dispatch reads what an earlier one wrote.
- **Ordering by fences.** Command buffers are ordered through per-buffer fences, so independent work from different evaluations can overlap on the GPU.
- **Errors.** A GPU failure is reported by the first read of an affected array, and every later GPU result on the device reports it too, as in MLX.

Kernels are compiled ahead of time into `nuvec.metallib` when Apple's Metal toolchain is installed; otherwise their sources are embedded and compiled when first used (`NUVEC_METALLIB=off` selects this mode).

## Unified memory, safely

Buffers live in shared memory: the CPU reads GPU results in place. Rust's ownership keeps CPU and GPU from racing: CPU access waits until submitted work using a buffer is done, and GPU writes need exclusive access to the buffer.

Two techniques keep memory low:

- **Reuse while the GPU runs.** A buffer dropped while submitted work still uses it goes back to the cache immediately; later GPU work may reuse it, ordered after those uses by its fences (MLX does the same).
- **In-place updates (donation).** When an update's input is owned by nothing else — no other handle, view or pending consumer — the update writes into it instead of copying. That covers inputs still being computed by earlier, in-flight work, which is what lets a decode loop submit step *n + 1* before reading token *n*.

## Memory limits

The allocator follows MLX's rules (a size-bucketed cache, least-recently-used eviction, a memory limit at 1.5× the GPU's recommended working set capped at 95% of RAM). Limits and statistics live on the process-wide device:

```rust
let gpu = nuvec::gpu::Device::shared().expect("a Metal GPU");
gpu.set_memory_limit(24 << 30);  // allocations beyond fail instead of swapping
gpu.set_cache_limit(4 << 30);    // trim the buffer cache
let stats = gpu.memory();        // active, peak, cache, limits, wired bytes
gpu.clear_cache();
```

## Wired memory

`set_wired_limit` keeps buffers resident while the GPU runs, through a Metal **residency set** attached to the command queue, instead of making them resident per command buffer (which can page memory back in under pressure). New buffers and memory-mapped checkpoints join the set up to the limit; raising the limit wires waiting buffers, lowering it unwires them. It needs macOS 15 and a Metal 3 GPU (`can_wire`), and the default is 0, as in MLX.

```rust
gpu.set_wired_limit(gpu.info().recommended_working_set as usize)?;
```

`nuvec_lm::WiredLimit` does this for the duration of a generation, as mlx-lm does.

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

In this mode every dispatch runs alone and is waited for, so kernels never overlap and each total is that kernel's own cost. `NUVEC_PROFILE=shapes` also splits totals by grid size. Your own programs can read the table with `nuvec::gpu::profile::report()`.

Because the GPU idles between dispatches in this mode, its clocks drop and absolute times are inflated; use the shares to find hotspots and the benchmarks to measure speed.
