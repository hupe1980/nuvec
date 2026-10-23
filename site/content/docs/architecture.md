+++
title = "Architecture"
description = "How nuvec is built: the crates, the lazy graph and its primitives, evaluation and scheduling, the Metal backend, memory management and the design choices behind them."
weight = 140
[extra]
group = "Project"
+++

## Crates

| Crate | Responsibility |
|---|---|
| `nuvec` | Facade: re-exports the everyday API and the layers below |
| `nuvec-core` | Arrays, graph nodes, primitives, dtypes, evaluation, derivative and batching rules, GPU dispatch |
| `nuvec-compile` | Tracing, graph simplification, element-wise fusion; `compile_mut` for training steps |
| `nuvec-macros` | `#[derive(Tree)]`, `#[derive(Module)]`, `#[derive(Static)]` |
| `nuvec-gpu` | Metal runtime: device, pipelines, buffers, command encoding, allocator, residency sets, profiler |
| `nuvec-kernels` | Metal kernel sources and the precompiled library |
| `nuvec-cpu` | Pure-Rust CPU kernels over typed slices |
| `nuvec-io` | safetensors, npy, IDX; memory-mapped files |
| `nuvec-nn` | Modules (every tree is one), layers, losses, optimizers |
| `nuvec-lm` | Models, tokenizer, KV cache, generation, speculative decoding |

`unsafe` code stays in the Metal runtime (`nuvec-gpu`, including file mapping), `nuvec-core`'s three kernel-dispatch calls and one CPU matrix-product call in `nuvec-cpu`; the other crates contain none.

## The graph

An `Array` points to a node holding a **primitive** (what to compute), its inputs, and one slot per output. Primitives form a closed Rust `enum`, so every rule — output shapes, derivatives, batching, fusion, GPU and CPU evaluation — is an exhaustive `match`: adding a primitive without a rule does not compile.

Promotion and broadcasting happen in the operations layer, which inserts explicit conversion and broadcast nodes, so primitives always see operands of equal shape and dtype. Views are primitives too, sharing their input's buffer.

## Evaluation

1. **Order.** The graph is walked breadth-first from the outputs (MLX's order), so independent nodes are adjacent and the GPU runs them concurrently. `schedule` does this ahead of time.
2. **Run.** Each node runs as a GPU kernel, as CPU code over the same memory, or as a view; nodes are detached after evaluation, so intermediates are freed as soon as their consumers have run.
3. **Publish.** GPU outputs are first exposed as in-flight data — later GPU work may read them at once — and are published to their slots once their command buffer completes, in commit order.

Ownership is used as a proof: when an evaluation holds the only strong reference to an input, it may write into the input's buffer. Pending work holds nodes only weakly, so this works even while the input is still being computed.

## Memory

The allocator rounds sizes into buckets, caches freed buffers, evicts least recently used ones under its limits, and lets GPU work reuse memory that earlier GPU work still uses, ordered by fences. A residency set wires buffers and mapped checkpoints up to a limit.

## Design choices

- **MLX semantics, Rust idioms.** Same promotion table, broadcasting, transforms and kernels as MLX; failed arrays and `Result` at evaluation boundaries instead of exceptions, ownership instead of reference-count heuristics, `Send + Sync` arrays.
- **Ported kernels first.** MLX's Metal kernels are the reference for speed; nuvec ports them, keeps their attribution, and tests against MLX. Its own kernels cover what MLX does not fuse or tile for (residual norms, norm + RoPE + cache writes, few-row quantized products).
- **No C++.** Metal is reached through Apple's system frameworks via [`objc2-metal`](https://crates.io/crates/objc2-metal); the CPU path is pure Rust.
