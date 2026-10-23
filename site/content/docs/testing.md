+++
title = "Testing against MLX"
description = "How nuvec verifies correctness: differential tests replaying recorded MLX results across dtypes, strides and gradients, reference models from mlx-lm, and Miri for undefined behaviour."
weight = 130
[extra]
group = "Project"
+++

A reimplementation is only useful if it computes the same thing. nuvec treats MLX as the oracle.

## Recorded MLX results

`scripts/mlx_fixtures.py` runs nearly a thousand cases through the pinned MLX version and records inputs and outputs into a binary file that the Rust test suite replays — no Python needed to run the tests. Cases cover:

- every operation over every dtype, with edge values (NaN, infinities, integer limits);
- random layouts: transposes, negative strides, broadcasts and offsets;
- type promotion and broadcasting between all dtype pairs;
- gradients (`vjp` and `jvp`) and batching (`vmap`) of every primitive;
- random-number streams, which must be bit-identical for a key.

Tolerances are derived per operation from the dtype's machine epsilon and the operation's error bound (for example, products scale with the inner dimension and the operands' magnitudes), so a passing test means "as accurate as MLX", not "close enough".

## Reference models

`scripts/lm_fixtures.py` builds tiny random models of every supported architecture with mlx-lm, in float32, bfloat16 and 4-bit, and records their logits and greedy continuations. nuvec-lm must match the logits within tight bounds and produce the same tokens — including Hugging Face layouts of the same weights. Layers and optimizers are checked against `mlx.nn` and `mlx.optimizers` the same way.

## Beyond MLX

- GPU kernels are checked against the CPU implementation on random shapes, including unaligned and transposed operands.
- Memory behaviour (in-place updates, buffer reuse, wired memory) is checked through allocator statistics.
- Results are bit-identical from run to run, also with memory reused under pending GPU work, and kernels compiled from embedded sources give the precompiled library's results.
- The core runs under **Miri** to catch undefined behaviour, and Metal's API validation layer runs over the GPU runtime tests.

`just ci` runs CI's gates locally: formatting, clippy, tests, docs, Miri, cargo-deny, the Metal library and hygiene checks. CI also tests the latest stable Rust and builds the portable crates on Linux.
