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
- The core runs under **Miri** to catch undefined behaviour, and Metal's API validation layer runs over the GPU runtime tests.

`just ci` runs everything CI runs: formatting, clippy, tests on the minimum and latest Rust, docs, Miri, cargo-deny and the Metal library checks.
