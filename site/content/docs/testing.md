+++
title = "Testing against MLX"
description = "How nuvec verifies correctness: differential tests replaying recorded MLX results across dtypes, strides and gradients, reference models from mlx-lm, and Miri for undefined behaviour."
weight = 130
[extra]
group = "Project"
+++

Run `just ci` for formatting, lint, workspace tests, documentation, Miri and Metal checks. The Rust tests read checked-in reference data. Regenerating the fixtures requires Python and the pinned MLX environment.

## Recorded MLX results

Array fixtures cover dtype promotion, noncontiguous views, gradients, batching and keyed random streams. Element-wise GPU tests compare output hashes with MLX, including every float16 and bfloat16 input word for unary functions. Product tolerances account for dtype and reduction length.

Run `just qualify-mlx --fixture-only` for the CPU and Metal numerical checks. The reference CPU ledger records chunk differences by operation and dtype for the M4 Pro on macOS build 26B131. [CPU replay and first-difference diagnostics](@/docs/arrays.md#checking-cpu-activation-results)

## Reference models

Run `just qualify-mlx` with Qwen3-8B-4bit and Qwen3.5-9B-4bit in your Hugging Face cache to compare 256 greedy tokens on each of eight frozen prompts per checkpoint. For another cached checkpoint, use `just parity-tokens MODEL`.

### Packed AutoAWQ and GPTQ checkpoints

```sh
NUVEC_REQUIRE_GPU=1 cargo test -p nuvec-lm --test lm packed_checkpoints::
just lm-fixtures --only packed
```

The test checks prompt logits and 12 greedy tokens with whole-prompt and chunked prefill for Qwen3 AutoAWQ, Llama GPTQ v1 and Qwen3 GPTQ v2. Logits must match bit for bit on the recording GPU architecture. Other architectures use an absolute tolerance of 1% of the largest absolute reference logit, with a minimum reference scale of one.

The second command regenerates the three seeded checkpoints and their reference outputs. Use `just lm-fixtures --help` to list the fixture groups.
