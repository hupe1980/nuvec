+++
title = "Benchmarks"
description = "nuvec-lm against mlx-lm on the same Mac: prefill and decode speed for seven models from 0.6B to 35B parameters, plus kernel-level comparisons with MLX and the method behind them."
weight = 120
[extra]
group = "Project"
+++

nuvec's performance target is MLX on the same machine. Every comparison here comes from a command in the repository, runs both engines on the same Mac on the same day, and reports losses as well as wins.

## Text generation

`just bench-parity` runs nuvec-lm and mlx-lm 0.32.2 over a roster of cached mlx-community models. Each engine loads the model once per pass and warms up; passes alternate (nuvec, mlx-lm, nuvec, mlx-lm), and the table shows medians of four runs. Both are timed identically from outside their generation loops: **prefill** is prompt tokens over the time to the first token, **decode** is the following tokens over the time after the first. Greedy decoding, 128 tokens. Ratios above 1.00× favour nuvec.

Apple M4 Pro, macOS 26.6:

| Model | Kind | Prefill 25 / 961 / 3,841 tokens | Decode 25 / 961 / 3,841 tokens |
|---|---|---|---|
| Qwen3-0.6B 4-bit | dense | 1.25× / 1.07× / 1.03× | 1.09× / 1.07× / 1.06× |
| Llama-3.2-3B-Instruct 4-bit | dense | 1.08× / 1.00× / 1.00× | 1.01× / 1.01× / 1.01× |
| Qwen3-8B 4-bit | dense | 1.18× / 1.01× / 1.00× | 1.03× / 1.01× / 1.02× |
| Qwen3.5-9B 4-bit | hybrid | 1.20× / 1.00× / 0.99× | 0.99× / 0.99× / 0.99× |
| Qwen3.8-27B 4-bit | hybrid | 1.17× / 1.00× / 1.00× | 1.00× / 0.99× / 1.00× |
| Qwen3-30B-A3B 4-bit | MoE | 1.11× / 0.99× / 0.99× | 1.03× / 1.03× / 1.03× |
| Qwen3.6-35B-A3B 4-bit | hybrid MoE | 1.14× / 0.98× / 0.97× | 1.02× / 1.01× / 1.02× |

Geometric means: **prefill 1.05×, decode 1.02×.** Absolute speeds at 25 prompt tokens, nuvec / mlx-lm: Qwen3-0.6B 433 / 396 tok/s, Qwen3-8B 54.0 / 52.6, Qwen3-30B-A3B 96.0 / 93.1, Qwen3.6-35B-A3B 91.8 / 90.3.

Speculative decoding adds up to **2.16×** on top for repetitive output; see [speculative decoding](@/docs/speculative-decoding.md).

## Kernels

`bench-gemm`, `bench-quant`, `bench-fast` and `bench-compile` run the same kernels in nuvec and MLX with the same method (sustained asynchronous bursts after a warm-up, best of three, alternating programs). Highlights:

- Quantized products at language-model shapes, dense (961 × 2048 × 8192) and mixture-of-experts gathers (7,688 rows over 256 experts): 1.00–1.02× MLX's time.
- Prefill attention in bfloat16 (2,048 and 3,841 keys, grouped-query heads): 0.98–1.02×.
- Quantized products of 4–12 rows (speculative verification): 12–22% faster than MLX's per-row kernel, through nuvec's own few-row kernel.
- Decode attention in half precision with keys streamed from memory: about 9% slower than MLX with the same kernel, the one known gap.

## Run them yourself

The benchmarks need [uv](https://docs.astral.sh/uv/) (to install the pinned MLX) and the models in your Hugging Face cache:

```sh
just bench-parity                      # the generation table (cached models only)
just bench-parity mlx-community/Qwen3-8B-4bit
just bench-quant gqmm                  # kernel benchmarks take an optional name filter
just bench-fast sdpa
```

Machines with heavy memory pressure (swap in use) show noisy numbers; the tables above note conditions when they were taken.
