+++
title = "Benchmarks"
description = "Measure model throughput, latency, memory and Metal kernel timings against MLX on the same Mac."
weight = 120
[extra]
group = "Project"
+++

Run `just bench-parity --tier smoke` with the Qwen3-8B and Qwen3-30B-A3B checkpoints in your Hugging Face cache. The quick tier adds a hybrid model and gpt-oss. Both engines run on the same Mac with the same token IDs. Each record includes checkpoint revisions, software versions and individual samples.

## Checking MLX parity

Run `just qualify-mlx` on the M4 Pro with local snapshots of Qwen3-8B-4bit and Qwen3.5-9B-4bit. The command uses MLX 0.32.5 and mlx-lm 0.32.2. It verifies the CPU and Metal fixtures, then compares 256 generated tokens on each of eight frozen prompts per checkpoint. The JSON report at `target/qualification/report.json` identifies the first divergent step when a prompt fails. For fixtures alone, run `just qualify-mlx --fixture-only`.

### M4 Pro qualification results

All 168 Metal rows matched the pinned MLX release. On macOS 27.2.1 the CPU profile has 98 exact rows and 70 with recorded numerical differences. All 16 dense and hybrid prompts matched for 4,096 generated tokens in total. [Numerical and per-prompt results](/nuvec/qualification/m4-pro-mlx-0.32.5.json)

All 40 required quick-tier cells passed the 3% regression gate against nuvec v0.15.0. The table gives geometric means of the per-cell median ratios against mlx-lm 0.32.2 using MLX 0.32.5. Ratios above 1 favour nuvec. [Individual ratios and raw samples](/nuvec/benchmarks/qualification/m4-pro-mlx-0.32.5-quick.json)

| Measured workload | Cells | Ratio against mlx-lm |
|---|---:|---:|
| Decode, three prompt lengths | 12 | 1.010× throughput |
| Time to first token, 2,048 prompt tokens | 4 | 0.995× inverse time |
| Allocator peak, load/decode/prefill | 12 | 1.014× inverse peak bytes |
| Warm load to first token | 4 | 1.443× inverse time |
| Four concurrent requests | 4 | 1.035× throughput |

## Method

Decode measures output tokens per second after prefill. Time to first token includes prefill. Warm load measures loading from the filesystem cache through the first generated token. The memory comparison includes mapped weights alongside the GPU allocator peak.

Ratios against mlx-lm use median samples. The previous-release gate compares each nuvec build's best sample, then repeats any cell below the threshold. The smoke threshold is 0.95× and the quick-tier threshold is 0.97×. See the JSON for individual cells and confirmed regressions. [Benchmark archive](/nuvec/benchmarks/index.json)

## Short matrix products on Metal

Run `just bench-gemm qk14 skinny9 skinny14 skinny15` for short transposed projections. Use `--help` to list shapes. These FP16 timings on the M4 Pro use the best of two alternating samples. The baseline is nuvec commit `2defccd`. [FP16 and BF16 samples](/nuvec/experiments/wide-gemv-m4-pro.json)

| Projection (M × K × N) | Baseline | Wide GEMV | MLX 0.32.5 |
|---|---:|---:|---:|
| 9 × 1024 × 4096 | 88.08 µs | 33.20 µs | 36.71 µs |
| 14 × 4096 × 4096 | 180.92 µs | 217.06 µs | 195.86 µs |
| 15 × 1024 × 4096 | 88.53 µs | 56.50 µs | 53.33 µs |

## Run them yourself

The benchmarks need [uv](https://docs.astral.sh/uv/), which installs the pinned MLX, and the models in your Hugging Face cache:

```sh
just bench-parity --tier smoke                     # two models, about 10 minutes
just bench-parity                                  # quick tier, about 45 minutes
just bench-parity mlx-community/Qwen3-8B-4bit --sections decode,ttft
just bench-parity --resume --budget 540            # continue a long run in steps
just bench-validate                                # check records against the schema
just bench-compare base.json candidate.json        # fails on a ratio drop above 2%
just bench-quant gqmm                              # kernel benchmarks take a name filter
```

Swap in use makes the numbers noisy; each record stores the swap in use at its start.
