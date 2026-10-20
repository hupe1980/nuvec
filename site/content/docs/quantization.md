+++
title = "Quantization"
description = "MLX-compatible affine quantization in nuvec: 1–8-bit packing, quantized matrix products with MLX's kernels, and mixture-of-experts gather products."
weight = 70
[extra]
group = "GPU"
+++

nuvec implements MLX's **affine** quantization bit for bit, so mlx-community checkpoints load without conversion and products match MLX's.

## How it works

Weights are split into groups of `group_size` values along each row (32, 64 or 128). Each group stores a scale and a bias, and each value becomes an integer code of `bits` bits (1 to 8), packed into `uint32` words:

```
w ≈ scale · code + bias
```

4-bit weights with groups of 64 take about 4.5 bits per weight including scales and biases — a quarter of `bfloat16`'s memory and bandwidth, which is what decode speed depends on.

```rust
use nuvec::{dequantize, quantize, quantized_matmul};

let (wq, scales, biases) = quantize(&w, 64, 4)?;              // w: [out, in]
let y = quantized_matmul(&x, &wq, &scales, &biases, true, 64, 4)?; // x · wᵀ
let w_approx = dequantize(&wq, &scales, &biases, 64, 4)?;
```

On the GPU, products use MLX's kernels chosen by shape: matrix-vector kernels (`qmv`, `qmv_fast`) for single rows, as in decode, tiled kernels (`qmm_t`, `qmm_n`) for many rows, and split-K when the output is small but the inner dimension long. For 4-bit products of 4 or more rows below the tiled kernels' threshold (speculative verification, small batches), nuvec's own kernel unpacks each weight once for two rows; each row's result equals the single-row product bit for bit, so results do not depend on how many rows share a product.

## Mixture of experts

`gather_qmm` multiplies each row by the expert weights chosen for it, the core of an MoE layer, without gathering weights first:

```rust
use nuvec::gather_qmm;

// x: [rows, 1, in]; wq/scales/biases: [experts, out, …]; experts: [rows] (uint32)
let y = gather_qmm(&x, &wq, &scales, &biases, None, Some(&experts), true, 64, 4, true)?;
```

With rows sorted by expert (the last argument) and at least four rows per expert on average, a grouped kernel reads each expert's weights once; with fewer, each row gets its own product, as in MLX. `gather_mm` is the same for full-precision weights.

## Training through quantized weights

Quantized products have derivative rules for the activations and for the scales and biases, so you can backpropagate through frozen quantized layers — the basis of QLoRA-style fine-tuning with `nuvec-nn`'s `QuantizedLinear` and `LoraLinear`.
