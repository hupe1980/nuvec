+++
title = "Quantization"
description = "MLX-compatible quantization in nuvec: affine packing of 1 to 6 or 8 bits, the MXFP4, MXFP8 and NVFP4 floating-point formats, quantized matrix products with MLX's kernels, and mixture-of-experts gather products."
weight = 70
[extra]
group = "GPU"
+++

nuvec implements MLX's quantization modes bit for bit — **affine** and the floating-point formats **MXFP4**, **MXFP8** and **NVFP4** — so mlx-community checkpoints load without conversion and products match MLX's.

## Affine

Weights are split into groups of `group_size` values along each row (32, 64 or 128). Each group stores a scale and a bias, and each value becomes an integer code of `bits` bits (1, 2, 3, 4, 5, 6 or 8), packed into `uint32` words:

```text
w ≈ scale · code + bias
```

4-bit weights with groups of 64 take about 4.5 bits per weight including scales and biases: about a quarter of `bfloat16`'s memory and bandwidth.

```rust
# use nuvec::prelude::*;
# use nuvec::random;
use nuvec::Quant;
# let w = random::normal([256, 512], DType::F32, 0.0, 1.0, None);
# let x = random::normal([4, 512], DType::F32, 0.0, 1.0, None);
let q = w.quantize(Quant::affine(64, 4)); // w: [out, in]; codes, scales and biases
let y = x.quantized_matmul(&q, true);     // x · wᵀ
let w_approx = q.dequantize(None);
# assert_eq!(y.shape(), [4, 256]);
# assert_eq!(w_approx.shape(), [256, 512]);
```

## Floating-point formats

The floating-point modes store each value as a small float with one shared scale per group, without biases:

| Mode | Values | Group | Scale | Bits per weight |
|---|---|---|---|---|
| `QuantMode::Mxfp4` | E2M1 (4-bit) | 32 | E8M0 (power of two) | 4.25 |
| `QuantMode::Mxfp8` | E4M3 (8-bit) | 32 | E8M0 | 8.25 |
| `QuantMode::Nvfp4` | E2M1 | 16 | E4M3 | 4.5 |

Scales are `uint8` codes. Codes and scales equal MLX's Metal quantizer's on both devices, including rounding, subnormals and NaN.

```rust
# use nuvec::prelude::*;
# use nuvec::random;
use nuvec::QuantMode;
# let w = random::normal([256, 512], DType::F32, 0.0, 1.0, None);
# let x = random::normal([4, 512], DType::F32, 0.0, 1.0, None).astype(DType::BF16);
let q = w.quantize(QuantMode::Mxfp4);  // codes and uint8 scales; q.biases is None
let y = x.quantized_matmul(&q, true);  // in x's dtype
let w_approx = q.dequantize(None);     // bfloat16
# assert_eq!((y.shape(), y.dtype()), (&[4, 256][..], DType::BF16));
# assert_eq!(w_approx.dtype(), DType::BF16);
```

Weights quantized elsewhere (a checkpoint's tensors) become a `Quantized` with `Quantized::affine(weight, scales, biases, group_size, bits)` (biases required, as in MLX) or `Quantized::fp(weight, scales, mode)`.

`view` reinterprets bits as another dtype (MLX's `mx.view`), and `to_fp8` / `from_fp8` convert floats to E4M3 bytes and back; checkpoints that store F8 tensors load them as `uint8`.

## Products on the GPU

Products use MLX's kernels, chosen by shape as MLX chooses them: matrix-vector kernels for few rows, as in decode; tiled kernels for many rows; split-K when the output is small but the inner dimension long. From 2 rows, MLX's multi-vector kernel reads each weight group once for several rows, so a row's low bits can depend on how many rows share a product, as in MLX. Under `nuvec::with_row_exact` (speculative verification, `BatcherOptions::batch_invariant`), up to 8 rows run as matrix-vector products in which each row's result equals its single-row product bit for bit.

## Mixture of experts

`gather_qmm` multiplies each row by the expert weights chosen for it, the core of an MoE layer, without gathering weights first:

```rust
# use nuvec::prelude::*;
# use nuvec::random;
use nuvec::{GatherQmm, Quant};
# let q = random::normal([8, 128, 256], DType::F32, 0.0, 1.0, None).quantize(Quant::affine(64, 4));
# let x = random::normal([4, 1, 256], DType::F32, 0.0, 1.0, None);
# let experts = array![0u32, 0, 3, 5];
// x: [rows, 1, in]; q: [experts, out, …]; experts: [rows] (uint32), sorted
let y = x.gather_qmm(&q, None, Some(&experts), GatherQmm::default().sorted_indices(true));
# assert_eq!(y.shape(), [4, 1, 128]);
```

With rows sorted by expert (`sorted_indices`), at least 16 of them (more than 64 under `with_row_exact`) and at least four per expert on average, a grouped kernel reads each expert's weights once; otherwise each row gets its own product. NVFP4 experts can carry one float32 global scale each (`GatherQmm::global_scale`), as mlx-lm's `quantize_model(..., global_scale=True)` writes them. `gather_mm` is the same for full-precision weights.

## Training through quantized weights

Quantized products have derivative rules for the activations, and in affine mode for the scales and biases, so you can backpropagate through frozen quantized layers — the basis of QLoRA-style fine-tuning with `nuvec-nn`'s `QuantizedLinear` and `LoraLinear`. Asking for the derivative of packed codes, `uint8` scales or a global scale is an error.
