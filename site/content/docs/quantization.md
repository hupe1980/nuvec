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

NVFP4 takes an optional float32 global scale, as MLX's `global_scale`: `ops::quantize_scaled(w, g)` makes the scales relative to `g / (448 · 6)`, and the result carries `g` (`Quantized::global_scale`, or `with_global_scale` for a checkpoint's), so `dequantize`, `quantized_matmul` and `gather_qmm` apply it. `qqmm` and `gather_qqmm` take the activations' scale (`global_scale_x`) with the weight's, both or neither, with MLX's kernels; without global scales a quantized weight's `qqmm` routes as `quantized_matmul` (MLX `main`).

`view` reinterprets bits as another dtype (MLX's `mx.view`), and `to_fp8` / `from_fp8` convert floats to E4M3 bytes and back; checkpoints that store F8 tensors load them as `uint8`. E4M3 encodes NaN with code `0x7f` or `0xff` and saturates infinities at ±448. The E8M0 scale byte `0xff` decodes to quiet NaN, which propagates across its entire group.

## Products on the GPU

The GPU chooses kernels by shape and hardware. Few-row products can reuse a weight group across rows. Larger products use tiled matrix multiplication, with split-K when the inner dimension is long and the output small. Batch size can affect rounding.

For speculative verification and batches of independent sequences, enable `GenerateOptions::row_exact` or build the product under `nuvec::with_row_exact`. With up to eight rows, each result equals its single-row product bit for bit. Use `BENCH_ROWS=5 just bench-rows affine8` to time a five-row product on your Mac. [Eight-bit measurements](/nuvec/experiments/affine8-five-rows-m4-pro.json)

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

With rows sorted by expert (`sorted_indices`), at least 16 of them (more than 64 under `with_row_exact`) and at least four per expert on average, a grouped kernel reads each expert's weights once; otherwise each row gets its own product. NVFP4 experts can carry one float32 global scale each (`Quantized::global_scale`, shape `[experts]`), as mlx-lm's `quantize_model(..., global_scale=True)` writes them. `gather_mm` is the same for full-precision weights.

## Training through quantized weights

Quantized products have derivative rules for the activations, and in affine mode for the scales and biases, so you can backpropagate through frozen quantized layers — the basis of QLoRA-style fine-tuning with `nuvec-nn`'s `QuantizedLinear` and `LoraLinear`. Asking for the derivative of packed codes, `uint8` scales or a global scale is an error.
