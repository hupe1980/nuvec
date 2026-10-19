+++
title = "Neural networks"
description = "nuvec-nn: modules as named parameter trees, Linear, quantized and LoRA layers, norms, RoPE, losses, and SGD, Adam, AdamW, Lion and Muon optimizers with schedules."
weight = 80
[extra]
group = "Libraries"
+++

`nuvec-nn` follows MLX's `mlx.nn` and `mlx.optimizers`. It is small on purpose: a module is a plain Rust struct that reports its parameters by name.

## Modules

```rust
use nuvec::{Array, Result, random};
use nuvec_nn::{Linear, Module, Parameters, losses, relu, value_and_grad};
use nuvec_nn::optim::{Adam, Optimizer};

#[derive(Clone)]
struct Mlp { hidden: Linear, out: Linear }

impl Module for Mlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("hidden", &mut self.hidden);
        p.module("out", &mut self.out);
    }
}

fn train(x: &Array, y: &Array) -> Result<()> {
    random::seed(0);
    let mut model = Mlp { hidden: Linear::new(784, 512, true)?, out: Linear::new(512, 10, true)? };
    let mut opt = Adam::new(1e-3);
    let loss = |m: &Mlp| {
        let logits = m.out.forward(&relu(&m.hidden.forward(x)?)?)?;
        losses::cross_entropy(&logits, y, losses::Reduction::Mean)
    };
    for _ in 0..100 {
        let (value, grads) = value_and_grad(&model, loss)?;
        opt.update(&mut model, &grads)?;
    }
    Ok(())
}
```

Parameters are named by path (`hidden.weight`, `layers.3.mlp.up_proj.scales`), so checkpoints load by name with `load_weights(weights, strict)`. `freeze` marks parameters as not trainable; `trainable_parameters` lists the rest.

## Layers

- `Linear`, `QuantizedLinear` (MLX's affine quantization), `Embedding`, `QuantizedEmbedding`;
- `RmsNorm`, `LayerNorm` and `Rope` on the fused kernels;
- `LoraLinear` over a full-precision or quantized base layer (LoRA and QLoRA);
- activations: `relu`, `silu`, `gelu_approx`, `gelu_fast_approx`.

## Losses and optimizers

- `losses::cross_entropy` with mean, sum or no reduction, on the fused kernel.
- `optim`: `Sgd` (momentum, Nesterov, weight decay), `Adam` and AdamW (`Adam::adamw`), `Lion` and `Muon`.
- Learning-rate schedules: exponential, step and cosine decay, linear warm-up, and joined schedules.
- `clip_grad_norm(gradients, max_norm)` returns the clipped gradients and the total norm.

Results match MLX's layers and optimizers on recorded fixtures. The repository's `just mnist` trains MLX's MNIST example model and compares it with MLX.
