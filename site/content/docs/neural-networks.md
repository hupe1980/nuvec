+++
title = "Neural networks"
description = "nuvec-nn: modules as named parameter trees, Linear, quantized and LoRA layers, norms, RoPE, losses, and SGD, Adam, AdamW, Lion and Muon optimizers with schedules."
weight = 80
[extra]
group = "Libraries"
+++

`nuvec-nn` follows MLX's `mlx.nn` and `mlx.optimizers`. A model is a plain Rust struct of layers with `#[derive(Module)]`: its parameters are named by the field names, as MLX's (`layers.0.weight`), and the core transforms differentiate it directly, returning gradients of the model's own type.

## Modules

```rust
use nuvec::{Array, Result, random, value_and_grad};
use nuvec_nn::optim::{Adam, Optimizer};
use nuvec_nn::{Linear, Module, losses, relu};

#[derive(Clone, Module)]
struct Mlp { hidden: Linear, out: Linear }

impl Mlp {
    fn forward(&self, x: &Array) -> Array {
        self.out.forward(&relu(&self.hidden.forward(x)))
    }
}

fn train(x: &Array, y: &Array) -> Result<()> {
    random::seed(0);
    let mut model = Mlp { hidden: Linear::new(784, 512), out: Linear::new(512, 10) };
    let mut opt = Adam::new(1e-3);
    let loss = |m: &Mlp| losses::cross_entropy(&m.forward(x), y, losses::Reduction::Mean);
    for _ in 0..100 {
        let (_value, grads) = value_and_grad(loss)(&model)?; // grads: Mlp
        opt.update(&mut model, &grads)?;
    }
    println!("{}", model.display()); // Mlp (407,050 params, 407,050 trainable) ...
    Ok(())
}
# fn main() -> Result<()> {
#     let x = random::normal([8, 784], nuvec::DType::F32, 0.0, 1.0, None);
#     train(&x, &nuvec::array![0u32, 1, 2, 3, 4, 5, 6, 7])
# }
```

Write the closure's parameter type (`|m: &Mlp|`): the transforms are generic, so Rust cannot infer it (E0282 otherwise).

## Fields and attributes

```rust
use nuvec::{Array, Param};
use nuvec_nn::{Linear, Module, RmsNorm};

#[derive(Clone, Module)]
struct Attention {
    q_proj: Linear,                        // a submodule: q_proj.weight, q_proj.bias
    #[tree(rename = "q_norm")] qn: Option<RmsNorm>, // `None` contributes nothing
    #[tree(skip)] heads: i32,              // configuration, hashed into the structure
    #[tree(skip)] freqs: Array,            // a computed constant: not saved, not loaded
    #[tree(buffer)] running_mean: Array,   // saved and loaded, never trained
    gate: Param,                           // a raw parameter
}
# fn uses(a: &Attention) { let _ = (&a.q_proj, &a.qn, a.heads, &a.freqs, &a.running_mean, &a.gate); }
```

- A field typed `Array` needs a role: `Param` (a parameter), `#[tree(buffer)]` or `#[tree(skip)]`; the derive rejects a bare one. The check is syntactic (a type alias of `Array` is not caught).
- A `#[tree(skip)]` field must be `Static` (numbers, strings, `#[derive(Static)]` types): its value is part of the model's structure, so a compiled step retraces when it changes. `#[tree(skip, static_unchecked)]` opts out (the value is then baked into compiled traces).
- Enums work as modules: a newtype variant is transparent (`Dense(Linear)` names `weight`), `#[tree(rename = "self_attn")]` adds a segment. There are no trait-object fields: use an enum of the concrete layers.
- Generated code names only `::nuvec::__derive`; a crate that depends on `nuvec_core` alone writes `#[tree(crate = "nuvec_core")]`.

## Using a model

- Names and counts: `parameters()`, `trainable_parameters()`, `num_params()`, `num_trainable()`, `summary()` (a table of path, shape, dtype, trainable, bytes) and `display()` (the nested modules with their counts).
- Freezing by path: `model.freeze(|_| true)`, then `model.unfreeze(|p| p.contains("lora_"))`; both return how many parameters changed. Quantized weights are buffers and never become trainable.
- `model.train(false)` switches every dropout layer to evaluation mode (`train(true)` back; `training()` reads it).
- Weights by name: `load_weights(path_or_named_arrays)` is strict and reports every missing, unknown and mismatched name in one error, leaving the model unchanged; `load_weights_partial` ignores unknown names; `save_weights(path)` writes MLX-compatible safetensors (`save_trainable` only the trainable leaves).
- `map_modules::<T>(filter, f)` edits or replaces every submodule of type `T` whose path matches (applying LoRA, say).

## Layers

Constructors take no `bool` and return no `Result`; initialisation draws MLX's values for a seed.

- `Linear::new(i, o)` (`.without_bias()`), `QuantizedLinear::from_linear(&l)` (MLX's affine quantization; `from_linear_with(&l, group_size, bits)`), `Embedding::new(n, d)`, `QuantizedEmbedding::from_embedding(&e)`;
- `RmsNorm::new(d).eps(e)`, `LayerNorm::new(d).eps(e).affine(false)` / `.without_bias()` and `Rope::new(d).traditional().base(b).scale(s)` on the fused kernels;
- `LoraLinear::from_base(base, rank)` over a full-precision or quantized base layer (LoRA and QLoRA), with `.scale(s)`, `.dropout(p)` and `.rslora()`;
- `Dropout::new(p)`, as MLX's `nn.Dropout`;
- activations: `relu`, `silu`, `gelu_approx`, `gelu_fast_approx`.

## Losses and optimizers

- `losses::cross_entropy` with mean, sum or no reduction, on the fused kernel.
- `optim`: `Sgd` (momentum, `.nesterov()`, weight decay), `Adam` (`.betas`, `.eps`, `.bias_correction()`), `AdamW` (MLX's default decay 0.01), `Lion` and `Muon`. `opt.update(&mut model, &grads)` updates the trainable leaves from a gradient tree of the model's type.
- The optimizer's state is a tree: `step`, `learning_rate` (0-D arrays, as MLX's `optimizer.state`) and the slots by parameter path (`slots.layers.0.weight.m`). Freezing or unfreezing never moves or resets state. `opt.init(&model)` creates it up front; `set_learning_rate(lr)` writes the rate. `Box<dyn Optimizer>` works.
- Parameter groups (MLX's `MultiOptimizer`): `MultiOptimizer::new(Sgd::new(1e-2)).group(|p| p.contains("lora_a"), Adam::new(1e-3))`; the first matching group wins.
- Learning-rate schedules: exponential, step and cosine decay, linear warm-up, and joined schedules, as float32 functions of the step array.
- `clip_grad_norm(&grads, max_norm)` returns the clipped gradient tree and the total norm.

## A compiled training step

`compile_mut` compiles the whole step (forward, gradients, update) with the model and the optimizer as its state, as MLX's `mx.compile(step, inputs=state, outputs=state)`:

```rust
use nuvec::{Array, Result, array, compile_mut, random, value_and_grad};
use nuvec_nn::losses::{Reduction, cross_entropy};
use nuvec_nn::optim::{Optimizer, Sgd};
use nuvec_nn::{Linear, Module};

#[derive(Clone, Module)]
struct Mlp { layers: Vec<Linear> }

impl Mlp {
    fn forward(&self, x: &Array) -> Array {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i + 1 < self.layers.len() {
                h = nuvec_nn::relu(&h);
            }
        }
        h
    }
}

# fn main() -> Result<()> {
random::seed(0);
let mut state = (Mlp { layers: vec![Linear::new(4, 8), Linear::new(8, 3)] }, Sgd::new(0.1));
state.1.init(&state.0);
let step = compile_mut(|(model, opt): &mut (Mlp, Sgd), (x, y): &(Array, Array)| {
    let loss_fn = |m: &Mlp| cross_entropy(&m.forward(x), y, Reduction::Mean);
    let (loss, grads) = value_and_grad(loss_fn)(model)?;
    opt.update(model, &grads)?;
    Ok(loss)
});
let x = random::normal([4, 4], nuvec::DType::F32, 0.0, 1.0, None);
let batch = (x, array![0u32, 1, 2, 0]);
for _ in 0..10 {
    let _loss = step(&mut state, &batch)?; // traces once, then replays
}
# Ok(())
# }
```

The step traces once: the learning rate and step counter are arrays in the optimizer's state, and dropout inside draws fresh masks every call (the random key is threaded through). Switching `train(false)`, freezing a layer or changing a hyperparameter retraces once.

## Checkpoints

A whole run, model and optimizer, saves and resumes by path in two lines:

```rust,no_run
# use nuvec::Result;
# use nuvec_nn::optim::{Adam, Optimizer};
# use nuvec_nn::{Linear, Module};
# fn main() -> Result<()> {
# let mut state = (Linear::new(4, 2), Adam::new(1e-3));
nuvec::save("run.safetensors", &state)?;
// ... later, in a fresh process:
state.1.init(&state.0);
nuvec::load_into("run.safetensors", &mut state)?;
# Ok(())
# }
```

`save_weights` writes only the model (MLX's `save_weights`; MLX loads it), `save_trainable` only what is trained (an adapter). For LoRA on LLMs, `nuvec_lm::lora::linear_to_lora_layers` replaces projections by path and `save_adapters`/`load_adapters` read and write mlx-lm's adapter directories.

Results match MLX's layers, optimizers and dropout masks on recorded fixtures. The repository's `just mnist` trains MLX's MNIST example model (with a `compile_mut` step) and compares it with MLX.
