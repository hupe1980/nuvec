+++
title = "LLM runtime"
description = "nuvec-lm runs Llama, Mistral and Qwen 2/3/3.5 models from mlx-community or Hugging Face checkpoints on Apple Silicon at mlx-lm's speed, with samplers and speculative decoding."
weight = 90
[extra]
group = "Libraries"
+++

`nuvec-lm` loads language models and generates text the way [mlx-lm](https://github.com/ml-explore/mlx-lm) does — the same model definitions, checkpoint conventions, samplers and decode loop — on nuvec instead of MLX.

## Supported models

| Family | `model_type` | Notes |
|---|---|---|
| Llama, Mistral | `llama`, `mistral` | Llama 3 RoPE scaling, tied embeddings |
| Qwen 2 | `qwen2` | q/k/v biases |
| Qwen 3 | `qwen3` | per-head q/k norms |
| Qwen 3 MoE | `qwen3_moe` | sparse layers per `decoder_sparse_step`, dense `mlp_only_layers` |
| Qwen 3.5 / 3.6 | `qwen3_5`, `qwen3_5_moe` | hybrid Gated DeltaNet and gated attention, partial RoPE, dense or mixture-of-experts |

Checkpoints load from a directory or a Hugging Face repository id found in the local cache — mlx-community's quantized checkpoints as well as original Hugging Face ones (fused or per-expert MoE weights, multimodal wrappers). Weights are memory-mapped, not copied; nothing is downloaded.

## Generating

```rust
use nuvec_lm::{GenerateOptions, Sampler, StreamDecoder, generate, load};

let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
let prompt = tokenizer.encode("<|im_start|>user\nWhat is unified memory?<|im_end|>\n<|im_start|>assistant\n");
let options = GenerateOptions {
    max_tokens: 256,
    sampler: Sampler { temperature: 0.7, top_p: 0.9, ..Default::default() },
    ..Default::default()
};
let mut out = StreamDecoder::default();
for id in generate(&model, &prompt, options)? {
    let id = id?;
    if model.config.eos_token_ids.contains(&id) { break; }
    print!("{}", out.push(&tokenizer, id));
}
```

- **Samplers:** greedy (temperature 0, the default), temperature, top-p, min-p and top-k, with mlx-lm's filters.
- **Tokenizer:** byte-level BPE read from `tokenizer.json`, special tokens included; `encode` adds the tokens the tokenizer's post-processor adds (Llama's `<|begin_of_text|>`), as mlx-lm does. `StreamDecoder` turns token ids into text without splitting multi-byte characters.
- **Cache:** `Generation::cache()` exposes the KV cache (and Gated DeltaNet state) to continue a conversation.

## How generation is fast

- **Chunked prefill** (2,048 tokens at a time, as mlx-lm) with the cache written once per position.
- **A pipelined decode loop:** step *n + 1* is built and submitted while the GPU still runs step *n*; token *n* is read afterwards. The KV cache is updated in place, even while the previous step is still in flight.
- **MLX's kernels** for every hot operation, mlx-lm's compiled activations, and fused kernels that shorten each layer's chain of dependent dispatches (residual add + norm; q/k norm + RoPE + cache write).
- **Wired memory** while generating (`WiredLimit`), as mlx-lm does, so weights are not paged out between steps.

See [benchmarks](@/docs/benchmarks.md) for the numbers and [speculative decoding](@/docs/speculative-decoding.md) for going beyond them.

## Command line

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Explain borrow checking" --chat \
    --max-tokens 256 --temp 0.7 --top-p 0.9
```

`--chat` wraps the prompt in the ChatML template used by Qwen models.
