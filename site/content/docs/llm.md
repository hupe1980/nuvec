+++
title = "LLM runtime"
description = "nuvec-lm runs Llama, Mistral, Qwen 2/3/3.5, gpt-oss and Gemma 4 models from mlx-community or Hugging Face checkpoints on Apple Silicon, matching mlx-lm's outputs and speed, with samplers, speculative decoding and batched serving."
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
| Qwen 3.5 / 3.6 | `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` | hybrid Gated DeltaNet and gated attention, partial RoPE, dense or mixture-of-experts |
| gpt-oss | `gpt_oss` | sliding and full layers, attention sinks, YaRN, biased router and experts with the clamped SwiGLU; MXFP4 and affine checkpoints, Hugging Face's MXFP4 originals converted at load |
| Gemma 4 | `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` | sliding and full layers with their own head sizes and RoPE, per-layer inputs, KV-shared layers, K = V, the experts branch, the logit soft cap; vision and audio weights are skipped |

`load` takes a directory or a Hugging Face repository id found in the local cache; nothing is downloaded.

- **mlx-community checkpoints:** affine, MXFP4, MXFP8 and NVFP4, with per-layer modes.
- **Original Hugging Face checkpoints:** fused or per-expert MoE weights and multimodal wrappers; AutoAWQ (4-bit) and GPTQ (4- and 8-bit) are converted at load to MLX's affine layout.
- **Not loaded directly:** Hugging Face MXFP4 (except gpt-oss's originals), FP8, compressed-tensors and ModelOpt checkpoints; convert them with `mlx_lm.convert` first.
- **Memory:** aligned checkpoints (files nuvec writes are) are memory-mapped without a copy. mlx-community checkpoints have unpadded headers and are read once, in parallel, skipping the vision and multi-token-prediction weights the language model does not use (as mlx-lm).
- Loading runs nothing through the model; `Model::warm_up` builds its kernels ahead of a first request.

## Generating

```rust,no_run
use nuvec_lm::{GenerateOptions, Sampler, StreamDecoder, generate, load};

# fn main() -> nuvec::Result<()> {
let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
let prompt = tokenizer.encode("<|im_start|>user\nWhat is unified memory?<|im_end|>\n<|im_start|>assistant\n");
let options = GenerateOptions {
    max_tokens: 256,
    sampler: Sampler { temperature: 0.7, top_p: 0.9, ..Default::default() },
    ..Default::default()
};
let mut out = StreamDecoder::default();
for id in generate(&model, &prompt, options)? {
    print!("{}", out.push(&tokenizer, id?));
}
# Ok(())
# }
```

- **End of sequence:** generation stops at the model's end-of-sequence ids (those of `config.json` and `generation_config.json`, as mlx-lm) without yielding them, or at `max_tokens`.
- **Samplers:** greedy (temperature 0, the default), temperature, top-p, min-p and top-k, with mlx-lm's filters chained in mlx-lm's order. Settings are checked when generation starts (`Sampler::validate`). With `GenerateOptions::seed`, sampled tokens are reproducible, also when batched with other requests.
- **Tokenizer:** BPE read from `tokenizer.json`, special tokens included: byte-level (GPT-2, Llama 3, Qwen 2/3/3.5 and o200k's gpt-oss split patterns) and SentencePiece-style with byte fallback (Gemma, Llama 2, Mistral v0.1–v0.3); `encode` adds the tokens the tokenizer's post-processor adds (Llama's `<|begin_of_text|>`), as mlx-lm does. Other split patterns (Mistral's Tekken) are not supported. `StreamDecoder` turns token ids into text without splitting multi-byte characters.
- **Cache:** `Generation::cache()` exposes the KV cache (and Gated DeltaNet state) for inspection.

## How generation is fast

- **Chunked prefill** (2,048 tokens at a time, as mlx-lm) with the cache written once per position.
- **A pipelined decode loop:** step *n + 1* is built and submitted while the GPU still runs step *n*; token *n* is read afterwards. The KV cache is updated in place, even while the previous step is still in flight.
- **MLX's kernels** for every hot operation, mlx-lm's compiled activations, and fused kernels that shorten each layer's chain of dependent dispatches (residual add + norm; q/k norm + RoPE + cache write).
- **Wired memory** while generating (`WiredLimit`), as mlx-lm does, so weights are not paged out between steps.

See [benchmarks](@/docs/benchmarks.md) for the numbers and [speculative decoding](@/docs/speculative-decoding.md) for going beyond them.

## Batched decoding

`PagedCache` holds the keys and values of many sequences in fixed-size blocks; `decode_batch` advances several of them, each at its own length, in one forward pass. Prefill each sequence as usual, then move it into the paged cache:

```rust,no_run
use nuvec::prelude::*;
use nuvec_lm::{PagedCache, decode_batch, load};

# fn main() -> Result<()> {
let (model, _tokenizer) = load("mlx-community/Qwen3-0.6B-4bit")?;
# let prompt_without_last_token = Array::from_slice(&[9707u32, 11], [1, 2]);
# let last_token = 1879u32;
let mut paged = PagedCache::new(&model, 4096, 32, 8)?; // 32-position blocks, 8 sequences
let mut kv = model.new_cache();
model.forward(&prompt_without_last_token, &mut kv).eval()?; // [1, L] token ids
let id = paged.insert(&kv)?;
let logits = decode_batch(&model, &[last_token], &[id], &mut paged)?; // [B, vocab]
let next = logits.argmax(-1);                          // the next token of each sequence
paged.remove(id)?;                                     // its blocks are free again
# Ok(())
# }
```

Each sequence gets exactly the tokens it would get decoding alone. Hybrid models keep each sequence's Gated DeltaNet state in a slot of a state pool, updated in place.

## Serving many requests

`Batcher` queues requests, admits them while there is room, prefills them (on quantized models, prompts of the same length over 64 tokens admitted together are prefilled in one batch) and decodes them together; each finishes at an end-of-sequence token or its `max_tokens`. Decoding is pipelined as in `generate`: each `step` submits the next step before reading the previous one's tokens, so one step can be in flight between calls; a request that ends meanwhile emits nothing more.

Products route as MLX routes them, so a request's output can differ in its low bits depending on what it is batched with, as with mlx-lm's `BatchGenerator`. `Batcher::new(..)?.batch_invariant(true)` serves every request exactly the tokens it gets alone, at some cost in throughput (up to about 20% on mixture-of-experts models at 8 requests).

```rust,no_run
use nuvec_lm::{Batcher, Event, GenerateOptions, load};

# fn main() -> nuvec::Result<()> {
let (model, _tokenizer) = load("mlx-community/Qwen3-0.6B-4bit")?;
# let prompts: Vec<Vec<u32>> = vec![vec![9707, 11], vec![785, 6722, 315]];
let mut batcher = Batcher::new(&model, 4096, 32, 8)?; // blocks, block size, batch
for prompt in prompts {
    batcher.submit(prompt, GenerateOptions::default())?;
}
while batcher.is_busy() {
    for event in batcher.step()? {
        match event {
            Event::Token { id, token } => { /* stream it */ }
            Event::Finished { id, reason } => { /* done */ }
        }
    }
}
# Ok(())
# }
```

When the cache runs out of blocks, the most recently admitted request is preempted and later prefilled again with the tokens it has produced; its later tokens can then differ in low bits from an uninterrupted run.

`submit` rejects empty prompts, prompts that could never fit in the cache and invalid sampler settings. `cancel(id)` finishes a waiting or running request with `FinishReason::Cancelled` at the next step. A request that outgrows the whole cache finishes with `CacheFull`; one whose prefill or decoding step fails finishes with `Error`, while the others go on (a step commits its cache positions only after its forward pass succeeded).

## Command line

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Explain borrow checking" --chat \
    --max-tokens 256 --temp 0.7 --top-p 0.9
```

`--chat` wraps the prompt in the ChatML template used by Qwen models.
