+++
title = "LLM runtime"
description = "nuvec-lm runs Llama, Mistral, Qwen 2/3/3.5, gpt-oss and Gemma 4 models from mlx-community or Hugging Face checkpoints on Apple Silicon, matching mlx-lm's outputs and speed, with samplers, stops, speculative decoding, batched serving, and pipeline or tensor parallelism across Macs."
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
- **Memory:** aligned checkpoints (files nuvec writes are) are memory-mapped without a copy, and only the pages of the tensors the model keeps are mapped: the vision and multi-token-prediction weights it drops are neither mapped, wired nor counted. mlx-community checkpoints have unpadded headers and are read once, in parallel, skipping the weights the language model does not use (as mlx-lm).
- **Aligned cache:** with `NUVEC_ALIGNED_CACHE=<dir>`, a misaligned checkpoint is copied once into `<dir>` with every tensor at a 256-byte boundary (`nuvec::io::align_safetensors`). Every later load maps that copy without a copy, at the cost of a second copy on disk. The copy is rebuilt when the source's size or modification time changes, and nuvec does not evict copies.
- Loading runs nothing through the model; `Model::warm_up` builds its kernels ahead of a first request.

## Generating

```rust,no_run
use nuvec_lm::chat::{self, ChatTemplate, RenderOptions};
use nuvec_lm::{GenerateOptions, Sampler, StreamDecoder, generate, load};

# fn main() -> nuvec::Result<()> {
let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
let template = ChatTemplate::from_dir(nuvec_lm::resolve("mlx-community/Qwen3-8B-4bit")?)?;
// mlx_lm.generate's prompt: the template over a user message, with a generation prompt.
let prompt = chat::prompt_ids(Some(&template), &tokenizer, "What is unified memory?", None, None, RenderOptions::new())?;
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

- **End of sequence:** generation stops at the model's end-of-sequence ids (those of `config.json` and `generation_config.json`, as mlx-lm) without yielding them, or at `max_tokens`. `Generation::finish_reason()` says which ended it. With `ignore_eos`, they are ordinary tokens, as in vLLM, and generation runs to `max_tokens` or a stop.
- **Samplers:** greedy (temperature 0, the default), temperature, top-p, min-p and top-k, with mlx-lm's filters chained in mlx-lm's order. Settings are checked when generation starts (`Sampler::validate`). With `GenerateOptions::seed`, sampled tokens are reproducible, also when batched with other requests.
- **Tokenizer:** BPE read from `tokenizer.json`, special tokens included: byte-level (GPT-2, Llama 3, Qwen 2/3/3.5 and o200k's gpt-oss split patterns), Tekken's (Mistral NeMo, Ministral, Mistral Small 3), and SentencePiece-style with byte fallback (Gemma, Llama 2, Mistral v0.1–v0.3); `encode` adds the tokens the tokenizer's post-processor adds (Llama's `<|begin_of_text|>`), as mlx-lm does; `encode_text` does not. `StreamDecoder` turns token ids into text without splitting multi-byte characters.
- **Cache:** `Generation::cache()` exposes the KV cache (and Gated DeltaNet state) for inspection.

## Chat templates

`nuvec_lm::chat` renders a checkpoint's own chat template exactly as transformers' `apply_chat_template` does (feature `chat`, on by default; pure-Rust `minijinja` in transformers' environment):

```rust,no_run
use nuvec::io::json;
use nuvec_lm::chat::{ChatTemplate, RenderOptions};

# fn main() -> nuvec::Result<()> {
let dir = nuvec_lm::resolve("mlx-community/Qwen3-8B-4bit")?;
let template = ChatTemplate::from_dir(&dir)?;
let tokenizer = nuvec_lm::Tokenizer::from_file(dir.join("tokenizer.json"))?;
let messages = json::parse(br#"[{"role": "system", "content": "Be brief."},
                                {"role": "user", "content": "Hi!"}]"#)?;
let options = RenderOptions::new()
    .generation_prompt(true)
    .arg("enable_thinking", json::Json::Bool(false));
let text = template.render(messages.as_array().unwrap(), &options)?;
let ids = template.encode(&tokenizer, messages.as_array().unwrap(), &options)?;
# Ok(())
# }
```

- **Sources:** `tokenizer_config.json`'s `chat_template` (one, or a named list), `chat_template.jinja` and `additional_chat_templates/*.jinja`, in transformers' precedence; with tools, `tool_use` is chosen when it exists. `TokenizerConfig` holds the special tokens the templates read (`bos_token`, `eos_token`, …).
- **Options:** tools, documents, a generation prompt or a continued final message, template arguments (`enable_thinking`, …), and the time `strftime_now` reads (UTC when unset; transformers uses the local time).
- **Exactness:** text and ids equal transformers 5.22.1's for 33 mlx-community checkpoints (Qwen 2.5 to 3.6, Llama 3.x, Mistral, Ministral, Gemma 4, gpt-oss) over 42 cases each, including tools, tool calls, reasoning and Python's number and `json.dumps` spelling (`scripts/chat_fixtures.py`).
- **Errors:** a template's `raise_exception` (for example "roles must alternate") is an error carrying its message.

## Stops

`GenerateOptions::stop` ends generation at stop tokens or stop strings, as vLLM's `stop_token_ids` and `stop`; `min_tokens` delays every ending:

```rust,no_run
use nuvec_lm::{FinishReason, GenerateOptions, Stop, TextStream, generate, load};

# fn main() -> nuvec::Result<()> {
let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
let prompt = tokenizer.encode("Q: Name three planets.\nA:");
let options = GenerateOptions {
    stop: Stop::strings(&tokenizer, ["\nQ:", "\n\n"])?.with_tokens(tokenizer.token_id("<|im_end|>")),
    min_tokens: 4,
    ..Default::default()
};
let mut text = TextStream::new(&tokenizer, &options);
let mut generation = generate(&model, &prompt, options)?;
for id in generation.by_ref() {
    print!("{}", text.push(id?));
}
print!("{}", text.finish());
if let Some(FinishReason::Stop(stopped)) = generation.finish_reason() {
    eprintln!("\nstopped at {stopped:?}");
}
# Ok(())
# }
```

- **Stop tokens** end generation like end-of-sequence, and are not yielded.
- **Stop strings** are matched on the bytes of the generated text as the tokenizer decodes it, so a stop string still matches when it spans tokens or splits a character. Generation ends at the token that completes the earliest match. That token is yielded, since tokens are atomic, and `Stopped::String { index, cut }` says which string matched and how many bytes to cut from the end of the decoded output. `TextStream` applies the cut and holds back any text that could begin a stop string, so streamed text never shows one, even in part. A request may have up to 64 stop strings of at most 1 KB each. `Stop::strings` reads every token's bytes once, so build it once and clone it for each request.
- **`min_tokens`:** the first `min_tokens` tokens are never end-of-sequence or stop tokens (their probability is zero), and no stop string ends generation before then.
- **The same everywhere:** `generate`, `Batcher` and `speculative_generate` end alike, with the same finish reasons. A stop inside an accepted draft ends speculative decoding at that token.

## How generation is fast

- **Chunked prefill** (2,048 tokens at a time, as mlx-lm) with the cache written once per position.
- **A pipelined decode loop:** step *n + 1* is built and submitted while the GPU still runs step *n*; token *n* is read afterwards. The KV cache is updated in place, even while the previous step is still in flight.
- **MLX's kernels** for every hot operation, mlx-lm's compiled activations, and fused kernels that shorten each layer's chain of dependent dispatches (residual add + norm; q/k norm + RoPE + cache write; mixture-of-experts routing and weighting).
- **Wired memory** while generating (`WiredLimit`), as mlx-lm does, so weights are not paged out between steps. The allocator's cache is emptied where mlx-lm empties it: after each prefill chunk, after the first generated token and every 256 after it, and every 512 `Batcher` steps.

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

`Batcher` queues requests, admits them while there is room, prefills them (on quantized models, prompts of the same length over 64 tokens admitted together are prefilled in one batch) and decodes them together. Each finishes as it would alone: at an end-of-sequence token, a stop, or its `max_tokens`. Decoding is pipelined as in `generate`: each `step` submits the next step before reading the previous one's tokens, so one step can be in flight between calls; a request that ends meanwhile emits nothing more.

Products route as MLX routes them, so a request's output can differ in its low bits depending on what it is batched with, as with mlx-lm's `BatchGenerator`. `BatcherOptions::batch_invariant` serves every request exactly the tokens it gets alone, at some cost in throughput (up to about 20% on mixture-of-experts models at 8 requests).

```rust,no_run
use nuvec_lm::{Batcher, BatcherOptions, Event, GenerateOptions, load};

# fn main() -> nuvec::Result<()> {
let (model, _tokenizer) = load("mlx-community/Qwen3-0.6B-4bit")?;
# let prompts: Vec<Vec<u32>> = vec![vec![9707, 11], vec![785, 6722, 315]];
let mut batcher = Batcher::new(&model, BatcherOptions { max_batch: 8, ..Default::default() })?;
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

`BatcherOptions` names every setting:
- `max_batch` is 32 by default, as mlx-lm's `BatchGenerator`.
- `block_size` is 16 by default.
- `blocks` is sized from memory by default: 90% (`memory_fraction`) of what the GPU's recommended working set leaves after the weights, at most room for `max_batch` sequences of 131,072 positions each. `Model::kv_bytes_per_position()` gives a position's size.
- `batch_invariant` is described above; `micro_batches` applies to pipelined models (below).

Invalid options are refused with an error naming the field.

When the cache runs out of blocks, the most recently admitted request is preempted and later prefilled again with the tokens it has produced; its later tokens can then differ in low bits from an uninterrupted run.

`submit` rejects empty prompts, prompts that could never fit in the cache and invalid options (`GenerateOptions::validate`). `cancel(id)` finishes a waiting or running request with `FinishReason::Cancelled` at the next step. A request that outgrows the whole cache finishes with `CacheFull`; one whose prefill or decoding step fails finishes with `Error`, which holds the `nuvec::Error` (`is_oom()` tells out-of-memory apart), while the others go on (a step commits its cache positions only after its forward pass succeeded).

## Several Macs: pipeline parallelism

A model too large for one Mac runs across several, one pipeline stage per Mac, as mlx-lm's `--pipeline`. Each rank runs a contiguous range of the layers: rank 0 the last, the highest rank the first.
- **Stages:** the stages pass each token's hidden state on with `send`/`recv`, a few kilobytes per token per link, so an ordinary Ethernet or Wi-Fi network is enough.
- **Loading:** each rank reads only its stage's tensors. The embeddings are read on the first stage only, the final norm and head on the last only (and the embeddings there too when they are tied).
- **Sampling:** only rank 0 computes the head and samples. Every rank receives its tokens, so `generate`, `decode_batch` and `Batcher` yield the same events on every rank, even on Macs of different GPU generations.

Start one process per Mac with MLX's launcher (the ring over TCP, or JACCL over Thunderbolt 5), each running the same program:

```rust,no_run
use nuvec_lm::{GenerateOptions, Split, generate, load_model_pipelined};

# fn main() -> nuvec::Result<()> {
let group = nuvec::distributed::init(true)?;
let model = load_model_pipelined("/models/Qwen3-32B-4bit", &group, Split::memory())?;
let options = GenerateOptions { seed: Some(0), ..GenerateOptions::default() };
for token in generate(&model, &[9707, 11], options)? {
    let _token = token?; // the same on every rank
}
# Ok(())
# }
```

**Split:**
- `Split::Even` is mlx-lm's even split (the extra layers go to the low ranks).
- `Split::Layers(vec![..])` gives the layers per rank, rank 0 first.
- `Split::memory()` sizes each stage by the Mac's free GPU memory, keeping 15% free for the KV cache. A group too small for the model fails on every rank before loading, naming the bytes.
- `Model::pipeline(&group, split)` does the same on a model already loaded whole.

**Serving:** a `Batcher` over a pipelined model splits the running requests into one micro-batch per stage and keeps them all in flight, so every Mac works on a different one. One request decodes at about one stage's speed plus the network hops; concurrent requests fill the pipeline. `BatcherOptions { micro_batches: Some(1), .. }` steps them as one batch.

**Limits:**
- Gemma 4 is not supported: its per-layer inputs and shared KV layers would cross stages, and mlx-lm does not pipeline it either.
- Speculative decoding on a pipelined model is an error.
- A Mac that leaves fails every rank's pending step, naming it, within the distributed timeout.

### Tensor parallelism

To decode one request faster across Macs cabled with Thunderbolt 5 (JACCL), split every layer instead, as mlx-lm's `model.shard(group)`. Each Mac keeps a slice of every attention, MLP, expert and Gated DeltaNet projection, and each layer's two outputs are summed over the group. An untied output head is split by vocabulary when the vocabulary divides. With greedy or seeded sampling every rank samples the same token from its own logits, as mlx-lm; otherwise rank 0 samples and shares its token.

```rust,no_run
use nuvec_lm::{GenerateOptions, generate, load_model_sharded};

# fn main() -> nuvec::Result<()> {
let group = nuvec::distributed::init(true)?;
let model = load_model_sharded("/models/Qwen3-30B-A3B-4bit", &group)?;
for token in generate(&model, &[9707, 11], GenerateOptions::default())? {
    let _token = token?;
}
# Ok(())
# }
```

- `Model::shard(&group)` does the same on a loaded model.
- Sizes that do not divide (heads, hidden units, whole quantization groups) are an error naming the layer, and the model is left as it was. With fewer key/value heads than ranks, the ranks share them.
- Every rank adds its partial sums in its own order, so logits match the single-process ones to within rounding, not bit for bit.
- Gemma 4 is not supported, and a model is either pipelined or sharded, not both.

Two ranks over Ethernet are slower than one Mac (an all-reduce per layer per token), so tensor parallelism is for JACCL.

`examples/pipeline_generate.rs` (`--tensor` for tensor parallelism) is a ready-made program for `mlx.launch`.

## Fine-tuning

`nuvec_lm::tuner` is mlx-lm's training loop: its JSONL datasets, sorted padded batches, the masked next-token loss, a compiled step over the model and any optimizer (gradient accumulation included), validation, and the trained weights saved in mlx-lm's layout (`mlx_lm.generate --adapter-path` loads them).

```sh
cargo run --release -p nuvec-lm --example lora -- \
    --model models/Qwen3-0.6B-4bit --data data --train --iters 1000 --batch-size 4
```

The `lora` example takes `mlx_lm.lora`'s options and defaults, and its YAML files (`--config`, with `lora_parameters`, `optimizer_config` and `lr_schedule`):

- `--fine-tune-type`: `lora`, `dora` or `full` (every parameter of the last `--num-layers` blocks, on a full-precision model);
- `--optimizer`: `adam`, `adamw`, `muon`, `sgd` or `adafactor`;
- `--grad-checkpoint`: each block recomputes its activations in the backward pass;
- `--resume-adapter-file`, `--test` and `--test-batches`, `--grad-accumulation-steps`.

`--model` is a local model directory; `--data` holds `train.jsonl` and, optionally, `valid.jsonl` and `test.jsonl`, in one of mlx-lm's formats (the first row decides):

- `{"text": ...}`: the text, ended with the tokenizer's EOS;
- `{"messages": [...], "tools": [...]}`: the conversation rendered by the checkpoint's chat template;
- `{"prompt": ..., "completion": ...}`: a user message and the assistant's answer.

With `--mask-prompt`, the loss counts only the last message of a chat row, or the completion. The keys can be renamed (`text_feature`, `chat_feature`, `prompt_feature`, `completion_feature` in the `--config` file). Rows, offsets and the batches' order (numpy's permutation from `--seed`) equal mlx-lm's (`Dataset::from_rows`, `Dataset::rows`). On chat rows with `--mask-prompt` (Qwen3-0.6B-4bit), the losses agree with `mlx_lm.lora`'s within 2.2e-3.

The loss is computed over row chunks of the output head (`Model::token_losses`), so the `[tokens, vocab]` logits never exist, and each LoRA layer keeps only its input for the backward pass. Against `mlx_lm.lora` on an M4 Pro (batch 4): Qwen3-0.6B-4bit trains at 1.13× its speed at 0.42× its peak memory, Llama-3.2-3B-Instruct-4bit at 1.02× and 0.60×, and with rows of about 2,000 tokens and `--grad-checkpoint` in both, Qwen3-0.6B at 1.03× and 0.28× (`scripts/bench_lora.py`).

From your own code: `lora::apply(&mut model, &AdapterConfig::new(16, params).fine_tune_type(..))`, then `tuner::train(&mut model, &mut optimizer, ..)`; `tuner::build_optimizer` and `tuner::build_schedule` take mlx-lm's names, and the loop reports the loss, speed and peak memory as `Event`s.

## Command line

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Explain borrow checking" \
    --max-tokens 256 --temp 0.7 --top-p 0.9
```

As `mlx_lm.generate`, the prompt goes through the checkpoint's chat template: `--system-prompt`, `--prefill-response` (the start of the answer), `--chat-template-config '{"enable_thinking": false}'` (template arguments) and `--ignore-chat-template` (the raw prompt). `--min-p` and `--top-k` set the other samplers.
