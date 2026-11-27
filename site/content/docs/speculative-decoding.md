+++
title = "Speculative decoding"
description = "Speculative decoding in nuvec-lm: DFlash and DSpark block drafters, draft trees, the model's own multi-token-prediction head, prompt lookup and SuffixDecoding drafts, verified in one forward pass, with output identical to plain decoding on request — up to 2.8× faster."
weight = 100
[extra]
group = "Libraries"
+++

Decoding is bound by memory bandwidth: every step reads all the model's weights. **Speculative decoding** lets a cheap *drafter* propose the next tokens and verifies them all in one forward pass, so every accepted token saves most of a step.

## Using it

```rust,no_run
use nuvec_lm::{DraftPolicy, GenerateOptions, MtpDrafter, load, speculative_generate};

# fn main() -> nuvec::Result<()> {
let (mut model, tokenizer) = load("mlx-community/Qwen3.5-9B-4bit")?;
// mlx-community conversions drop the MTP head: attach it from the original checkpoint
// (`uv run scripts/extract_mtp.py Qwen/Qwen3.5-9B` downloads only those tensors).
model.attach_mtp("Qwen--Qwen3.5-9B.safetensors")?;
let prompt = tokenizer.encode("Write a story about a lighthouse keeper.");
let drafter = MtpDrafter::new(&model)?;
let mut generation =
    speculative_generate(&model, &prompt, GenerateOptions::default(), drafter, DraftPolicy::Auto)?;
let ids: Vec<u32> = generation.by_ref().collect::<nuvec::Result<_>>()?;
let stats = generation.stats(); // steps, verifies, drafted, accepted, pauses
# Ok(())
# }
```

Stops, `ignore_eos` and `min_tokens` apply as in `generate` (see [stops](@/docs/llm.md#stops)), and a stop inside an accepted draft ends generation at that token; `generation.finish_reason()` says why it ended.

Verification routes products as MLX's, so greedy output can differ from plain greedy decoding's in its low bits (and, at a near tie, in a token), as mlx-lm's can; a verification checks up to 15 draft tokens. With `GenerateOptions { row_exact: true, .. }`, greedy output is the model's own greedy output, **bit for bit**: drafts only decide how many tokens a step produces, never which. Each verified row is then computed as serial decoding computes it (row-exact products, `nuvec::with_row_exact`, and the attention planner's serial rule), on the GPU and on the CPU, with at most 7 draft tokens per verification, at some cost in speed.

## Drafters

| Drafter | Needs | Drafts well on |
|---|---|---|
| `DFlashDrafter` | a DFlash, DFlash 2 or DSpark drafter trained for the model (z-lab's: Qwen 3–3.8, Gemma 4, gpt-oss, Llama 3.1, …; DeepSeek's DSpark: Qwen3 4B–14B) | any text, best on reasoning and math |
| `MtpDrafter` | the model's multi-token-prediction head (Qwen 3.5, 3.6, 3.8) | any text: the head predicts the model's next-but-one token |
| `PromptLookup` | nothing | output that repeats its input: code edits, quotes, summaries, tool calls |
| `SuffixDecoding` | nothing; remembers earlier requests | agent loops that repeat earlier prompts and outputs |

- **DFlash** (arXiv 2602.06036) drafts a whole block in one forward pass of a small block-diffusion model that reads the target's hidden states from several layers. **DFlash 2** (e.g. `z-lab/Qwen3.8-27B-DFlash2`) adds convolutions over the block around each attention and MLP, and picks the draft with a candidate selector that rescores each position's top tokens against the previous choice. `DFlashDrafter::load(dir, &model)` loads either from a drafter directory in its bfloat16 weights; `load_quantized(dir, &model, Quant::affine(64, 4))` quantizes them on load (about twice as fast to draft). It shares the target's embeddings and output head, follows the drafter's RoPE (YaRN for gpt-oss), attention biases, embedding scale and logit soft cap (Gemma 4), and refuses drafters for another model. Not on pipelined models.
- **DSpark** (arXiv 2607.05147, DeepSeek's DeepSpec checkpoints such as `deepseek-ai/dspark_qwen3_8b_block7`) is the DFlash backbone with two small heads: every block slot predicts the next token, one after another with a low-rank bias from the previous token (the Markov head), and a confidence head scores each drafted token (`DFlashDrafter::last_confidence`). `DFlashDrafter::load` reads it; its copies of the target's embeddings and head are not loaded.
- **MTP.** The head (one decoder layer) takes the next token's embedding and the model's hidden state, and predicts the token after. Its first draft token comes from the last verified position; further ones reuse its own hidden state and are accepted less often. After each verification its cache is re-fed with the model's real hidden states, so it never drifts. `Model::attach_mtp` quantizes the head like the body (an 8-bit router stays 8-bit).
- **SuffixDecoding** (arXiv 2411.04975) matches the longest suffix of the context in the prompt and a bounded store of earlier sequences, and drafts the most frequent continuation. Pass `&mut SuffixDecoding` and call `remember` after each request to keep its store across requests.
- Write your own by implementing `Drafter::draft` (and `observe`, to follow the context; `needs_hidden`, to receive the model's hidden states; `candidates`, to draft trees).

## Draft trees

A **draft tree** (DDTree, arXiv 2604.12989) verifies the most likely prefixes of a DFlash drafter's per-position candidates in one pass instead of a single chain: each node attends to the context and its ancestors, and the cache keeps the accepted path. Its size each step maximises the expected tokens per measured verification time (EVICT, arXiv 2605.00342).

Trees apply to attention-only models (Qwen3, Qwen3-MoE, Llama, gpt-oss, Gemma 4) with DFlash drafters, and not with `row_exact` (a tree row attends over masked keys, so it cannot keep serial decoding's bits). DFlash 2 drafts chains (its selector scores pairs of positions).

## Policies

- **`DraftPolicy::Auto`** (default): each drafter's measured draft length for the model. DFlash and MTP time plain decoding, chains and draft trees, keep the fastest and probe the others now and then, so speculation that does not pay falls back to plain decoding. Other drafters use a kill switch: when the accepted tokens per verification over the last 16 verifications fall below the drafter's break-even, the next 64 tokens are decoded plainly, then drafting resumes.
- **`DraftPolicy::Tree(n)`**: a draft tree of up to `n` nodes every step where trees apply, a chain of `n` otherwise, for measurements.
- **`DraftPolicy::Adaptive(n)`**: up to `n` tokens, pausing 1, 2, 4, … 16 steps after a fully rejected draft (prompt lookup's classic back-off).
- **`DraftPolicy::Fixed(n)`**: up to `n` tokens every step, for measurements.

## Choosing an exact draft width

With the Qwen3.5-9B 4-bit target and its DFlash checkpoint in the local Hugging Face cache, run:

```sh
just bench-exact-verify --out target/exact-verify/report.json
```

Compare output tokens per second at fixed draft widths of three, five and seven. The benchmark checks each accepted prefix and the next target token against serial greedy generation. On a 48 GB M4 Pro, three exact drafts gave the highest throughput on all three Qwen3.5-9B prompts. [Measurements by prompt and width](/nuvec/experiments/exact-verify-m4-pro.json)

To inspect decisions in your own generation, call `generation.record_rounds(false)` before consuming tokens and `generation.take_rounds()` to drain completed rounds. Each round includes its output position, proposed IDs, accepted count and next ID. Call `record_rounds(true)` for stage timing, which synchronizes the draft on the device.

## Sampling

With a temperature, verifications use the lossless rule: a draft token is accepted with probability `min(1, p / q)` against the sampler's filtered distribution (temperature, top-p, min-p, top-k), and a rejected one is replaced by a sample from the residual. Tokens are distributed exactly as plain sampling's; a seed repeats a run.

The rule is also available on its own, on the device, for runtimes that bring their own drafters: `nuvec::speculative::accept_greedy` and `accept_sampled` return `[accepted, next, eos_hit]` in one `[3]` array, so a verification costs one small read. `nuvec::fast::gated_delta_steps_with` keeps every step's Gated DeltaNet state rounded to a storage dtype (bfloat16 caches), for exact rollback.

## Measuring speedups

Use `just bench-dflash` for a block drafter or `just bench-spec` for multi-token prediction. Compare output tokens per second with plain decoding on representative prompts. A wider draft can accept more tokens and still take longer to verify, especially when its rows route to different experts. Turn on `row_exact` when your application requires serial greedy output.

For repeated tool calls and file edits, replay a multi-turn trace with the `bench` example's `--trace FILE` option. Prompt lookup reuses text in the current context. SuffixDecoding can also use earlier turns.

## Scope

Speculation runs at batch 1 on the contiguous cache, for every supported model (draft trees on attention-only models): Gated DeltaNet, Mamba-2 and short-convolution layers keep each verified position's state and roll back to the last accepted one, and Mamba-2 verifies with decoding's one-step update, so with `row_exact` the tokens are plain decoding's. The `Batcher` and paged caches do not speculate.
