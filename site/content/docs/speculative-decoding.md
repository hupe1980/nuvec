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

The report pairs fixed widths of three, five and seven drafts with serial greedy generation on copy, code and prose prompts. Each verification also includes its seed token. The benchmark checks every proposed prefix and the target's next token, including the final partial round. Compare the per-prompt decode rates and their seven-to-three geometric mean. Each fenced eight-row verification is timed against a paired plain decode run. Time to the first token includes preparing the drafter in this benchmark.

On November 27, 2026, on a 48 GB M4 Pro running macOS 27.2.1, seven exact drafts produced **0.729×** the output tokens/s of three across the three prompts (geometric mean). Each exact run emitted the same 128 tokens as serial generation. The fast seven-draft control diverged on the coding prompt. The machine was also running workspace tests. [Raw samples and verification decisions](/nuvec/experiments/exact-verify-m4-pro.json)

| Prompt | Plain tok/s | 3 drafts tok/s | 5 drafts tok/s | 7 drafts tok/s | Eight-row target time / plain step |
|---|---:|---:|---:|---:|---:|
| Copy | 47.75 | 70.51 | 62.98 | 54.84 | 2.70× |
| Code | 43.77 | 60.19 | 49.91 | 40.09 | 2.91× |
| Prose | 45.79 | 53.85 | 50.95 | 40.31 | 3.20× |

An experimental kernel that shares a weight load across eight rows passed the strided row-independence test and emitted the same 128 tokens on a held-out Python prompt. It reduced median seven-draft throughput from 73.75 to 57.84 tok/s across three paired runs, so the runtime retains the existing kernel. [Held-out prompt and paired observations](/nuvec/experiments/exact-verify-eight-row-trial.json)

To inspect decisions in your own generation, call `generation.record_rounds(false)` before consuming tokens and `generation.take_rounds()` to drain completed rounds. Each round includes its output position, proposed IDs, accepted count and next ID. Call `record_rounds(true)` for stage timing, which synchronizes the draft on the device.

## Sampling

With a temperature, verifications use the lossless rule: a draft token is accepted with probability `min(1, p / q)` against the sampler's filtered distribution (temperature, top-p, min-p, top-k), and a rejected one is replaced by a sample from the residual. Tokens are distributed exactly as plain sampling's; a seed repeats a run.

The rule is also available on its own, on the device, for runtimes that bring their own drafters: `nuvec::speculative::accept_greedy` and `accept_sampled` return `[accepted, next, eos_hit]` in one `[3]` array, so a verification costs one small read. `nuvec::fast::gated_delta_steps_with` keeps every step's Gated DeltaNet state rounded to a storage dtype (bfloat16 caches), for exact rollback.

## Results

DFlash in fast mode on Apple M4 Pro, code (a CSV-processing function in each model's chat format), 256 greedy tokens, 4-bit drafters, drafts of 7:

| Target + drafter | Plain decoding | DFlash | Drafts accepted |
|---|---:|---:|---|
| Qwen3.5-9B + `z-lab/Qwen3.5-9B-DFlash` | 44.7 tok/s | **91.7 tok/s (2.05×)** | 200 of 388 |
| Qwen3.8-27B + `z-lab/Qwen3.8-27B-DFlash2` | 11.3 tok/s | **32.1 tok/s (2.8×)** | 210 of 321 |

Measured under load from other work; Qwen3.8-27B decodes 15.1 tok/s plainly on a quiet machine (2.1×). A drafter quantized on load adds about 1 GB to peak memory (its tensors are read one at a time).

Qwen3-8B with `z-lab/Qwen3-8B-DFlash-b16` against dflash-mlx 0.1.8 (MLX 0.32.4) in the same session (`just bench-dflash`: the GSM8K "Natalia" problem in Qwen3's chat format, 4-bit drafters):

| | Plain | DFlash | Tokens from drafts |
|---|---:|---:|---:|
| MLX (mlx-lm, dflash-mlx) | 52.5 tok/s | 63.4 tok/s (1.21×) | 72% |
| nuvec | 51.6 tok/s | **111.3 tok/s (2.16×)** | 71% |

Both accept the same share of drafts. With fast routing, nuvec's verification of 8 rows costs about one decode step on this Qwen3-8B prompt. With `row_exact`, nuvec decodes the same tokens as plain decoding at plain decoding's speed.

DSpark against DFlash on Qwen3-8B 4-bit, drafts of 7, 4-bit drafters (tok/s):

| Prompt | Plain | `z-lab/Qwen3-8B-DFlash-b16` | `deepseek-ai/dspark_qwen3_8b_block7` |
|---|---:|---:|---:|
| Code (CSV function) | 49.0 | 130.1 (194 of 428 drafts accepted) | **132.6–149.2** (204 of 361) |
| GSM8K "Natalia" | 45.1 | 101.0 (70 of 203) | **199.6–235.7** (88 of 91) |

Draft trees against chains on Qwen3-8B with `z-lab/Qwen3-8B-DFlash-b16`:

| Plain | Chain of 7 | Tree of 7 | Tree of 15 | `Auto` |
|---:|---:|---:|---:|---:|
| 54.0 tok/s | 135.2 | 135.5 | 98.6 | 130.6 |

On an M4 Pro a verification costs one decode step up to 8 rows and 1.7× at 16, so a tree of 15 accepts 24% more tokens than the chain but costs more than it gains. On mixtures of experts extra rows read other experts (8 rows cost four decode steps on Qwen3-Coder-30B-A3B), so `Auto` decodes plainly.

MTP, 4-bit models, 256 tokens after a short story opening, greedy, row-exact; MTP drafts one token per step (`Auto`).

| Model | Plain decoding | MTP speculation | Drafts accepted |
|---|---:|---:|---|
| Qwen3.5-9B (hybrid) | 49.0 tok/s | **66.4 tok/s (1.36×)** | 104 of 152 |
| Qwen3.8-27B (hybrid) | 14.1 tok/s | **23.3 tok/s (1.66×)** | 111 of 144 |
| Qwen3.6-35B-A3B (hybrid MoE) | 92.5 tok/s | 92.7 tok/s (1.00×) | 61 of 93 |

Dense models gain the most: a verified extra token costs little more than one step. In a mixture-of-experts model each row routes to other experts, so a verification costs nearly two steps; the kill switch keeps it at parity.

Against mlx-lm in the same session (`just bench-spec`, 2026-10-26, the same prompt, 256 greedy tokens):

| Model | mlx-lm plain | nuvec plain | nuvec MTP | mlx-lm with a draft model |
|---|---:|---:|---:|---:|
| Qwen3-8B | 52.1 tok/s | 52.5 | — | 51.3 (Qwen3-0.6B drafts) |
| Qwen3.5-9B | 48.8 tok/s | 48.4 | **65.0 (1.34×)** | — |
| Qwen3.8-27B | 15.2 tok/s | 15.1 | **23.6 (1.56×)** | — |
| Qwen3.6-35B-A3B | 90.5 tok/s | 91.0 | 90.5 (0.99×) | — |

mlx-vlm 0.7.1 also drafts with Qwen3.5's MTP head: on Qwen3.5-9B 4-bit with the same prompt (its chat template applied) it decodes 54.5 tok/s against its own 50.5 tok/s (1.08×), accepting 35% of its drafts, and its speculative output differs from its greedy output.

Prompt lookup on Qwen3-8B: 1.53× on a paragraph repeated eight times, 1.0× on natural text.

Agent traces (`crates/nuvec-lm/benches/prompts/traces/`, each turn's prompt the whole history; `bench --trace FILE`), Qwen3-8B 4-bit, greedy, row-exact (`--row-exact`), output identical to plain decoding on every turn:

| Trace | Plain | Prompt lookup | SuffixDecoding |
|---|---:|---:|---:|
| Code rewrite (4 turns) | 52.0 tok/s | 51.0 | 51.1 |
| Tool calls (5 turns) | 50.2 tok/s | 61.3 (1.22×) | **62.6 (1.25×)** |
| File-edit loop (4 turns) | 52.8 tok/s | 80.6 (1.53×) | **90.0 (1.71×)** |

SuffixDecoding drafts from earlier turns as well as the current context, so it accepts more tokens in total on every trace.

## Scope

Speculation runs at batch 1 on the contiguous cache, for every supported model (draft trees on attention-only models): Gated DeltaNet, Mamba-2 and short-convolution layers keep each verified position's state and roll back to the last accepted one, and Mamba-2 verifies with decoding's one-step update, so with `row_exact` the tokens are plain decoding's. The `Batcher` and paged caches do not speculate.
