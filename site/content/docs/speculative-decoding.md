+++
title = "Speculative decoding"
description = "Speculative decoding in nuvec-lm: the model's own multi-token-prediction head, prompt lookup and SuffixDecoding drafts, verified in one forward pass with output identical to plain decoding — up to 1.66× faster on natural text."
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

Greedy output is the model's own greedy output, **bit for bit**: drafts only decide how many tokens a step produces, never which tokens. A verification computes each row exactly as serial decoding would (row-exact products, `nuvec::with_row_exact`, and the attention planner's serial rule), and the verify width is capped where that holds.

## Drafters

| Drafter | Needs | Drafts well on |
|---|---|---|
| `MtpDrafter` | the model's multi-token-prediction head (Qwen 3.5, 3.6, 3.8) | any text: the head predicts the model's next-but-one token |
| `PromptLookup` | nothing | output that repeats its input: code edits, quotes, summaries, tool calls |
| `SuffixDecoding` | nothing; remembers earlier requests | agent loops that repeat earlier prompts and outputs |

- **MTP.** The head (one decoder layer) takes the next token's embedding and the model's hidden state, and predicts the token after. Its first draft token comes from the last verified position; further ones reuse its own hidden state and are accepted less often. After each verification its cache is re-fed with the model's real hidden states, so it never drifts. `Model::attach_mtp` quantizes the head like the body (an 8-bit router stays 8-bit).
- **SuffixDecoding** (arXiv 2411.04975) matches the longest suffix of the context in the prompt and a bounded store of earlier sequences, and drafts the most frequent continuation. Pass `&mut SuffixDecoding` and call `remember` after each request to keep its store across requests.
- Write your own by implementing `Drafter::draft` (and `observe`, to follow the context; `needs_hidden`, to receive the model's hidden states).

## Policies

- **`DraftPolicy::Auto`** (default): each drafter's measured draft length for the model, plus a kill switch. When the accepted tokens per verification over the last 16 verifications fall below the drafter's break-even, the next 64 tokens are decoded plainly (pipelined), then drafting resumes.
- **`DraftPolicy::Adaptive(n)`**: up to `n` tokens, pausing 1, 2, 4, … 16 steps after a fully rejected draft (prompt lookup's classic back-off).
- **`DraftPolicy::Fixed(n)`**: up to `n` tokens every step, for measurements.

## Sampling

With a temperature, verifications use the lossless rule: a draft token is accepted with probability `min(1, p / q)` against the sampler's filtered distribution (temperature, top-p, min-p, top-k), and a rejected one is replaced by a sample from the residual. Tokens are distributed exactly as plain sampling's; a seed repeats a run.

The rule is also available on its own, on the device, for runtimes that bring their own drafters: `nuvec::speculative::accept_greedy` and `accept_sampled` return `[accepted, next, eos_hit]` in one `[3]` array, so a verification costs one small read. `nuvec::fast::gated_delta_steps_with` keeps every step's Gated DeltaNet state rounded to a storage dtype (bfloat16 caches), for exact rollback.

## Results

Apple M4 Pro (48 GB), 4-bit models, 256 tokens after a short story opening, greedy; MTP drafts one token per step (`Auto`).

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

Agent traces (`crates/nuvec-lm/benches/prompts/traces/`, each turn's prompt the whole history; `bench --trace FILE`), Qwen3-8B 4-bit, greedy, output identical to plain decoding on every turn:

| Trace | Plain | Prompt lookup | SuffixDecoding |
|---|---:|---:|---:|
| Code rewrite (4 turns) | 52.0 tok/s | 51.0 | 51.1 |
| Tool calls (5 turns) | 50.2 tok/s | 61.3 (1.22×) | **62.6 (1.25×)** |
| File-edit loop (4 turns) | 52.8 tok/s | 80.6 (1.53×) | **90.0 (1.71×)** |

SuffixDecoding drafts from earlier turns as well as the current context, so it accepts more tokens in total on every trace.

## Scope

Speculation runs at batch 1 on the contiguous cache, for every supported model, hybrids included. The `Batcher` and paged caches do not speculate.
