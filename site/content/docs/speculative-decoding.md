+++
title = "Speculative decoding"
description = "Speculative decoding in nuvec-lm: pluggable drafters, prompt-lookup drafting and greedy verification in one forward pass — up to 2× faster decode with identical output."
weight = 100
[extra]
group = "Libraries"
+++

Decoding is limited by memory bandwidth: every token reads all the model's weights once. Verifying several tokens in one forward pass reads the weights once too, so it costs little more than generating one. **Speculative decoding** exploits this: a cheap *drafter* proposes the next tokens, the model checks them all at once, and every accepted token saves a full step.

## Using it

```rust
use nuvec_lm::{GenerateOptions, PromptLookup, load, speculative_generate};

let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
let prompt = tokenizer.encode(&code_and_instructions);
let mut generation = speculative_generate(
    &model, &prompt, GenerateOptions::default(), PromptLookup::default(), 4,
)?;
let ids: Vec<u32> = generation.by_ref().collect::<nuvec::Result<_>>()?;
let stats = generation.stats(); // steps, drafted, accepted
```

The output is the model's own greedy output: drafts only decide how many tokens a step produces, never which tokens.

## How it works

Each step runs the last token plus up to `N` draft tokens through the model. The model's argmax after each position is compared with the draft; the longest agreeing prefix is kept, plus the model's own next token, and the KV cache is trimmed past the rejected drafts.

- **When there is nothing to draft,** steps are pipelined exactly like normal generation, so plain text costs about the same as without speculation.
- **After a fully rejected draft,** drafting pauses for 1, 2, 4… up to 16 steps, so a drafter that guesses badly costs little.

## Drafters

Drafters implement one method:

```rust
pub trait Drafter {
    fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32>;
}
```

`PromptLookup` needs no model: it finds the most recent earlier occurrence of the context's last three tokens and proposes what followed. It shines whenever output repeats input — code edits, refactors, quotes, summaries, structured tool calls.

## Results

Qwen3-8B 4-bit on an Apple M4 Pro, 256 tokens, drafts of up to 4 tokens:

| Prompt | Plain decoding | Speculative | Drafts accepted |
|---|---:|---:|---|
| Repetitive text | 53.2 tok/s | **105.7 tok/s (1.98×)** | 198 of 202 |
| Natural text | 53.4 tok/s | 51.4 tok/s (0.96×) | 1 of 16 |
| Code rewrite | 52.8 tok/s | 53.3 tok/s (1.01×) | none proposed |

## Current scope

- Greedy decoding (temperature 0).
- Models whose layers are all attention. Hybrid models (Qwen 3.5 and 3.6) are refused, because the Gated DeltaNet state cannot be rolled back after a rejected draft.
