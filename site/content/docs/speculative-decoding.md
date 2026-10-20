+++
title = "Speculative decoding"
description = "Speculative decoding in nuvec-lm: pluggable drafters, prompt-lookup drafting and greedy verification in one forward pass — up to 2.2× faster decode with identical output, hybrid models included."
weight = 100
[extra]
group = "Libraries"
+++

Decoding is limited by memory bandwidth: every token reads all the model's weights once. Verifying several tokens in one forward pass reads the weights once too, so it costs much less than generating them one by one. **Speculative decoding** exploits this: a cheap *drafter* proposes the next tokens, the model checks them all at once, and every accepted token saves a full step.

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

Each step runs the last token plus up to `N` draft tokens through the model. The model's argmax after each position is compared with the draft; the longest agreeing prefix is kept, plus the model's own next token, and the KV cache is trimmed past the rejected drafts. Hybrid models keep their Gated DeltaNet state after every position while a draft is verified, and return to the last accepted one.

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

Apple M4 Pro, 4-bit models, 256 tokens, drafts of up to 4 tokens. Repetitive text is a paragraph repeated eight times; natural text is a short-story request.

| Model | Prompt | Plain decoding | Speculative | Drafts accepted |
|---|---|---:|---:|---|
| Qwen3-8B | repetitive | 53.2 tok/s | **114.8 tok/s (2.16×)** | 198 of 202 |
| Qwen3-8B | natural | 53.9 tok/s | 55.0 tok/s (1.02×) | 13 of 20 |
| Qwen3.5-9B (hybrid) | repetitive | 48.6 tok/s | 55.4 tok/s (1.14×) | 53 of 72 |
| Qwen3.5-9B (hybrid) | natural | 48.0 tok/s | 48.3 tok/s (1.01×) | 14 of 48 |
| Qwen3.6-35B-A3B (hybrid MoE) | repetitive | 91.8 tok/s | 94.9 tok/s (1.03×) | 33 of 52 |
| Qwen3.6-35B-A3B (hybrid MoE) | natural | 91.8 tok/s | 90.2 tok/s (0.98×) | 9 of 28 |

Gains depend on how often the output repeats the context: prompt lookup finds fewer drafts in the hybrid models' output, and verifying several tokens of a mixture-of-experts model reads more experts.

## Scope

Greedy decoding (temperature 0), for every supported model, hybrids included.
