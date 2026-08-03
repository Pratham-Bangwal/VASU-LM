# KV-Cache In-Place Token History

Date: 2026-08-03

Status: implemented and parity-qualified; KV-cache default unchanged.

## Root cause

The cached generation branch preallocated attention K/V storage but still used
`torch.cat` to rebuild the complete token-history tensor after every generated
token. That copy grows with sequence length and is unnecessary because the
model context limit already provides a fixed upper bound.

## Change

Cached generation now allocates one token-history tensor at the model context
capacity, copies the prompt once, and writes each generated token at the next
logical position. Sampling and repetition penalty receive a view of the exact
populated prefix. The uncached reference branch is unchanged.

Tests verify exact dynamic/preallocated/uncached greedy parity, EOS and context
stopping, complete history lengths for sampling, stable backing storage across
history views, and absence of token-history concatenation in the preallocated
generation path.

## Performance evidence

Five interleaved, fixed-seed, one-thread VASU-60M CPU trials with a 32-token
prompt and 128 generated tokens measured:

- baseline median: 69.5288 tokens/s;
- optimized median: 71.7665 tokens/s (+3.22%);
- baseline mean: 69.4228 tokens/s;
- optimized mean: 70.2901 tokens/s (+1.25%).

CUDA was unavailable. The change improves the opt-in cached path but is not
sufficient evidence to enable KV caching by default.

## Compatibility

The public generation arguments and returned token IDs are unchanged. Model
parameters, state-dict keys, checkpoints, tokenizer assets, prompts, datasets,
masks, training execution, optimizer/scheduler state, and exact resume remain
compatible.

## Non-authorization

The diagnostic used a fixed-seed random model, opened no checkpoint or dataset,
created no optimizer, and performed no training. VASU-140M readiness and
authorization remain unchanged.
