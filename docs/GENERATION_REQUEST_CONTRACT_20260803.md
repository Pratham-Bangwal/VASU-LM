# Generation Request Contract

Date: 2026-08-03

Status: implemented and qualified on CPU.

## Root cause

Generation validation was previously distributed across Python iteration,
sampling tensor operations, and the cached branch. Invalid temperature values
were silently clamped, invalid top-k/top-p values failed late, negative token
limits silently returned empty output, and an invalid cache implementation
could remain hidden when its branch was not reached. Cached and uncached prompt
boundaries also differed.

## Contract

Before model execution, generation now requires:

- a non-negative integer `max_new_tokens` that is not a boolean;
- boolean `do_sample` and `use_kv_cache` values;
- `dynamic` or `preallocated` as the cache implementation;
- a non-empty encoded prompt within the declared model context;
- when sampling is active, finite positive temperature, `top_k=None` or a
  positive integer, `0 < top_p <= 1`, and a finite positive repetition penalty
  or `None`.

Sampling-only controls remain ignored in greedy mode. This preserves existing
frozen evaluation presets that use zero-valued sentinels with
`do_sample=False`. A zero-token request is valid and executes no model forward,
but all request and cache-selection validation still runs.

## Qualification

The deterministic smoke qualification proves that invalid limits fail before
`model.eval()`, invalid cache selection fails for a zero-token request, zero
tokens cause zero forward calls, all six representative invalid sampling cases
are rejected, and greedy sentinels remain compatible.

Frozen evidence:
`evaluation/fixtures/generation_request_contract_qualification_20260803.json`.

Reproduction command:

```powershell
python scripts\smoke_generation_request_contract.py
```

## Compatibility

Valid generation requests retain their token-selection semantics and return
types. The change affects only invalid or ambiguous inputs. Model architecture,
parameters, state-dict keys, checkpoints, tokenizer assets and token IDs,
datasets, masks, optimizer/scheduler state, training, and exact resume remain
unchanged. Uncached generation and disabled KV caching remain the defaults.

## Non-authorization

Qualification used a synthetic fake model, loaded no checkpoint or dataset,
created no optimizer, and performed no training. It does not change any
VASU-140M readiness or training authorization state.
