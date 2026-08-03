# KV-Cache Long-Context Qualification

Date: 2026-08-03

Status: passed on CPU; KV-cache default unchanged.

## Why this qualification was needed

The preallocated cache and token-history optimizations had short bounded parity
coverage, but no single frozen qualification exercised exact context capacity,
backing-storage reuse, and malformed tensor ranks together. The audit also
found that rank validation read shape index 3 before confirming a four-rank
tensor, allowing an incidental `IndexError` instead of the public fail-closed
`ValueError` contract.

## Change and evidence

Rank is now checked before dimensional fields. The deterministic qualification
uses a fixed-seed two-layer model with a 64-token context, a one-token prompt,
and a generation request larger than the remaining context. Uncached, dynamic,
and preallocated modes produced the same 63 token IDs and stopped exactly at
capacity. Direct cache checks reached all 64 positions, rejected overflow,
reused the same backing allocation after reset, and rejected malformed ranks
with `ValueError`.

Frozen evidence:
`evaluation/fixtures/kv_cache_long_context_qualification_20260803.json`.

Reproduction command:

```powershell
python scripts\smoke_kv_cache_long_context_qualification.py
```

## Compatibility

The cache API and generation API are unchanged. Model parameters, state-dict
keys, checkpoints, tokenizer assets, prompt formats, datasets, masks, optimizer
and scheduler state, training behavior, and exact resume remain compatible.

## Decision boundary

This is CPU synthetic correctness evidence. It does not establish CUDA
throughput, justify default activation, open a checkpoint or dataset, create an
optimizer, or authorize training. Uncached generation remains the reference
and default path.
