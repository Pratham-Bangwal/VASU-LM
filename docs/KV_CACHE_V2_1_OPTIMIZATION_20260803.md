# Preallocated KV Cache v2.1 Optimization

Date: 2026-08-03

Status: implemented and parity-qualified; default adoption rejected.

## Root cause

The preallocated cache previously scanned every other layer length on every
update and constructed old key/value views in attention only to determine the
cached length. These operations scale poorly with layer count and add Python
overhead to every generated token.

## Change

The cache now advances through an ordered layer-update state machine. One
logical token range becomes committed only after every model layer writes the
same range. Out-of-order updates and attempts to observe a partially completed
forward fail explicitly. Reset recovers a partial cycle without reallocating
storage. Attention reads a scalar layer length instead of creating unused old
cache views.

## Evidence

Focused KV-cache and VASU-140M CPU-qualification tests passed with exact logits
and greedy-generation parity. A five-trial, fixed-seed, one-thread synthetic
VASU-60M CPU diagnostic measured:

- previous median: 58.1873 tokens/s;
- v2.1 median: 58.1999 tokens/s (+0.02%);
- previous mean: 57.3396 tokens/s;
- v2.1 mean: 58.0528 tokens/s (+1.24%).

CUDA was unavailable in the profiling process. The measured CPU improvement is
well below the existing 10% promotion threshold, so uncached generation remains
the default and both cache implementations remain opt-in. This is intentionally
recorded as a negative adoption result.

## Compatibility

The cache is ephemeral inference state. Model parameters, state-dict keys,
checkpoint containers, tokenizer assets, prompts, datasets, masks, training
forwards, optimizer/scheduler state, and exact-resume behavior are unchanged.
Existing checkpoints remain directly loadable.

## Non-authorization

The diagnostic used a fixed-seed randomly initialized model, opened no dataset
or checkpoint, created no optimizer or persistent model artifact, and performed
no training. It does not change any VASU-140M readiness or authorization gate.
