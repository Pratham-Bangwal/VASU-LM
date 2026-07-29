# Phase 3 CPU Data-Loader Measurement — 2026-07-30

Status: completed read-only measurement. This record neither authorizes
training nor changes any runtime default.

## Question

For the immutable `fineweb_100k` memmap source on the available Windows
CPU-only environment, do DataLoader workers or `pin_memory` improve loader
throughput enough to justify a configuration change?

## Fixed contract

- Dataset: `data/processed/pretrain/fineweb_100k.bin`, SHA-256
  `b0335f8d4bc48a6d6b3241b874a7fc95af6d59f6ac325164aeb23c75eb4110f9`.
- Dataset shape: batch size 2, sequence length 256.
- Protocol: eight warm-up batches, 40 measured iterations, three independent
  replicates per variant, no model step, no optimizer update, no dataset write.
- Environment: Windows, CPU, PyTorch `2.8.0+cpu`, CUDA unavailable.
- Baseline: `workers=0`, `pin_memory=false`.
- Treatments: `workers=0`, `pin_memory=true`; and `workers=2`,
  `prefetch_factor=2`, `pin_memory=false`.

## Immutable evidence

All raw reports, aggregates, contracts, and derived comparisons are retained in
`evaluation/results/performance_phase3_cpu_loader_20260730/`.

| Artifact | SHA-256 |
|---|---|
| `workers0_no_pin_aggregate.json` | `8cc58eba151cb6a7f6a7bddd271ede9a015c41db80c985ad01fcfec866232458` |
| `workers0_pin_aggregate.json` | `e31489136e6b1b053dc312108bd88b046249cf362015049155c08cb1d3774386` |
| `workers2_no_pin_aggregate.json` | `fd15842e3884987001f5a52c4b3dcf866ea0744b1fd86ffc023b04fbd314f936` |
| `workers0_pin_replicated_comparison.json` | `e806527210e9844462041f381d60c1f5666635ae278583299a5686034e7690d9` |
| `workers2_no_pin_replicated_comparison.json` | `20696168b3ba5120fa77cf89f3d30161a2f2925e263fe52ce81c69f5e9a19015` |

## Results

| Variant | Median loader throughput | Delta from baseline |
|---|---:|---:|
| `workers=0`, no pinning | 12,711,994 tokens/s | reference |
| `workers=0`, pinning requested | 12,600,492 tokens/s | -0.88% |
| `workers=2`, no pinning | 1,766,751 tokens/s | -86.10% |

The no-worker baseline and requested-pinning variant had identical median
loader and transfer times. PyTorch emitted the expected warning that pinned
memory is unavailable without an accelerator; the small throughput difference
is therefore not evidence of a pinning effect. Two workers made median loader
time 6.35 times larger and transfer time 9.47 times larger.

## Decision

Keep `workers=0` as the supported CPU configuration for this workload. Do not
enable pinning based on CPU-only data, and reject `workers=2` for this local
memmap benchmark. No loader implementation or default is changed because this
evidence does not establish a cross-hardware configuration policy.

The KV-cache adoption decision is deferred. Existing CUDA evidence records a
memory reduction without throughput improvement, while this environment has no
CUDA device. Cache parity regression coverage passed, but CPU-only timing is
not a substitute for the required matched CUDA benchmark.

## Compatibility

No model, checkpoint, tokenizer, dataset, mask, schedule, optimizer, or
exact-resume artifact changed. No training or authorization action occurred.
