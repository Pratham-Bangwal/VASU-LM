# Evaluation and Benchmarks

The additive VASU internal capability suite is documented in
[`EVALUATION.md`](EVALUATION.md). It keeps objective structural/factual
results, heuristic concept/repetition signals, and incomplete human-review
fields separate rather than presenting an opaque quality score.

## Current evaluation

VASU-31M instruction checkpoints are compared on fixed prompts using manual scores for relevance, factuality, instruction following, fluency, and repetition control. The current `ultrachat_fineweb` baseline is **2.225 / 5**.

VASU-60M base milestones use raw autoregressive continuations without chat formatting. Evaluated preserved milestones include steps 5,000, 10,060, and 54,060. These reports track qualitative changes in grammar, repetition, topic retention, coherence, and factuality.

The current VASU-60M milestone is step 54,060. Step 100,000 has not been completed or evaluated.

## Current quantitative signals

- training loss;
- held-out validation loss;
- peak CUDA memory;
- GPU temperature;
- fixed-prompt manual evaluation for instruction checkpoints.

Validation in block training uses a small fixed held-out sample and is not a comprehensive benchmark.

## Planned expansion

- larger fixed prompt sets;
- category-level factuality and reasoning checks;
- repeatable perplexity evaluation over a broader held-out set;
- inference speed and tokens-per-second measurement;
- established small-model benchmarks where tokenizer and harness integration are verified;
- direct VASU-60M versus VASU-31M comparison after VASU-60M instruction tuning.

No broad benchmark score should be reported until the harness and dataset split are documented and reproducible.

## Bounded KV-cache benchmark

`scripts/profiling/profile_vasu.py` performs synchronized synthetic CUDA
measurements for VASU-31M and VASU-60M. It compares uncached, dynamic, and
preallocated-cache greedy generation with exact token-parity checks. On the
RTX 4050 at 64 and 128 generated tokens, the preallocated implementation did
not meet the 10% VASU-60M throughput promotion threshold; it remains opt-in.

## AdamW backend benchmark

The bounded VASU-60M AMP benchmark in
`scripts/profiling/profile_training_backends.py` found fused AdamW materially
faster than standard AdamW on the RTX 4050: roughly 14.1 ms versus 31.0 ms
median optimizer time in two short runs. Fused AdamW remains an explicit
CUDA-only `TrainConfig.optimizer_backend="fused"` option; the default remains
the historical standard backend pending longer-run and exact-resume validation.

## Pipeline and checkpoint audit

The bounded real-data tool `scripts/profiling/profile_data_pipeline.py` keeps
its model state ephemeral and reads existing memmap datasets only. On the RTX
4050, VASU-60M batch-size-2, sequence-length-256 steps were compute-bound:
FineWeb and factual-mixture loader wait stayed below 0.6% of median step time.
Windows workers improved isolated loader throughput, but not enough to justify
changing the safe worker-zero default.

`scripts/profiling/profile_checkpoint_io.py` writes only under
`tmp/profiling/checkpoints/` and deletes its artifacts by default. A VASU-60M
standard AdamW optimizer-boundary checkpoint measured about 700 MiB; an exact
mid-accumulation checkpoint measured about 934 MiB because it must preserve
roughly 233 MiB of accumulated gradients. This is expected exact-resume cost,
not redundant metadata.
