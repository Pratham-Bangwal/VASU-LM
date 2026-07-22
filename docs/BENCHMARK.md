# Evaluation and Benchmarks

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
