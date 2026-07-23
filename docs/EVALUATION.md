# VASU Internal Capability Evaluation

`evaluation.framework` is a reproducible internal evaluation system. It is not
MMLU, GSM8K, TruthfulQA, or any other external benchmark.

## Metric separation

- Objective tasks report exact numeric, exact-answer, JSON, count, or other
  structural pass/fail results.
- Heuristic concept checks report required/prohibited concept and repetition
  signals separately; they are not factual-quality scores.
- Human-review tasks deliberately remain incomplete until a reviewer supplies
  a decision. Blank human review blocks promotion.

The versioned `evaluation/suites/vasu_capability_v1.json` fixture contains
small source-recorded stable facts, deterministic arithmetic, structural
formatting, concept, uncertainty, and general-language tasks.

## Usage

```powershell
python -m evaluation.framework.runner `
  --suite evaluation/suites/vasu_capability_v1.json `
  --checkpoints vasu_60m_alpaca_masked_v3_from_200k `
  --mode greedy `
  --output-dir evaluation/results/capability_v1
```

Greedy mode is deterministic. Sampled mode uses an explicit fixed seed list.
Runs record suite/checkpoint/tokenizer hashes, prompt formatter, model config,
generation settings, device, PyTorch version, timestamp, and command options.
Interrupted runs use `--resume`; incompatible manifests are rejected and
completed checkpoint/task/seed combinations are not regenerated.

## Promotion use

Objective accuracy, repetition signals, runtime, and human review are never
collapsed into one quality score. Parent-versus-candidate comparisons require
an explicit `--promotion-type`: `continued_pretraining`, `instruction`, or
`conversation`. The suite names each gate's categories explicitly.

For every category, improvement is `candidate_accuracy - parent_accuracy` and
regression is `parent_accuracy - candidate_accuracy`. Threshold comparisons are
inclusive: an improvement must be `>=` its minimum and a regression must be
`<=` its maximum. Missing objective categories block promotion rather than
being averaged away. Required human-review tasks need a structured
`human_review` result with `status` (`approved` or `rejected`) and a nonblank
reviewer; missing, malformed, or rejected review blocks promotion.

Pass completed reviews through `--human-review-results review.json`, where
`review.json` maps suite human-task IDs to objects such as
`{"status": "approved", "reviewer": "reviewer-name", "notes": "..."}`.
Its SHA-256 is part of the resumable run manifest.
