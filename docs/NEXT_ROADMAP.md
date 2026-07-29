# VASU Next Roadmap

Status: planning only. This roadmap does not authorize data generation,
training configuration, checkpoint resume, or model training.

## Phase 1 — Candidate E decision gate

1. Obtain an independent decision on
   `CANDIDATE_E_INDEPENDENT_REVIEW_DECISION.md`.
2. If accepted, implement and review the new logical-data generator against
   the frozen ID/range/template specification.
3. Run deterministic regeneration, split-isolation, tokenizer-boundary,
   mask-alignment, and replay-accounting checks.
4. Require a separate immutable-release review before creating a production
   Candidate E release.

Exit criterion: an approved data-review record, not a training authorization.

## Phase 2 — Evaluation v3

1. Integrate paired bootstrap intervals into arithmetic treatment/control
   reports.
2. Add versioned output taxonomy: exact, incorrect, malformed, prompt leakage,
   truncation, and unanswered.
3. Freeze cross-domain retention and repetition snapshots with hashes before
   any future experiment. A versioned, non-overwriting snapshot contract now
   binds explicitly selected scalar metrics to source-report hashes and a
   declared evaluation identity.
4. Publish a human-readable dashboard summary derived from the read-only JSON
   dashboard. Compatible frozen snapshots can now be rendered as a
   deterministic text summary; it remains descriptive and cannot select or
   promote a checkpoint.

Exit criterion: one reproducible report can compare any two frozen evaluation
snapshots without checkpoint selection or promotion logic.

The read-only paired arithmetic comparison utility is:

```powershell
python evaluation/compare_verified_arithmetic_runs.py `
  --baseline-dir <completed-baseline-run> `
  --candidate-dir <completed-candidate-run>
```

It rejects mismatched evaluator, split, manifest, tokenizer, generation, and
per-example identity before calculating a deterministic bootstrap interval.

Freeze and compare a cross-domain snapshot only after the evaluation suite,
split, parser, and generation identity have been declared:

```powershell
python evaluation/freeze_evaluation_snapshot.py `
  --label <checkpoint-label> `
  --source <name>=<versioned-result.json> `
  --metric <metric-name>=<name>:<dot.path.to.numeric.metric> `
  --identity suite=<suite-id> --identity split=<split-id> `
  --identity generation=<generation-id> --identity parser=<parser-id> `
  --output <new-snapshot.json>

python evaluation/compare_frozen_evaluation_snapshots.py `
  --baseline <baseline-snapshot.json> `
  --candidate <candidate-snapshot.json>
```

## Phase 3 — Measured performance work

The runtime-benchmarking contract is now available for all existing raw
profilers. It freezes selected measurements with workload, environment, and
raw-report hashes; comparisons fail closed across incompatible workloads or
hardware.

1. Benchmark preallocated KV cache versus uncached and dynamic-cache decoding
   on the existing parity prompts. This remains CUDA-gated; do not infer a
   throughput decision from CPU-only evidence.
2. Profile data-loader, packing, host-to-device transfer, and model-step time
   separately using no-update benchmarks. The CPU FineWeb loader subtask is
   complete: `workers=0` was retained, `workers=2` was rejected locally, and
   requested pinning showed no meaningful CPU-only effect. See
   [`PERFORMANCE_PHASE3_CPU_LOADER_20260730.md`](PERFORMANCE_PHASE3_CPU_LOADER_20260730.md).
3. Apply only improvements that preserve outputs, checkpoint keys, tokenizer,
   and schedule identity; re-run parity and regression tests after each.

Exit criterion: a measured improvement or a documented rejection with evidence.

## Phase 4 — Next model-family proposal

1. Use evaluation and performance evidence to decide whether the current 60M
   architecture should continue or a new family is justified.
2. If a new family is proposed, specify parameter budget, context length,
   tokenizer compatibility, initialization, memory estimate, migration plan,
   and fresh smoke/resume/evaluation gates.
3. Do not alter VASU-60M checkpoint-facing architecture in place.

Exit criterion: a reviewed architecture proposal; no training is implied.

## Phase 5 — Open-source maturity

1. Add contributor guidance, reproducibility checklist, and release notes.
2. Keep CI green across supported Python versions.
3. Maintain status, architecture, dataset lineage, and experiment decisions as
   changes land.
