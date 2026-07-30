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

The initial recommendation is documented in
[`VASU_140M_FAMILY_PROPOSAL.md`](VASU_140M_FAMILY_PROPOSAL.md). It proposes an
isolated 140M family. The additive configuration, immutable family identity,
exact parameter contract, and meta-device construction preflight are complete;
training remains unauthorized. See
[`VASU_140M_IMPLEMENTATION_READINESS.md`](VASU_140M_IMPLEMENTATION_READINESS.md).

1. **Complete:** use evaluation and performance evidence to select an isolated
   capacity/context family rather than changing VASU-60M in place.
2. **Complete:** freeze its dimensions, parameter budget, context length,
   tokenizer boundary, memory estimate, migration plan, identity fingerprint,
   and remaining gates.
3. **Complete:** bounded FP32 CPU forward/backward and dynamic/preallocated
   cache-parity qualification passed. See
   [`VASU_140M_CPU_QUALIFICATION_20260730.md`](VASU_140M_CPU_QUALIFICATION_20260730.md).
4. **CUDA-gated:** measure peak memory, throughput, thermals, checkpoint I/O,
   serialization, and exact resume before any experiment plan.
5. **Data-gated:** independently specify and review isolated 513-token releases;
   do not repack or generate them as part of architecture readiness.
6. **Complete:** additive model-only checkpoint round-trip and explicit
   wrong-family rejection passed without changing legacy checkpoint
   containers. See
   [`VASU_140M_CHECKPOINT_QUALIFICATION_20260730.md`](VASU_140M_CHECKPOINT_QUALIFICATION_20260730.md).
7. **Complete:** corrected v2 exact resume matched model, AdamW, scheduler,
   sampler, scaler, partial gradients, sample order, and RNG state. The failed
   v1 evidence and loader-RNG root cause are preserved in
   [`VASU_140M_EXACT_RESUME_QUALIFICATION_20260730.md`](VASU_140M_EXACT_RESUME_QUALIFICATION_20260730.md).
8. **Next non-CUDA gate:** independently specify and review the isolated
   513-token data/mask contract without generating a release.

Exit criterion: the implementation-readiness contract is complete. Promotion
to training planning requires every remaining gate and a separate decision;
no training is implied.

## Phase 5 — Open-source maturity

1. Add contributor guidance, reproducibility checklist, and release notes.
2. Keep CI green across supported Python versions.
3. Maintain status, architecture, dataset lineage, and experiment decisions as
   changes land.
