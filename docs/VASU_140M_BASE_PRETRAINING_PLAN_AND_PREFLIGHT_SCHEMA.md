# VASU-140M Base-Pretraining Plan and Final-Preflight Schema

Status: schema implementation only; no actual experiment plan, configuration,
schedule, authorization envelope, or training run exists.

## Purpose and ordering

The readiness protocol permits an immutable scientific plan only after CUDA,
base-data release, frozen evaluation, and real-data exact-resume gates pass.
Those gates are not all complete. This package therefore implements only the
future plan's fail-closed schema and file validator. It deliberately does not
instantiate a plan under `configs/`, select hyperparameters, or claim gate
acceptance.

The generic `vasu/training/config_schema.py` checks only a few fields, and the
older VASU-60M capability authorization path relies on a reviewed Boolean edit.
Those mechanisms remain valid for their existing lineages but are insufficient
for the first VASU-140M base checkpoint, whose initialization, corpus, schedule,
evaluation, resume evidence, budgets, and output paths must all be immutable.

## Required plan identity

The additive `vasu_140m_base_pretraining_plan_v2` schema binds:

- the exact `vasu_140m_v1` family, family/config fingerprints, tokenizer, and
  clean reviewed repository commit;
- independently accepted CUDA/AMP, base-data release, evaluation-v2, and
  real-data exact-resume artifacts and decision files, with every evidence
  path globally unique;
- one primary question, prediction, explicit falsifier, and honest
  first-lineage comparison against the same randomly initialized model before
  its first optimizer update;
- fresh random initialization, with no parent checkpoint and a bound
  initialization implementation;
- release, distinct schedule, per-source manifest/token/mask/lineage hashes,
  no-replacement policy, 513-token storage, and 512-position training view;
- exact record, microbatch, accumulation, optimizer-update, input-position,
  supervised-target, scheduler, and warmup accounting;
- AdamW implementation and all optimizer/scheduler parameters;
- both development and held-out coverage for likelihood, factuality,
  arithmetic, repetition, robustness, and manual review;
- machine-readable promotion and rejection rules without a single aggregate
  capability score;
- mandatory final validation/checkpoint, bounded validation/checkpoint
  intervals, and an exact runtime envelope whose wall-time, optimizer-update,
  and input-position caps reconcile with the training budget;
- a minimum 10 GB disk reserve, required ordered thermal limits capped at 88 C
  for abort, checkpoint integrity, validation, non-finite, OOM, and
  resume-divergence hard stops;
- distinct, initially absent checkpoint, log, and evaluation result paths
  under their dedicated VASU-140M repository roots;
- pending independent review and `training_authorized=false`; and
- one canonical plan SHA-256 over every field above.

The selected release and evaluation suite must be byte-identical to the
corresponding accepted gate artifacts. The schedule must be a separate bound
artifact. The schema mathematically rejects a plan when source totals, schedule records,
processed records, microbatches, accumulation groups, optimizer updates,
scheduler totals, processed positions, or supervised-target counts disagree.

## Fresh-lineage requirement

No VASU-140M checkpoint exists, so the first base-pretraining run must use a
fresh random initialization whose seed and implementation hash are frozen.
VASU-31M/60M checkpoints, Candidate A, converted weights, partial loads, and
shape-tolerant loading are invalid. Its scientific control is the exact same
initialized VASU-140M model evaluated before any optimizer update. That is a
within-lineage calibration reference, not a trained parent checkpoint.

## Why v2 supersedes v1

The pre-commit v1 review correctly evaluated the earlier bytes, but an
author-side audit then found gaps that could admit a scientifically weak or
operationally unsafe future plan: selected artifacts were not reconciled back
to their accepted gates, the first-lineage comparison was implicit, runtime
caps were absent, source manifests were not individually bound, intervals
could exceed the full run, final validation/checkpoint were optional, a
one-byte disk reserve was legal, thermal limits were unbounded, and outputs
could use arbitrary repository paths. Version 2 fails closed on all of those
cases. The v1 decision is historical evidence only and cannot accept v2.

## Held-out policy

Development dimensions must be available before training for baseline and
diagnostic measurements. Held-out inventories remain sealed before training
and open only at the plan's declared post-training decision point. A plan that
marks held-out evaluation as opened before training fails validation.

## Final preflight design

After an actual plan is independently accepted and committed, a separately
reviewed final-preflight implementation must:

1. require a clean worktree and exact reviewed runtime commit;
2. validate every bound artifact byte and accepted decision identity;
3. re-run release, schedule, tokenizer, family, evaluation, CUDA, and
   real-data-resume validators;
4. recompute all record/microbatch/update/token accounting;
5. prove all output paths are absent and same-volume atomic checkpointing is
   available;
6. verify conservative free disk, active thermal telemetry, checkpoint
   retention, sidecars, explicit resume, and crash recovery;
7. emit a deterministic eligibility report with `training_started=false` and
   no optimizer construction; and
8. require a new independent identity review of the clean-commit report.

A passed preflight is eligibility evidence, not authorization.

## Exact authorization design

VASU-140M base training should not mutate the reviewed plan or rely on a lone
Boolean. A future detached, canonical envelope must bind the accepted plan and
preflight identities, exact runtime commit, command, device, precision,
release, schedule, evaluation package, output paths, budgets, safety limits,
expiration, and one-use authorization ID. It must record an explicit human
approver and a distinct independent plan reviewer.

The future runner must validate and exclusively consume that envelope before
model/optimizer allocation. Reuse, expiry, identity drift, dirty repository,
existing output, or changed hyperparameters must fail before any training
state is created. A consumed receipt is written only after launch ownership is
acquired. The user must separately authorize that exact envelope; earlier
general permission does not authorize training.

## Compatibility and non-authorization

This implementation is additive and does not change existing VASU-31M/60M
configs, authorization records, launchers, checkpoints, datasets, masks,
schedules, or the tokenizer. It creates no VASU-140M configuration or plan and
does not make the current generic trainer launchable for VASU-140M.

No source acquisition, release construction, model initialization, CUDA
training workload, optimizer, checkpoint, authorization record, or training
run is authorized.
