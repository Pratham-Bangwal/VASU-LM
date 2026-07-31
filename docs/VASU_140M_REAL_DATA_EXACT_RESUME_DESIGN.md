# VASU-140M Real-Data Exact-Resume Qualification Design

Status: **design-only; no real-data workload or optimizer update is authorized.**

## Purpose

Qualify the exact checkpoint/resume path on the future accepted VASU-140M
base-pretraining release before a full experiment can be proposed. The existing
synthetic CPU qualification proves trainer mechanics but not full-context data,
loss masks, source schedules, CUDA/AMP state, or real release identities.

## Preconditions

Execution is ineligible until all of the following are independently accepted:

- a published, immutable `vasu_140m/base_pretraining/v1` release;
- exact train split token, mask, logical-record, source, and manifest hashes;
- a deterministic no-replacement source schedule and schedule SHA-256;
- the VASU-140M evaluation development contract;
- a clean reviewed commit containing the qualification implementation; and
- a separate one-shot human execution approval.

Held-out evaluation records are never inputs to this qualification.

## Frozen bounded workload

The eventual execution package must bind these values before review:

| Field | Requirement |
| --- | --- |
| Family | `vasu_140m_v1` only |
| Device | Intended CUDA device and accepted BF16/FP16 mode |
| Records | Small deterministic prefix spanning both sources and at least one source boundary |
| Record width | 513 stored tokens/masks; 512 input/target positions |
| Batch size | 1 unless later CUDA evidence justifies another value |
| Accumulation | At least 2 microbatches |
| Updates | Exactly 2 completed optimizer updates |
| Interruption | After a nonzero partial accumulation and separately at a source boundary |
| Optimizer/scheduler | Exact prospective implementation, parameters, and identities |
| Outputs | System-temporary qualification directory only |

No checkpoint may be written under `checkpoints/`. No retry occurs after a
failed immutable execution without a new reason and output identity.

## Equivalence requirements

Independent processes run an uninterrupted control and interrupted/resumed
branch. They must match bit-for-bit where the backend guarantees determinism,
and otherwise fail closed until the nondeterminism is explained and bounded;
tolerance-based acceptance cannot be introduced silently.

Required exact comparisons include model tensors, master/optimizer tensors,
scheduler, AMP scaler, sampler and source-schedule position, partial gradients,
microbatch/update counters, validation state, consumed logical record IDs,
source transitions, loss-mask target counts, and Python/NumPy/PyTorch CPU/CUDA
RNG states. Final losses and gradient norms are recorded but do not replace
state equality.

## Checkpoint integrity

The interruption checkpoint must atomically contain:

- family, model-config, tokenizer, release, manifest, source, token, mask,
  schedule, evaluation-development, and repository identities;
- model, optimizer, scheduler, scaler, gradients, and RNG states;
- exact epoch/source/record/batch/microbatch/update positions;
- checkpoint byte size and SHA-256 sidecar; and
- schema version and creation phase.

Reload must reject before state mutation if any identity, size, hash, dtype,
shape, record width, target-mask alignment, schedule position, optimizer
backend, or accumulation parameter changes.

## Adversarial qualification

Tests must cover truncated/corrupted checkpoints and sidecars, family/tokenizer/
release/schedule/config mismatch, swapped token or mask files, changed source
order, changed accumulation, missing partial gradients, invalid scaler state,
disk exhaustion, injected atomic-rename failure, stale temporary directories,
Windows open-handle behavior, and mutation after prevalidation but before load.

Failure preserves diagnostic evidence in an isolated temporary path and never
replaces a valid checkpoint. Successful completion removes temporary model
checkpoints after recording their hashes and verifies disk cleanup.

## Result contract

The immutable report binds implementation/test/commit identities, hardware and
software environment, every input hash, workload settings, control/resume state
digests, checkpoint hash/size, all adversarial results, cleanup, and
`training_authorized=false`. Passing advances only the real-data resume gate.

## Compatibility and non-authorization

The design requires additive qualification tooling and does not change the
trainer, persistent checkpoint schema, model architecture, tokenizer, or data
contract. Any implementation change discovered as necessary requires its own
compatibility review. This design authorizes no data release, CUDA workload,
optimizer update, production checkpoint, configuration, or training run.
