# VASU-140M Repository Policy Guard

Date: 2026-08-03

Status: implemented; current-phase CI enforcement, non-authorizing.

## Why this is needed

The VASU-140M readiness auditor records the current gate state, but a passing
test suite alone does not make the safety boundary obvious in CI. The policy
guard turns the frozen readiness conclusion into an explicit regression step.

## Enforced current phase

The guard requires the exact frozen 2026-08-03 readiness report, with CUDA/AMP
complete and the other five gates blocked. It rejects the presence of:

- a VASU-140M base-pretraining processed release;
- a VASU-140M base-pretraining manifest;
- any `checkpoints/vasu_140m` artifact;
- an actual VASU-140M base-pretraining experiment plan; or
- an actual VASU-140M base-pretraining authorization record.

A legitimate future phase transition must first create new reviewed evidence,
version the readiness contract, and deliberately update this guard. Merely
placing a file at a protected path cannot advance readiness.

## Compatibility

The guard and its CI step are read-only. They change no model architecture,
tokenizer, dataset, mask, schedule, checkpoint schema, optimizer/scheduler
state, trainer, evaluation payload, or exact-resume behavior. Existing VASU-31M
and VASU-60M artifacts and the published VASU-140M instruction seed are outside
the prohibited base-pretraining paths and remain compatible.

## Non-authorization

Passing the guard means only that the repository remains in its reviewed
blocked phase. It does not authorize a source, data release, checkpoint,
experiment plan, optimizer update, authorization record, or training run.
