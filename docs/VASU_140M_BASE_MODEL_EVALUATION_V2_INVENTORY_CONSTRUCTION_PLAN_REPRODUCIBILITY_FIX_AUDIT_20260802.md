# VASU-140M Inventory Construction Plan Reproducibility Fix Audit

Date: 2026-08-02

Status: **author-side corrective validation; construction remains unauthorized.**

## Root Cause

The initially committed plan bound
`evaluation/benchmarks/ultrachat_promotion_v1.json` to the historical CRLF
bytes retained by the main Windows worktree. The repository line-ending policy
checks that text file out with LF in a clean clone, so a clean detached
checkout correctly rejected the stale identity. No benchmark content changed.

## Correction

- The bound benchmark identity now uses its canonical LF byte hash:
  `a92cd70430570b0f94d32747d4b9b54beeb83c4cc24af6d0c348f2da190ce215`.
- Bound `.json`, `.md`, and `.py` artifacts are hashed with CRLF normalized to
  LF. This is restricted to declared text formats; other formats retain raw
  byte identities.
- Regression coverage proves that a historical CRLF checkout validates against
  the canonical text identity, while non-line-ending drift remains rejected.

## Updated Identity

- Superseded plan identity:
  `4bd36c7d7b862e782614347f9c837e3fbaf12c7eaff0d80b7df146c6d71db784`
- Corrected plan identity:
  `3a0b17aa613af4fd1bec4a76e12bbc545505a289719d394dc62e27e8e10b1494`

The original independent decision remains historical evidence for the design;
this corrective record does not relabel author-side validation as an
independent review.

## Validation

- `python -m pytest tests\\test_vasu_140m_base_v2_inventory_plan.py tests\\test_vasu_140m_base_v2_inventory.py tests\\test_vasu_140m_base_v2_scoring.py tests\\test_repository_line_endings.py -q`
  — 81 passed.
- Frozen plan smoke replay passed with the corrected plan identity.
- Ruff passed for the changed Python files.
- `git diff --check` found no whitespace errors; it reported only pre-existing
  CRLF warnings for unrelated documentation files.

## Compatibility and Authorization

This is an evaluation-plan identity repair only. It changes no benchmark
content, model architecture, tokenizer, dataset, mask, checkpoint,
optimizer/scheduler state, training configuration, or existing evaluation
result. It creates no prompt inventory, held-out key, source acquisition,
dataset release, model execution, checkpoint, schedule, authorization record,
or training run. `construction_authorized`, `evaluation_run_authorized`, and
`training_authorized` remain false.
