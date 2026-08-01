# VASU-140M Clean-Checkout Reproducibility Fix Audit

Date: 2026-08-02

Status: **author-side corrective validation; non-authorizing.**

## Findings and Correction

A detached clean checkout exposed two test-environment assumptions:

1. The resume-checkpoint rejection test used the current working directory as
   its non-system-temporary path. A detached worktree created beneath the
   system temporary directory made that assumption false.
2. The synthetic base-plan test hash-bound the actual accepted CUDA/AMP and
   exact-resume qualification results while that evidence was hidden by the
   generic `evaluation/results/` ignore rule. A clean checkout therefore
   lacked required, reviewed inputs.

The checkpoint test now creates a unique sibling of the system temporary
directory, which is provably outside the allowed qualification root. The exact
already-reviewed CUDA/AMP and exact-resume execution results are retained as
versioned, hash-bound exceptions under `evaluation/results/`; they are not
regenerated, modified, or treated as training authorization.

## Impact

This repair changes no model, tokenizer, dataset, mask, schedule, optimizer,
checkpoint, release, or evaluation score. It makes the preflight test package
reproducible from a clean clone and preserves the original evidence byte
identities. Training remains unauthorized.
