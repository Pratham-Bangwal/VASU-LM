# Candidate E Pre-release Acceptance Checklist

Status: review required; non-authorizing; no Candidate E data release exists.

## Purpose

This checklist is the required independent review boundary between Candidate E
format-preparation code and any immutable dataset construction. It does not
authorize data generation, a mixture, a training configuration, or training.

## Implemented, tested components

- `vasu/data/arithmetic_steps.py` deterministically derives bounded symbolic
  states exclusively from arithmetic-v2 verification metadata and appends the
  recomputed final answer.
- `vasu/data/arithmetic_step_supervision.py` compiles a shared
  `Question ... Response` prefix into final-answer control and verified-steps
  treatment variants. It fails closed if tokenizer encoding merges the
  prompt/response boundary.
- The stored loss mask is zero for every prompt token and one for every target
  response token and terminal EOS. The future trainer view is
  `stored_mask[1:]`; packed padding and cross-example transitions remain zero.
- The paired-split validator requires equal source IDs, prompts, recomputed
  answers, and normalized-expression hashes between arms, and rejects ID,
  prompt, and semantic-expression overlap across train, development, and
  evaluation.

Validation evidence: full repository suite passed with 1031 tests passing and
8 explicitly skipped after the paired-split validator was introduced.

## Independent review decisions required

1. Approve the shared response prefix and the two target serializations as the
   sole intended experimental difference.
2. Approve new Candidate E stable-ID namespace, operand ranges, template
   families, split counts, and seeds. They must be distinct from every
   arithmetic-v2 held-out release; no defaults are implied by the compiler.
3. Approve a final immutable-release schema, artifact retention location, and
   deterministic regeneration requirement.
4. Verify the authoritative tokenizer's prompt/response boundary for every
   planned record before any artifact is published.
5. Decide the matched token/update accounting method. The treatment has longer
   supervised responses, so equal example count is not a matched budget.
6. Freeze development and one-time held-out evaluation protocols before any
   training authorization is considered.

## Explicitly not authorized by this checklist

- Generating or publishing Candidate E data artifacts.
- Constructing mixture schedules or capability-training configurations.
- Registering authorization records, resuming a checkpoint, or launching any
  training process.

Existing checkpoints, tokenizer, releases, schedules, and inference behavior
remain unchanged and compatible.
