# VASU-140M Base Checkpoint Selection and Training Plan Gate

Status: **blocked by an empty compatible-checkpoint set; non-authorizing.**

## Decision

No VASU-140M base checkpoint is selected. The checkpoint inventory contains
VASU-31M and VASU-60M artifacts but no artifact that can satisfy the
`vasu_140m_v1` family contract. Selecting a smaller-family checkpoint by
filename, partial weight load, conversion, or shape-tolerant loading would
destroy the causal parent identity required for a valid instruction-stage
experiment.

This is a correct fail-closed outcome, not a reason to reuse Candidate A.
Candidate A is the preferred VASU-60M continued-pretraining base; it is not a
VASU-140M checkpoint.

## Bound release and model identities

| Item | Required identity |
|---|---|
| Model family | `vasu_140m_v1` |
| Model configuration SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Family SHA-256 | `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b` |
| Parameter count | `137,841,408` |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Instruction release manifest file SHA-256 | `b5a88ac62cd1a8ba2d449873673b935ec439e3e61b7bef10c0c6314aec67260a` |
| Instruction release logical-record SHA-256 | `95ec1692e229a8b1288d985cceff5dad4c8b1ac5df1e62975740b071721a140b` |
| Split counts | 898 train / 48 development / 50 evaluation |
| Record contract | `uint16[513]` tokens with `uint8[513]` shifted-mask storage |

The release is published and valid, but its consumed one-build authorization
explicitly has `training_authorized=false`. Publication authority is not
training authority.

## Selection requirements

A future candidate is eligible only if all conditions hold:

1. Its checkpoint metadata passes `validate_checkpoint_family_identity` for
   `vasu_140m_v1`, including the exact family/configuration fingerprint and
   parameter count.
2. It loads strictly into the frozen VASU-140M model with no missing,
   unexpected, resized, converted, or selectively ignored tensors.
3. Its tokenizer identity matches the frozen tokenizer above.
4. Its provenance identifies its immutable parent, base-pretraining data,
   schedule, optimizer state, global step, repository commit, and checkpoint
   file SHA-256. A model-only checkpoint can be evaluated, but cannot support
   resumable training.
5. It is a base-pretraining checkpoint, not a VASU-60M instruction checkpoint
   or a checkpoint already contaminated by this instruction release.
6. It has passed the separately required CUDA/runtime, exact-resume, and
   frozen base-model evaluation gates, with an independent acceptance record.

## Inventory result

The repository checkpoint inventory contains no directory or `.pt` file
identified as VASU-140M. Its current compatible-candidate set is therefore
empty.

| Candidate class | Result | Reason |
|---|---|---|
| VASU-60M Candidate A `final.pt` | Rejected | `vasu_60m_v1`, 58,337,792 parameters; wrong family and tensor shapes. |
| Other VASU-60M milestones and continuation checkpoints | Rejected | Same `vasu_60m_v1` incompatibility; branch lineage does not alter architecture. |
| Legacy VASU-31M checkpoints | Rejected | Wrong model family and dimensions. |
| Existing VASU-140M checkpoint | Absent | No eligible artifact is present. |

## Alternatives considered

| Alternative | Assessment |
|---|---|
| Treat Candidate A as the parent | Rejected: it confounds the 140M capacity experiment and cannot strictly load. |
| Convert or partially transplant 60M weights | Rejected: changes initialization and creates unvalidated lineage. It would require a separate architecture/initialization experiment. |
| Create a fresh 140M base-pretraining program | Correct future path, but blocked until a separate scientific plan, source release, CUDA evidence, authorization, and resulting 140M checkpoint exist. |
| Select no checkpoint | Recommended now: preserves scientific validity and fails closed. |

## Non-executable future instruction-stage plan

Once a qualifying VASU-140M base checkpoint exists, a new immutable plan must
bind its file SHA-256 and all requirements above before any configuration is
created. It must state a single hypothesis: whether supervised use of the
published instruction seed improves frozen instruction-following behavior
without unacceptable regression in frozen factuality, repetition, arithmetic,
or base-language evaluations.

The plan must include a parent-matched control, fixed prompt format and
generation settings, the published release hashes, development-only
diagnostics, held-out evaluation isolation, promotion and rejection thresholds,
token and optimizer-update budgets, optimizer/scheduler settings, runtime
safeguards, exact-resume validation, and a hash-bound authorization record.
None of those fields are selected here. This document is deliberately not a
training configuration, schedule, authorization, or launch instruction.

## Next gate

The next valid engineering objective is a separately reviewed VASU-140M
base-pretraining readiness package. It must not use or modify the published
instruction release, and it must produce a newly qualified VASU-140M base
checkpoint before instruction-stage planning can continue.
