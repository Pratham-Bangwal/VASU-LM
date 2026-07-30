# VASU-140M Instruction Seed Release Independent Review Packet

Status: independent review completed and accepted; non-authorizing.

Date: 2026-07-30

## Decision requested

Accept or reject the specification-only
`vasu_140m_instruction_seed_v1` source and release plan. Acceptance permits a
later, separately authorized release-construction implementation and immutable
release review. It does not permit data construction or training.

## Materials

- [release plan](VASU_140M_INSTRUCTION_SEED_RELEASE_PLAN.md)
- `configs/data/releases/vasu_140m_instruction_seed_v1.plan.json`
- `vasu/data/vasu_140m_release_plan.py`
- `scripts/smoke_vasu_140m_release_plan.py`
- `tests/test_vasu_140m_release_plan.py`
- `evaluation/fixtures/vasu_140m_instruction_seed_v1_plan_report.json`
- [completion audit](VASU_140M_INSTRUCTION_SEED_RELEASE_MILESTONE_AUDIT_20260730.md)

## Requirement-to-evidence audit

| Requirement | Evidence | Result |
| --- | --- | --- |
| Accepted record contract | Exact fixture and GPT-5.5 decision file hashes | Pass |
| Correct training-stage scope | Explicit response-masked instruction stage; not base pretraining | Pass |
| Source provenance | Two exact purpose-written source and manifest identities | Pass |
| License | Every one of 1,000 records declares CC0-1.0 | Pass |
| Human review | Every source ID has a current hash-bound approval | Pass |
| Cross-source deduplication | Zero exact groups and zero threshold-0.85 near candidates | Pass |
| Evaluation inventory | Ten frozen files, 2,618 prompts, exact hashes | Pass |
| Contamination handling | Four exact matches quarantined; zero eligible matches | Pass |
| Split preservation | All source validation remains development-only | Pass |
| Deterministic evaluation split | Seed 140513, source/capability strata, frozen assignment hash | Pass |
| Capability coverage | Frozen seven-category counts after quarantine | Pass |
| Output isolation | Versioned paths, overwrite false, all planned outputs absent | Pass |
| No release construction | Read-only qualification; no logical/token/mask artifact | Pass |
| No training semantics | No base checkpoint, plan, configuration, or authorization | Pass |

## Reviewer questions

1. Is an instruction-only seed release the correct interpretation of the
   accepted response-masked 513-token contract?
2. Is preserving source validation as development stronger than repartitioning
   all reviewed examples?
3. Is the four-item quarantine sufficient and traceable?
4. Are 2,618 frozen prompts and eight-word fragment checks adequate for this
   bounded internal source?
5. Does the largest-remainder, source/capability-stratified evaluation
   selection avoid avoidable imbalance?
6. Are the construction and training non-authorizations unambiguous?

## Explicit exclusions

This packet contains no production logical manifest, token or mask binary,
source mutation, base checkpoint selection, schedule, training configuration,
compute budget, experiment hypothesis, authorization, checkpoint, or training
run.
