# Candidate A Parent-Checkpoint Scientific Decision Record

Status: **decided; Candidate A remains unauthorized**

Decision date: 2026-07-26  
Candidate: `capability_cpt_a_factual_20m_v2`

## Decision

Candidate A must use the immutable FineWeb step-200,000 checkpoint as its
parent:

`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`

SHA-256:
`88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f`

The existing Candidate A configuration already binds that exact path and hash.
No configuration, resolved-manifest, schedule, dataset, tokenizer, mask,
checkpoint, or hyperparameter edit is required or made by this decision.

## Scientific rationale

Candidate A is the controlled treatment for Candidate C. Both branches must
start from the same parent and use the same approximately 20M-token
continuation budget. Candidate C is the control mixture: 91% FineWeb replay,
9% Wikimedia factual data, and 0% arithmetic. Candidate A is the treatment
mixture: 86% FineWeb replay, 9% Wikimedia factual data, and 5% verified
arithmetic v2. The intended experimental variable is therefore the 5%
arithmetic source.

Starting Candidate A from Candidate C `final.pt` would add Candidate C's
completed approximately 20M-token stage before Candidate A's own 20M-token
stage. That would create unequal total token exposure, sequential-curriculum
effects, confounded attribution, altered checkpoint lineage, and weaker
Candidate A-versus-C comparability. It would not answer the controlled
treatment question.

Candidate C `final.pt` remains eligible as the preferred practical
continued-pretraining base candidate because of its completed control-run
results. That practical selection is distinct from the parent needed for the
scientific comparison. The preferred instruction-tuned interactive assistant
remains the separate masked Alpaca v3 checkpoint:

`checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.

## Direct-comparison invariants

| Invariant | Candidate A | Candidate C | Status |
| --- | --- | --- | --- |
| Parent checkpoint | FineWeb step-200,000 | FineWeb step-200,000 | Same |
| Token budget | 20,004,864 | 20,004,864 | Same |
| Records | 78,144 | 78,144 | Same |
| Dataloader microbatches | 39,072 | 39,072 | Same |
| Optimizer updates | 2,442 | 2,442 | Same |
| Scheduler length | 2,442 updates | 2,442 updates | Same |
| Warmup | 49 updates | 49 updates | Same |
| Sequence length | 256 | 256 | Same |
| Batch size | 2 | 2 | Same |
| Gradient accumulation | 16 microbatches | 16 microbatches | Same |

The mixtures differ only as required for the ablation: Candidate A reallocates
5 percentage points of FineWeb replay to verified arithmetic v2 while keeping
the 9% Wikimedia share fixed. This preserves the intended direct A-versus-C
comparison.

## Scope and follow-on constraints

This decision resolves only the Candidate A parent-checkpoint blocker. It does
not authorize Candidate A, change `training_authorized: false`, create a final
signed authorization record, or start training. Candidate A still requires a
separately reviewed hardened production-runtime configuration and a
candidate-specific final authorization record before any launch can be
considered. Candidate B remains unauthorized and conditional.

A later sequential experiment may start from Candidate C `final.pt`, but it
must be assigned a new candidate identity and must not replace, redefine, or
be presented as Candidate A.

## Compatibility

This documentation-only decision changes no model architecture, tokenizer,
datasets, schedules, masks, checkpoints, checkpoint schema, or training
hyperparameters. Existing Candidate C and Alpaca v3 roles remain unchanged.
