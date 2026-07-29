# Candidate E Step-Supervision Matched-Pair Design

Status: proposed; non-authorizing; no implementation or training

## Hypothesis

Answer-only supervision over deterministic intermediate arithmetic states will
improve held-out direct-operation exact accuracy more than an equal-token,
final-answer-only continuation. This tests learning signal, not arithmetic
token volume.

## Identities and comparison

- Parent: Candidate A final (`8a54ff13ca3ca3dac385270b2160c34e639a9ca8178603df33d30d54f0bd6103`)
- Control proposal: `capability_cpt_e_final_answer_control_10m_from_a_v1`
- Treatment proposal: `capability_cpt_e_verified_steps_10m_from_a_v1`
- Shared budget: a fresh, matched nominal 10M-token continuation; exact
  accounting must be fixed only after deterministic data construction.

Both branches retain the same parent, tokenizer, optimizer, scheduler, batch
shape, seed, source proportions, runtime safeguards, and frozen evaluations.
The sole intended difference is the arithmetic target serialization.

## Data and objective

A new immutable release must be derived from structured arithmetic-v2
verification metadata. Every treatment step must be deterministically computed
and independently rechecked before tokenization. Control examples expose the
same prompt and final answer but supervise only the final answer/EOS; treatment
examples supervise a bounded canonical sequence of verified intermediate states
and the identical final answer. Prompt tokens and padding must be unsupervised
in both branches. No chain-of-thought text may be invented or model-generated.

Train, development, and held-out evaluation splits require new disjoint stable
IDs, normalized expressions, operand ranges, and template families. Existing
arithmetic-v2 held-out splits remain untouched until a new plan explicitly
establishes their role.

## Required gates before authorization

1. Deterministic regeneration and SHA-256 identities for texts, tokens, masks,
   manifests, and schedules.
2. Per-step symbolic verification; target-shift mask alignment; no supervised
   prompt, PAD, or cross-record target.
3. Matched token/update accounting and a replay-safety analysis.
4. Frozen per-operation development gates, one-time held-out gates, broad
   retention gates, malformed-rate reporting, and explicit rejection criteria.
5. CUDA smoke, exact-resume, disk, thermal, checkpoint, and clean-tree gates.
6. Separate human review and authorization records for each new identity.

## Compatibility and boundary

This design changes no existing artifact. Checkpoints, tokenizer, datasets,
masks, schedules, and inference remain compatible. Candidate B remains blocked.
No Candidate E data preparation, config, authorization, or training may begin
until this design is independently reviewed and approved.
