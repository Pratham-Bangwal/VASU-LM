# VASU-140M Checkpoint Qualification — 2026-07-30

Status: passed for additive family identity, atomic model-only round trip,
strict state restoration, and explicit wrong-family rejection. This does not
qualify optimizer/scheduler state, exact resume, CUDA checkpoint I/O, data, an
experiment, or training.

## Evidence identity

- Family: `vasu_140m_v1`
- Family SHA-256:
  `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b`
- Config SHA-256:
  `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059`
- Result:
  `evaluation/results/vasu_140m_checkpoint_qualification_20260730.json`
- Canonical payload SHA-256:
  `6c06b95f6e76a3c5b91641aa716f4743cb82f5242b3dccff4aafe975b075788d`
- Result-file SHA-256:
  `1c6927a0e76ec6f5b135ba93fb35ca14fed62ece7a965bf666e7cfd1c690c34b`

The JSON result is an ignored local evaluation artifact. The retained result
contains no checkpoint tensors.

## Additive identity contract

New family-aware workflows may add `model_family_identity` to the existing
checkpoint mapping. Its exact v1 fields are:

- schema `vasu.model-family-checkpoint-identity.v1`;
- versioned family ID;
- family SHA-256;
- complete model-config SHA-256;
- exact unique parameter count.

The legacy checkpoint writer and loader are unchanged. Existing VASU-31M and
VASU-60M checkpoints are neither migrated nor required to contain this field.
The new validator is opt-in and fail-closed for future family-aware workflows.

Before `load_state_dict`, it verifies both the checkpoint identity and the
destination model's exact `ModelConfig`. A family ID, family digest, config
digest, parameter count, missing identity, empty state, or destination-config
mismatch is rejected.

## Disposable round trip

The qualification used seed 140043, CPU FP32, PyTorch 2.8.0+cpu, and no
optimizer or scheduler. It used the existing atomic checkpoint writer with
post-write reload verification and `fsync`.

| Observation | Value |
|---|---:|
| Checkpoint bytes | 551,408,411 |
| Checkpoint SHA-256 | `252fc6c750b62fb2b5bc230fa221f44d50d0cbf2579b214f89605e019a900ecd` |
| Save observation | 0.559 seconds |
| Memory-mapped load call | 0.003 seconds |
| State tensors compared | 111 |

The load-call timing reflects memory mapping and is not a full read-through
benchmark. Exact comparison subsequently touched every tensor.

All 111 state tensors matched bit-for-bit after strict loading into a fresh
VASU-140M model. Input/output weight tying remained intact. Declaring the
checkpoint as `vasu_60m_v1` was rejected. Loading it into a VASU-60M
destination while declaring `vasu_140m_v1` was also rejected before
`load_state_dict`.

The temporary 551 MB checkpoint and its `.tmp` sibling were absent after the
run. Free disk before and after was identical. The retained report declares
`training_authorized: false`.

## Decision

The model-only checkpoint round-trip and wrong-family rejection gate is
accepted. The identity block is suitable for the future VASU-140M checkpoint
path because it adds protection without changing legacy containers.

The next non-training gate is exact-resume qualification with a synthetic,
isolated training state. That future work must prove optimizer, scheduler,
sampler, RNG, and partial-accumulation equivalence and must not authorize a
real experiment.
