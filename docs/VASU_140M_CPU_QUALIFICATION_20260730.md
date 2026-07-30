# VASU-140M CPU Qualification — 2026-07-30

Status: passed for the bounded CPU construction, forward/backward, and
KV-cache parity scope. This is not a throughput benchmark, CUDA qualification,
checkpoint/resume qualification, data approval, experiment authorization, or
training authorization.

## Evidence identity

- Family: `vasu_140m_v1`
- Family SHA-256:
  `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b`
- Config SHA-256:
  `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059`
- Result:
  `evaluation/results/vasu_140m_cpu_qualification_20260730.json`
- Canonical payload SHA-256:
  `d2b1149d019904ff5ef17657624e4642805d621ca447da931858ec4508171b29`
- Result-file SHA-256:
  `25960a97dbd3133a1166ff4fad821241cf2b555685a3fed8592d4c96cad5216d`

The result path is an ignored local evaluation artifact. The qualification
tool publishes it immutably and refuses to overwrite an existing result.

## Frozen bounded workload

| Setting | Value |
|---|---:|
| Device | CPU |
| Numeric type | FP32 |
| PyTorch | 2.8.0+cpu |
| Python | 3.13.7 |
| Torch threads | 8 |
| Seed | 140042 |
| Batch size | 1 |
| Forward/backward sequence length | 8 |
| Cache prompt length | 4 |
| Decode steps | 3 |
| Parameters | 137,841,408 |

Synthetic token IDs were used. No tokenizer, dataset, mask, checkpoint, or
training configuration was read.

## Results

The model constructed in 0.567 seconds. The single forward observation took
0.044 seconds and the backward observation took 0.184 seconds. These timings
are environment evidence only; one unwarmed observation is not a benchmark.

The output shape was `[1, 8, 32000]`. Logits, cross-entropy loss, and all
gradients were finite. All 110 parameter tensors received gradients, after
which every gradient was cleared. The utility never constructed an optimizer
and performed no optimizer update.

Dynamic and preallocated KV caches both matched uncached execution for the
complete prompt and every decode step under `rtol=1e-4`, `atol=1e-5`.
Maximum absolute error was `5.7220458984375e-06` for both implementations.
Both caches ended at sequence length 7. The full 512-token preallocated FP32
cache reserved 37,748,736 bytes.

Model state keys were identical before and after qualification. The result
reported `training_authorized: false`.

## Decision

The bounded CPU execution gate is accepted. This establishes that the exact
VASU-140M-v1 shape can execute a finite FP32 forward/backward graph and that
both cache implementations preserve logits on the tested CPU workload.

It does not establish:

- useful CPU or CUDA throughput;
- CUDA memory, kernel, thermal, or numerical behavior;
- full 512-token forward/backward feasibility;
- checkpoint round-trip or wrong-family rejection;
- exact resume;
- 513-token data or mask readiness;
- model quality or a scientific reason to train.

The subsequent checkpoint round-trip and wrong-family gate passed; see
`VASU_140M_CHECKPOINT_QUALIFICATION_20260730.md`. Exact resume is the next
non-training engineering gate. CUDA qualification remains separately
hardware-gated.
