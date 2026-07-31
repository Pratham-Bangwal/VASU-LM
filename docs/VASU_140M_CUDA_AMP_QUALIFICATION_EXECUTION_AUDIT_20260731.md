# VASU-140M CUDA/AMP Qualification Execution Audit — 2026-07-31

## Result

The one authorized synthetic CUDA/AMP qualification completed successfully.
It advances only readiness gate 1; it does not authorize a dataset release,
optimizer, base-pretraining plan, or training.

## Immutable result evidence

| Field | Value |
|---|---|
| Result path | `evaluation/results/vasu_140m_cuda_amp_qualification_20260731.json` |
| Result file SHA-256 | `b62859fe9f28a79861ecc826dfc14fb13168b953464b1d1ee2e9338511fe9fe0` |
| Canonical report SHA-256 | `ca1ac908b5c6621105b4d6b10fdf93146592a2302601acc761a80b882be32472` |
| Device | NVIDIA GeForce RTX 4050 Laptop GPU, CUDA device 0 |
| Precision | BF16 autocast |
| Workload | synthetic, batch 1, sequence 512, 3 warmup + 5 measured iterations |
| Peak allocated / reserved | 1,388,354,048 / 1,631,584,256 bytes |
| Throughput | 6,141.99 tokens/s |
| Temperature | 43°C before; 52°C maximum; below 88°C stop |
| Checkpoint I/O | 551,408,411 bytes; strict reload passed; temporary cleanup passed |
| Optimizer updates | 0 |

Every result check passed: family identity, finite iterations, cleared
gradients, unchanged state keys, strict temporary checkpoint reload, cleanup,
and no optimizer update. The output is immutable and ignored by Git. No
`vasu_140m_cuda_amp_*` temporary directory remained after success.

## Review-identity correction

The first independent-review request was correctly rejected because its pasted
prompt omitted the final four characters `9fe0` from the result-file SHA-256
and because this audit package was not yet committed, making the identity smoke
fail closed on a dirty worktree. The immutable evidence itself was not changed.
The corrected review must use the full SHA above after this audit package is
committed and the read-only identity smoke succeeds from a clean worktree.

## Compatibility and non-authorization

The qualification did not modify VASU model architecture, tokenizer, datasets,
masks, production release, checkpoints under `checkpoints/`, schedules, or
authorization records. `training_authorized=false` remains in the result.

## Remaining gates

1. Separately reviewed VASU-140M base-pretraining data release.
2. Frozen base-model evaluation contract.
3. Real-data CUDA exact-resume/checkpoint qualification.
4. Immutable scientific experiment plan, independent review, preflight, and
   a new hash-bound human training authorization.
