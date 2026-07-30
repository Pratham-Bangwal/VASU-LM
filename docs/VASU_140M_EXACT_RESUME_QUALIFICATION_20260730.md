# VASU-140M Exact-Resume Qualification — 2026-07-30

Status: passed after an evidence-driven loader-RNG isolation fix. The
qualification used four synthetic records and two optimizer updates; it was
not real-data training and does not authorize an experiment.

## Evidence history

The first immutable run failed:

- result: `evaluation/results/vasu_140m_exact_resume_qualification_20260730.json`;
- payload SHA-256:
  `47cd56e8aaa5e99dff6f080c2021d9b5797b00e1402962d143ed5b0a97f0d369`;
- file SHA-256:
  `59a3834bdedb0ee3c442af4532c83370859cf641a17a9abebec19b382bf9b37b`.

Sampler position, sample order, progress counters, scheduler, scaler, dataset
identity, family identity, and partial gradients matched. Model, AdamW, and
PyTorch RNG state did not.

Root cause analysis showed that constructing a second `DataLoader` iterator
after restoring the checkpoint consumed two values from PyTorch's global RNG.
The uninterrupted path created one iterator; the resumed path created two.
Dropout therefore received a different RNG stream after resume. Earlier tiny
tests used a model without stochastic layers and could not expose this defect.

The trainer now gives training and validation loaders separate, deterministically
seeded `torch.Generator` instances. DataLoader bookkeeping no longer consumes
model RNG state. This is additive and does not alter sampler order, dataset
identity, checkpoint fields, or legacy checkpoint loading. Random worker-side
transforms must still be stateless or deterministic.

The corrected immutable v2 run passed:

- result:
  `evaluation/results/vasu_140m_exact_resume_qualification_v2_20260730.json`;
- payload SHA-256:
  `ded0dd8b57e5b6ea12b2ef8b16988f8a1047f50712d3f1711afc0b2ed7f50350`;
- file SHA-256:
  `2dfacfe26f9a0345a4dd01eed6dfeed4144238041c6506365a677884e500ec97`.

Both failed and passing results are preserved. The failed result is negative
evidence and must not be replaced or relabelled.

## Frozen workload

| Setting | Value |
|---|---:|
| Family | `vasu_140m_v1` |
| Seed | 140044 |
| Synthetic records | 4 |
| Sequence length | 4 |
| Batch size | 1 |
| Gradient accumulation | 2 |
| Optimizer | Standard AdamW |
| Scheduler | LambdaLR |
| Optimizer updates | 2 |
| Interruption | After microbatch 1 |
| Real data | false |

The synthetic token dataset carries a deterministic content-derived resume
identity. The interruption checkpoint was 1,102,823,453 bytes with SHA-256
`81f842fa05dc901734f51f88b66f5c0bcef163ad7b7bc20be52ad6635ce3826f`.
It recorded global step 0, next batch 1, one accumulated microbatch, family
identity, dataset identity, and all 110 gradient tensors.

## Corrected equivalence result

The uninterrupted and resumed branches matched exactly for:

- every model tensor;
- complete AdamW state;
- scheduler state;
- AMP scaler state;
- final sampler state;
- consumed sample order `[3, 0, 1, 2]`;
- global step, optimizer-step count, accumulation position, and resume phase;
- next Python, NumPy, and PyTorch random values.

Both branches ended at global step 2 with an exhausted epoch sampler and
`post_train_pre_validation` phase. The resumed process restored next batch 1,
one partial microbatch, and global step 0 before continuing.

The control and resumed model digest was
`baeab5c89dd89297c4c6d60b53190a367cca0e70f584fcce65306bbfc90a6a7a`.
Their optimizer digest was
`fd74bcaa615c5b4a6c321c1d3ce85cec44df63e295790656ae9810f513ccf369`.

All temporary checkpoints and worker artifacts were removed, and free disk
before and after was identical. The retained report states
`training_authorized: false`.

## Decision and limits

The CPU FP32 exact mid-accumulation resume gate is accepted for the frozen
synthetic VASU-140M workload. The trainer's family-aware path now proves
identity binding plus exact model, optimizer, scheduler, sampler, partial
gradient, scaler, and RNG continuation.

This does not qualify CUDA/AMP execution, multiworker random transforms,
full-context training, real datasets, 513-token releases, model quality, or a
training plan. CUDA qualification and independent data/evaluation gates remain
closed.
