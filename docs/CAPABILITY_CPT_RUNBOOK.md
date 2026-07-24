# VASU-60M Capability-CPT Runbook

This runbook is a review artifact. It does not authorize training. Candidate C,
A, and B currently retain `training_authorized: false`.

## Preflight

Run from `D:\VASU` with no training process active:

```powershell
git status --short
git rev-parse HEAD
Get-PSDrive D

.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config configs/training/capability_cpt_c_control_20m_v2.json `
  --validate-only
```

Before authorization, independently recompute the parent, tokenizer, resolved
manifest, schedule, CUDA-report, validation, and source hashes listed in
`CAPABILITY_CPT_AUTHORIZATION_PACKET.md`. Require an empty output directory.
The launcher checks its conservative retained-checkpoint estimate plus the
configured 10 GiB margin before model allocation and before every save.

## Authorization

Authorization is two deliberate changes:

1. complete a copy of
   `configs/authorization/capability_cpt_c_control_20m_v2.authorization.template.json`;
2. in a separately reviewed change, alter only Candidate C's
   `training_authorized` field from `false` to `true`.

Never add a bypass flag. Candidate A and B remain false.

## Launch commands after authorization

Candidate C:

```powershell
.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config configs/training/capability_cpt_c_control_20m_v2.json
```

Candidate A, only after a separate C-result review and A authorization:

```powershell
.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config configs/training/capability_cpt_a_factual_20m_v2.json
```

## Monitoring and safe interruption

Monitor `nvidia-smi`, available disk, process identity, step/loss/LR, optimizer
skips, validation metrics, and newly written checkpoint hashes. The launcher
warns at 82 C and stops after two consecutive readings at 87 C or one reading
at 90 C. It also aborts on the configured validation, optimizer-skip, disk,
checkpoint, identity, and CUDA-OOM conditions. Ctrl+C is a safe manual
interruption when the current checkpoint write has completed.

Resume is always explicit. Choose only a validated `latest.pt`, periodic
checkpoint, or thermal-stop checkpoint from the same candidate:

```powershell
.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config <candidate-config> `
  --resume-from <validated-periodic-checkpoint>
```

The launcher validates the checkpoint on CPU before model allocation and
restores/verifies schedule/dataset identity, sampler position,
accumulation count and gradients, optimizer, scheduler, GradScaler, RNG,
global step, validation-event state, and hashes before processing the next
sample. `final.pt` is evaluation-only and is rejected as a resume source.

Candidate C resume:

```powershell
.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config configs/training/capability_cpt_c_control_20m_v2.json `
  --resume-from checkpoints/vasu_60m/capability_cpt_c_control_20m_v2/latest.pt
```

Candidate A requires its own reviewed production-runtime configuration and
authorization after C. Once that exists, its initial and resume commands use
`capability_cpt_a_factual_20m_v2.json` and the corresponding Candidate A
checkpoint directory. The current false authorization and missing A runtime
fields intentionally prevent premature launch.

## Validation and evaluation

The existing configuration/artifact validation command is safe:

```powershell
.venv\Scripts\python.exe train_vasu_60m_capability_cpt.py `
  --config <candidate-config> `
  --validate-only
```

After a checkpoint is produced, use the factual baseline evaluator for fixed
FineWeb/Wikimedia loss, factual prompts, repetition, response length, and
encyclopedic-style indicators:

```powershell
.venv\Scripts\python.exe -m evaluation.evaluate_factual_cpt_baseline `
  --config configs/evaluation/vasu_60m_factual_cpt_v1.json `
  --checkpoint <checkpoint> `
  --result-path <versioned-result.json> `
  --text-output-path <versioned-result.txt>
```

Add the produced checkpoint to `evaluation/checkpoint_config.json` with
`model_config: vasu_60m` and `prompt_format: plain`, then run the deterministic
capability gate:

```powershell
.venv\Scripts\python.exe -m evaluation.framework.runner `
  --suite evaluation/suites/vasu_capability_v1.json `
  --checkpoints vasu_60m_fineweb_200k <candidate-id> `
  --parent vasu_60m_fineweb_200k `
  --promotion-type continued_pretraining `
  --mode greedy `
  --output-dir <new-versioned-output-directory> `
  --device cuda
```

Run full hash-bound arithmetic-v2 evaluation separately for development and
evaluation splits. Each completed record is persisted atomically; use
`--resume` after interruption:

```powershell
.venv\Scripts\python.exe -m evaluation.evaluate_verified_arithmetic_v2 `
  --checkpoint <checkpoint> `
  --split dev `
  --output-dir <new-versioned-dev-output> `
  --device cuda

.venv\Scripts\python.exe -m evaluation.evaluate_verified_arithmetic_v2 `
  --checkpoint <checkpoint> `
  --split eval `
  --output-dir <new-versioned-eval-output> `
  --device cuda
```

For Candidate C, substitute an explicitly selected checkpoint under
`checkpoints/vasu_60m/capability_cpt_c_control_20m_v2/`. Candidate A uses the
same commands only after its separate runtime configuration and authorization
are reviewed.

## Promotion decision

1. Validate all selected checkpoints strictly and hash them.
2. Compare C to the parent using the packet's C gates.
3. Record all objective, heuristic, repetition, runtime, and human dimensions
   separately.
4. Promote C only if every configured gate and additional guardrail passes.
5. Compare A primarily to C and secondarily to the parent.
6. Do not open B authorization unless every B prerequisite is documented.

## Rollback

The immutable rollback model is the step-200,000 parent. A candidate failure
does not alter it. Stop the candidate process, preserve the atomic run manifest
and last valid checkpoint for diagnosis, keep the candidate out of the
checkpoint registry used by production chat, and restore evaluation/default
references to the parent or previously approved assistant checkpoint. Never
overwrite or delete the milestone parent.
