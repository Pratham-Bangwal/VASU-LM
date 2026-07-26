# Candidate A Authorization Review

## Review scope

This is a preauthorization review for
`capability_cpt_a_factual_20m_v2`. No training authorization was applied and
no training was started. Candidate A remains `training_authorized: false`.
Candidate B also remains unauthorized.

## Current decision

**Runtime safeguards verified; ready for final authorization preparation, but
not approved for authorization.**

The current artifact is internally valid as an unauthorized schedule release,
and its parent-checkpoint blocker is resolved in favor of the FineWeb
step-200,000 checkpoint. The Candidate C-equivalent hardened production-runtime
configuration is now present and validated. Candidate A remains blocked from
launch until a separate candidate-specific final authorization record is
created and approved.

## Mixture and training accounting

| Item | Verified value |
| --- | --- |
| Mixture | 86% FineWeb replay, 9% Wikimedia factual, 5% verified arithmetic v2 |
| Target tokens | 20,004,864 |
| Records | 78,144 fixed records |
| Batch / sequence | 2 / 256 |
| Dataloader microbatches | 39,072 |
| Gradient accumulation | 16 microbatches |
| Optimizer updates | 2,442 |
| Tokens per microbatch | 512 |
| Warmup updates | 49 |
| Scheduler total steps | 2,442 |
| Final accumulation group | complete |
| Optimizer | standard AdamW, learning rate `1e-5`, weight decay `0.1` |
| Validation/checkpoint intervals | 100 / 200 optimizer updates |

The arithmetic source is used only through its training records. Its `dev`
and `eval` JSONL splits are separately referenced as development/evaluation
only and are not scheduled training sources.

## Resolved parent-checkpoint decision

The current config uses:

`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`

Candidate A must retain the FineWeb step-200,000 parent. Candidate A and
Candidate C are matched approximately 20M-token branches from the same
weights: C is the 91% FineWeb / 9% Wikimedia / 0% arithmetic control, and A
is the 86% FineWeb / 9% Wikimedia / 5% verified-arithmetic-v2 treatment. The
5% arithmetic allocation is the main experimental variable.

Starting Candidate A from Candidate C `final.pt` would append a second 20M
stage, creating unequal total token exposure, sequential-curriculum effects,
confounded attribution, altered lineage, and weaker A-versus-C comparability.
Candidate C `final.pt` remains eligible as the preferred practical
continued-pretraining base; it is not the parent for the controlled Candidate
A treatment branch. A later sequential Candidate-C-parented experiment would
require a separate candidate identity.

The current config already uses the selected parent path and hash. No config,
resolved-manifest, schedule, dataset, tokenizer, mask, checkpoint, or
hyperparameter change is required. The formal record is
`docs/CAPABILITY_CPT_A_FACTUAL_20M_V2_PARENT_CHECKPOINT_DECISION.md`.

Candidate A and Candidate C remain directly comparable in parent checkpoint,
20,004,864-token budget, 78,144 records, 39,072 microbatches, 2,442 optimizer
updates, 2,442-step scheduler, 49-update warmup, sequence length 256, batch
size 2, and gradient accumulation 16.

## Hash-bound identities verified

| Artifact | SHA-256 |
| --- | --- |
| Candidate A config | `32e589bce495ac4d70c4da52cd69c31f4f575d323a8f8b4f6490819af68b5b79` |
| Current parent checkpoint | `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f` |
| Tokenizer | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Resolved mixture manifest | `bda898d742d44123f647cd318aedc2bf28dedb24986da688716aee2778cb0ba9` |
| Schedule | `cc9cd9b7d2a2cd55a91c805691ce1c872fac82cd6295c5626383f8aea2ca15ce` |
| FineWeb extension token source | `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e` |
| FineWeb manifest | `d6f1d112fbd0135563c173feaf700162982718a6ced6abd537df0e749c267a00` |
| Wikimedia token source | `fee0b88832fc569ce73393bc232e29a80cb981e31779ce4c42129d15a0be876f` |
| Wikimedia tokenized manifest | `cec126ae4e804a4f42699f96625752199c73b0caafbc4bf8b6d3bc3f47560a81` |
| Arithmetic tokens | `9abbab5e6a3d6d1121d47a89e9e829a3b51b072af2dc3a9daed5bae5d6fe474a` |
| Arithmetic loss mask | `beaf4d1f5dc325d1fc2a37cf8adc6677738e009fdf68d87fbf1bb032013098c4` |
| Arithmetic manifest | `8da3a1acfb2d2481c295b0d226c4695613b56670989d001fa27e65bed366115c` |
| Arithmetic dev split | `e3130b9584d052bdf6848b0d394e8515d522fd5f682ab9b6c9795535b7a903d0` |
| Arithmetic eval split | `5d3fe19e950af146f1c345e6164e7bbd56618e38d8cc76df328e40eca8da6dd2` |
| CUDA validation artifact | `3555983cc9bcd58e6df70592f21c6b6c0ff064208b342bbc6f4cdc88eab05268` |

The repository validator confirmed the config, resolved manifest, schedule,
parent, tokenizer, source identities, replay-safety metadata, and technical
artifact hashes. The arithmetic mask is target-aligned by the existing v2
tests; PAD tails and cross-record masking are covered by the packed-source
tests. No train/evaluation split overlap was introduced.

## Hardened production-runtime review

Candidate A now uses the same `vasu_capability_runtime_v1` production runtime
configuration as the successfully completed Candidate C run. This targeted
configuration correction adds production validation, explicit-resume, atomic
checkpoint, retention, disk, thermal, abort, and domain-best policies without
changing Candidate A's parent, sources, schedule, masks, mixture identity,
token budget, optimizer/scheduler hyperparameters, or authorization state.

| Safeguard | Result |
| --- | --- |
| Exact mid-epoch resume, optimizer/scheduler/warmup/RNG restoration | Passed by capability-runtime and resumable-training coverage |
| Schedule position and partial accumulation recovery; no repeated/skipped records | Passed by deterministic scheduled-mixture exact-resume coverage |
| Atomic checkpointing and corruption detection | Passed |
| Checkpoint experiment-identity validation | Passed; identity binds parent, tokenizer, schedule, sources, validation, optimizer, and scheduler |
| Explicit resume only; isolated A output directory | Passed |
| Periodic retention; `latest.pt`, `final.pt`, and domain-best behavior | Passed; dry-run reported no removals from the empty isolated directory |
| Disk-space and thermal safeguards | Passed by runtime validation and focused tests; 10 GiB margin and 82/87/90 C policy are configured |
| Non-finite loss/gradient detection and optimizer-skip accounting | Passed by capability-runtime coverage |
| Validation intervals and graceful interruption | Passed by capability-runtime coverage |
| Authorization-record and hash-bound config validation | Passed; candidate-specific approved record remains required |
| Unauthorized launch rejection | Passed before model training |

The production identity binds the exact Candidate A parent, tokenizer, resolved
manifest, schedule, source hashes, validation configuration, standard AdamW
backend, cosine scheduler, 2,442 total scheduler steps, and 49 warmup updates.
It rejects a mismatched Candidate C or Candidate B checkpoint, an altered
Candidate A config, or a checkpoint with mismatched tokenizer, schedule,
source, parent, optimizer, scheduler, or experiment identity.

Candidate A retains `training_authorized: false`. A temporary in-memory copy
with only that flag changed to `true` was rejected because the exact
candidate-specific authorization record does not exist. Candidate B remains
blocked independently.

## Preflight and validation results

- Candidate A validate-only workflow: passed with the full production identity;
  it performed no optimizer update.
- Retention dry-run: passed with no deletion candidates in Candidate A's empty,
  isolated checkpoint directory.
- Normal launch attempt: rejected before model training with
  `Training is blocked because training_authorized is false.`
- Source integrity: passed for the 86% FineWeb, 9% Wikimedia, and 5% verified
  arithmetic-v2 schedule; schedule SHA-256 is
  `cc9cd9b7d2a2cd55a91c805691ce1c872fac82cd6295c5626383f8aea2ca15ce`.
  Arithmetic token/mask equality, shifted-target alignment, PAD tails,
  cross-record masking, development/evaluation exclusion, and deterministic
  schedule replay passed through the focused tests.
- Production accounting: passed: 78,144 records, 39,072 microbatches, batch
  size 2, sequence length 256, accumulation 16, 2,442 updates, 20,004,864
  target tokens, 49 warmup updates, a 2,442-step scheduler, and a complete
  final accumulation group.
- Live CUDA smoke: not run because CUDA is unavailable in this environment.
  The Candidate-A-compatible CUDA artifact hash was verified as
  `3555983cc9bcd58e6df70592f21c6b6c0ff064208b342bbc6f4cdc88eab05268`.
- Candidate A training: not started.

## Prepared authorization material

No final authorization record was created. The remaining final-authorization
task must bind the exact config SHA above, parent SHA, tokenizer SHA,
resolved-manifest SHA, schedule SHA, all source token/mask/manifest hashes,
the 20,004,864-token budget, 2,442-update budget, and the exact launch command:

```powershell
python .\train_vasu_60m_capability_cpt.py `
  --config .\configs\training\capability_cpt_a_factual_20m_v2.json
```

Any resumed run must provide an explicit, identity-checked checkpoint with
`--resume-from`; no implicit checkpoint discovery is authorized.

## Compatibility

No model architecture, tokenizer, checkpoint, dataset, schedule, mask,
mixture identity, or training hyperparameter was changed. The Candidate A
runtime-safeguard configuration is additive and matches Candidate C's hardened
runtime policy. Candidate C and the Alpaca v3 assistant remain unaffected.
Candidate A is not trained or authorized.
