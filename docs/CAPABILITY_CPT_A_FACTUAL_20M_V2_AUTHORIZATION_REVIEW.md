# Candidate A Authorization Review

## Review scope

This is a preauthorization review for
`capability_cpt_a_factual_20m_v2`. No training authorization was applied and
no training was started. Candidate A remains `training_authorized: false`.
Candidate B also remains unauthorized.

## Current decision

**Blocked pending corrections; not approved for authorization.**

The current artifact is internally valid as an unauthorized schedule release,
but it is not ready for a final authorization record. The principal blockers
are the unresolved parent-checkpoint decision and the absence of the hardened
production-runtime, thermal, disk, retention, and authorization metadata that
are required for a real launch.

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

## Parent checkpoint challenge

The current config uses:

`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`

This is the cleanest parent for isolating Candidate A against the original
Candidate C control design: both branches would start from the same weights,
so arithmetic effects are easier to attribute and total exposure remains
comparable. Its drawback is that Candidate A does not inherit Candidate C's
measured Wikimedia improvement.

The completed Candidate C checkpoint,
`checkpoints/vasu_60m/capability_cpt_c_control_20m_v2/final.pt`, is the better
parent if the scientific question is specifically whether arithmetic can be
added while retaining Candidate C's factual gains. Its drawback is sequential
confounding: Candidate A would include Candidate C's 20M-token update plus its
own 20M-token update, so arithmetic and continued exposure cannot be
attributed as cleanly to the A-vs-C comparison.

**Recommendation:** use the Candidate C final checkpoint for the intended
"preserve Candidate C improvements" experiment, but only after an explicit
new review updates the parent identity, regenerated schedule/config hashes,
and authorization packet. If strict A-vs-C causal isolation is the priority,
retain the current step-200,000 parent and describe A as a parallel ablation.
The current config must not be silently changed; this review does not choose
or apply that edit.

## Hash-bound identities verified

| Artifact | SHA-256 |
| --- | --- |
| Candidate A config | `11744b55fa40d234e5c0e0e49b5f853ffdbb5791eceea64204f9106a1ce494ea` |
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

## Safety and resume review

Deterministic schedule replay and exact-resume behavior are covered by the
scheduled-mixture and resumable-training tests. The authorization gate now
requires an approved, candidate-specific, hash-bound authorization record for
any `training_authorized: true` launch; changing the config, parent,
tokenizer, mixture, schedule, token budget, or update budget invalidates the
authorization. Candidate A remains blocked by its false flag, and Candidate B
remains blocked independently.

The current A config does not contain the production runtime section that
declares thermal monitoring, disk checks, retention, explicit resume, and
interval runtime policy. It therefore cannot satisfy the hardened launch
requirements without a separately reviewed config update. The current output
directory is not treated as an authorization target.

## Preflight and validation results

- Candidate A validate-only workflow: passed; it performed no optimizer update.
- Schedule replay/source integrity: passed through resolved-manifest validation.
- Arithmetic mask alignment and v2 release tests: passed.
- Exact-resume and corruption/retention/disk/thermal unit tests: passed.
- Live CUDA smoke: not run because CUDA is unavailable in this environment;
  the recorded CUDA validation artifact was hash-verified.
- Candidate A training: not started.

## Prepared authorization material

No final authorization record was created. If the parent decision is resolved
and the runtime safeguards are added, the future packet must bind the exact
config SHA above (or a newly generated SHA), parent SHA, tokenizer SHA,
resolved-manifest SHA, schedule SHA, all source token/mask/manifest hashes,
the 20,004,864-token budget, 2,442-update budget, and the exact launch command:

```powershell
python .\train_vasu_60m_capability_cpt.py `
  --config .\configs\training\capability_cpt_a_factual_20m_v2.json
```

Any resumed run must provide an explicit, identity-checked checkpoint with
`--resume-from`; no implicit checkpoint discovery is authorized.

## Compatibility

No model architecture, tokenizer, checkpoint, dataset, schedule, or
hyperparameter was changed. Candidate C and the Alpaca v3 assistant remain
unaffected. Candidate A is not trained or authorized.
