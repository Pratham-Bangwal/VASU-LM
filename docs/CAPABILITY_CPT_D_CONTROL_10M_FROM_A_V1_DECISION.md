# Candidate D Control Scientific Decision Record

Status: **completed; accepted as the matched non-arithmetic control comparison checkpoint**

Decision date: 2026-07-29  
Candidate: `capability_cpt_d_control_10m_from_a_v1`  
Parent: `checkpoints/vasu_60m/capability_cpt_a_factual_20m_v2/final.pt`  
Selected checkpoint: `checkpoints/vasu_60m/capability_cpt_d_control_10m_from_a_v1/final.pt`

## Scope and completed training record

Candidate D control completed successfully with exit code 0. It is the non-arithmetic member of the matched Candidate D sequential pair: both planned runs start from Candidate A `final.pt`, SHA-256 `8a54ff13ca3ca3dac385270b2160c34e639a9ca8178603df33d30d54f0bd6103`. Candidate A final remains the common parent for both matched runs.

The control consumed 39,072 records and 10,002,432 processed tokens in 19,536 batch-size-2 microbatches. With sequence length 256 and gradient accumulation 16, it completed exactly 1,221 optimizer updates; its cosine scheduler had 1,221 steps and 25 warmup updates, and the final accumulation group was complete. Its deterministic mixture was 35,556 FineWeb records (91%), 3,516 Wikimedia records (9%), and zero arithmetic records. `final.pt`, `latest.pt`, and each of `best_fineweb.pt`, `best_wikimedia.pt`, and `best_arithmetic.pt` exist. The maximum observed thermal warning was 83 C; no thermal abort occurred.

## Hash-bound evidence

| Artifact | SHA-256 |
| --- | --- |
| Candidate D control `final.pt` | `3f513727ed0ea63a9b4aaf963c736f30b409e6901caddb50bf85c1db6922c384` |
| Arithmetic run manifest | `2093057584d7ac5a3ba202c77934bd76268a6dd1261d693a1c006b2d10a95c87` |
| Arithmetic per-example JSONL | `4cb227778066c3ca29f60a65ca6d3bca6c31a01b57abd7b9da7f0191130a757a` |
| Arithmetic category summary | `23066894a8e3aac6e62cd7198f8b3b8d986502a703d03ff755025b03f5e30658` |
| Arithmetic text summary | `5ef34ce4a029668c723425bee1a2e4c400ffd86084b31f464c7478e7371f34a4` |
| Factual parent JSON | `3d8cd45a807f8071688ba03b8196e551d2512ea6693b9878e9940bdcf765f95e` |
| Factual selected-checkpoint JSON | `44ddff0f58462ba4ecb056f6e35522fffed2987a165a19aaccb9afee4f37b618` |
| Factual best-Wikimedia JSON | `4e02cb846ec4795365a2499b92eb390d1555772abd5b46d455a8f3a5fb2517ba` |
| Factual comparison JSON | `9f2b01b3689a0d576ddef308bdbb27519bc8f43d48656bfa0b44a261de6c8a51` |
| Factual parent text report | `889b7de505fd803e119536272f5203fec5b8c7e80945060805909e906091582a` |
| Factual selected-checkpoint text report | `128813cccd0d25cd806c280fd9c0036a2c480d43598fe037e723e08bd23a92f3` |
| Factual best-Wikimedia text report | `d1946e374413c3d2a2759cf154ab1fc2c7bc9b8dc3944fb8547c11ab6e9779f9` |

Arithmetic evidence is retained at `evaluation/results/candidate_d_control_final_arithmetic_dev/`: `run_manifest.json` records status `complete`, the selected checkpoint hash, and evaluation identity `8c1411dd0ad35f55d9c10d20d2881887b35b099f69a4acfe772b651f9792cab6`. The JSONL contains exactly 1,000 unique development IDs. Factual evidence is retained at `evaluation/results/capability_cpt_d_control_10m_from_a_v1/`; the comparison uses evaluation configuration SHA-256 `8b8939bdaf1e7138b47891946b7019b5bc83dc20650c6553ec9b3124ae9701c6` and benchmark SHA-256 `17c356f0b3a5093511402b08f307b770b8cc4443b3ca3e1cceaa0eb7044115b9`.

## Arithmetic control result

| Checkpoint | Development exact accuracy | Held-out exact accuracy |
| --- | ---: | ---: |
| Candidate A final (parent) | 117 / 1000 (0.117) | 124 / 1000 (0.124) |
| Candidate D control final | 112 / 1000 (0.112) | Not evaluated in this control decision |

Candidate D control is lower than Candidate A by 0.005 absolute on the shared development evaluation. This is consistent with no meaningful arithmetic improvement from another 10M-token continuation containing no arithmetic source. No statistical-significance claim is made because the arithmetic evaluator does not provide one here. The result is the required matched-control evidence: non-arithmetic continuation alone does not explain a future treatment arithmetic gain.

## Factual and broad-language retention

| Metric | FineWeb step-200k | Candidate A | Candidate D control |
| --- | ---: | ---: | ---: |
| FineWeb loss | 3.356130 | 3.305468 | 3.299042 |
| Wikimedia loss | 3.393557 | 3.276147 | 3.264998 |
| Cloze | 0.070 | 0.110 | 0.120 |
| Multiple choice | 0.410 | 0.450 | 0.440 |

Against FineWeb step-200k, the control improves FineWeb loss by approximately 1.70% and Wikimedia loss by approximately 3.79%. Its normalized-cloze difference is `+0.05`, CI95 `[+0.01, +0.09]`, distinguishable from zero. Its multiple-choice difference is `+0.03`, but the confidence interval crosses zero. Relative to Candidate A, the control lowers FineWeb loss by about 0.006426 and Wikimedia loss by about 0.011149, raises cloze by 0.010, and lowers multiple choice by 0.010; the multiple-choice movement is likely within noise. Broad-language and factual quality were therefore preserved and slightly improved.

## Scientific decision and authorization boundary

Candidate D control is scientifically acceptable and is accepted as the matched-control comparison checkpoint for Candidate D treatment. Its completed 10M non-arithmetic continuation did not create arithmetic improvement, while preserving and slightly improving the broad-language and factual measures. It therefore supports proceeding to consideration of Candidate D treatment authorization under the matched-pair plan.

This decision does **not** authorize Candidate D treatment, change its configuration, or start training. Candidate D treatment remains `training_authorized: false` and prerequisite-blocked pending separate hash-bound authorization. Candidate B remains unauthorized. Candidate D control `final.pt` is a scientific matched-control comparison checkpoint, not the preferred production continued-pretraining base. Candidate A `final.pt` remains the preferred continued-pretraining base until treatment evaluation is complete. Masked Alpaca v3 remains the preferred instruction-tuned assistant.

## Compatibility

This decision changes only documented scientific status and checkpoint role. It does not alter model architecture, checkpoint schema, tokenizer, datasets, masks, schedules, mixtures, evaluator scoring, training configurations, or hyperparameters. Existing checkpoints remain load-compatible VASU-60M artifacts, and exact-resume behavior remains unchanged.
