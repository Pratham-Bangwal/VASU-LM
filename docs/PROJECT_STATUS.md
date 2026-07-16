# VASU Project Status

## Project snapshot

- Framework: PyTorch
- Project type: educational and experimental language-model framework
- Active model: VASU-60M preferred experimental assistant
- Stable instruction-tuned fallback: VASU-31M
- Hardware: RTX 4050 Laptop GPU 6 GB, Intel i5-13420H, 16 GB RAM, Windows

## VASU-31M

Status: stable fallback and completed 31M experiment cycle.

Completed work:

- FineWeb pretraining;
- Alpaca instruction tuning;
- masked-Alpaca experiments;
- UltraChat tuning;
- fixed-prompt checkpoint comparison;
- manual evaluation baseline.

Current assistant checkpoint:

`checkpoints/ultrachat_fineweb/best.pt`

Manual baseline: **2.225 / 5**.

VASU-31M remains available as the stable historical fallback while VASU-60M is the active experimental assistant.

## VASU-60M

Status: **VASU-60M training phase complete.** Masked Alpaca v3 is the preferred assistant checkpoint. The completed UltraChat masked-v2 branch is preserved as experimental and is not promoted.

Current phase: stabilization, evaluation, documentation, and next-generation planning. No further Alpaca or UltraChat training is currently authorized.

Verified completed work:

- architecture planning completed;
- 58,337,792-parameter configuration validated;
- CPU construction and forward smoke test passed;
- CUDA forward/backward smoke test passed;
- batch size 2 with sequence length 256 passed;
- tiny real-data training and validation passed;
- checkpoint save/resume restored `global_step` successfully;
- resumable block-training pipeline validated;
- automatic thermal protection validated;
- atomic checkpoint saving and corruption filtering validated;
- bounded retention and low-disk protection validated;
- raw base-generation milestone evaluation established.

Current preserved base checkpoint:

`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`

Standard-Alpaca comparison checkpoint:

`checkpoints/vasu_60m/alpaca/best.pt`

Masked-v2 comparison checkpoint:

`checkpoints/vasu_60m/alpaca_masked_v2/best.pt`

Current masked-Alpaca-v3 checkpoint:

`checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`

Masked Alpaca v3 completed at global step 200,711 with assistant-token
train loss 2.627628 and validation loss 2.522585. The checkpoint was
revalidated as a finite 58,337,792-parameter VASU-60M state before the
UltraChat v2 dry run.

Experimental UltraChat masked-v2 checkpoint:

`checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt`

UltraChat masked v2 completed its controlled 590-step epoch from global step 200,711 to 201,301. Expanded greedy and sampled evaluation did not justify promotion.

`chat.py` now loads the preferred Alpaca masked-v3 checkpoint by default using `get_vasu_60m_config()`, strict model-state loading, explicit Alpaca prompt formatting, CUDA when available, and sampled decoding (`temperature=0.45`, `top_k=20`, `top_p=0.8`, inherited repetition penalty `1.1`).

Interactive smoke validation used `What is the internet? Explain it in simple words.` The preferred checkpoint loaded successfully, generated on CUDA, and exited cleanly. This confirms the entry point works; it is not a correctness assessment of the answer.

Compatibility impact: documentation and default-selection reporting do not alter the VASU-60M architecture, `assets/tokenizer.json`, checkpoint schema, FineWeb data/checkpoints, Alpaca data/checkpoints, UltraChat data/checkpoints, evaluation reports, or training runners.

| Alpaca metric | Value |
| --- | ---: |
| Training epochs | 1 |
| Final global step | 150,572 |
| Final resumed-segment train loss | 2.743232 |
| Validation loss | 2.674379 |
| Peak CUDA memory | 1,376.37 MiB |
| Maximum completion-session temperature | 74°C |
| Resume | Verified |

| Masked Alpaca v2 metric | Value |
| --- | ---: |
| Training epochs | 1 |
| Final global step | 150,711 |
| Train supervised-token loss | 2.852573 |
| Validation supervised-token loss | 2.732960 |
| Peak CUDA memory | 1,374.88 MiB |
| Maximum temperature | 86Â°C |
| Thermal stop | No |

## Completed base-pretraining operation

- batch size: 2;
- gradient accumulation: 16;
- sequence length: 256;
- learning rate: 1e-4;
- weight decay: 0.1;
- AMP: enabled;
- continuation orchestration block size: 200 optimizer steps, with an exact shorter final block;
- periodic save interval: 10 optimizer steps;
- automatic thermal stop: 88°C;
- normal block cooldown: 10 seconds;
- thermal cooldown in the long runner: 10 minutes;
- long-run time limit: 11 hours;
- hard continuation target: global step 200,000.

The controlled Alpaca run used batch size 2, gradient accumulation 16, sequence length 256, learning rate 5e-6, weight decay 0.01, AMP, gradient clipping 1.0, atomic saves, bounded retention, a 10 GiB free-disk requirement, and an 88°C thermal stop.

## Current quality assessment

The base checkpoint shows improved grammar and topic relevance but still has factual hallucination, repetition, semantic drift, and weak long-range planning. The first Alpaca evaluation raised a prompt-format concern. A repository and token-level audit proved that both the prepared standard-Alpaca data and the evaluator used `User: {instruction}\nAssistant:`; the suspected plain-prompt mismatch was not the root cause.

A corrected deterministic comparison now explicitly declares prompt formats, includes the VASU-60M FineWeb step-150,000 baseline, and honors checkpoint filtering. The Alpaca generations remain repetitive and frequently incorrect under the verified training template. No VASU-60M manual score has been assigned, and the corrected outputs have not been promoted as proof of improvement or regression.

Masked Alpaca v2 preserves fixed records, supervises only assistant response and terminating EOS targets, and masks prompt and padding tokens. Its deterministic outputs are more complete in some cases but remain repetitive, inaccurate, and instructionally unreliable. This observation is not a promotion or a scored improvement claim.

Corrected reports:

- `evaluation/checkpoint_comparison_corrected.txt`;
- `evaluation/checkpoint_comparison_corrected.json`;
- `evaluation/checkpoint_score_summary_corrected.json`.
- `evaluation/checkpoint_comparison_alpaca_masked_v2.txt`;
- `evaluation/checkpoint_comparison_alpaca_masked_v2.json`;
- `evaluation/checkpoint_score_summary_alpaca_masked_v2.json`.

## Historical pre-UltraChat decision gate

At this earlier decision point, deterministic standard and masked-v2 outputs remained weak, UltraChat had not started, and another Alpaca epoch was not run. The later UltraChat experiment and final model-selection decision are recorded in the current status above.

The original FineWeb training region was audited and found to support only step 152,086 without replay. A compatible training-only extension is now prepared and validated:

- source: `HuggingFaceFW/fineweb-edu`, `CC-MAIN-2025-26`;
- pinned revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- shard: `data/processed/pretrain/fineweb_extension_500m.bin`;
- actual tokens: 500,000,478;
- manifest: `data/processed/pretrain/fineweb_manifest.json`;
- original validation region: preserved unchanged;
- logical capacity through step 200,000: validated.

The explicitly reviewed multi-shard loader is implemented and dry-run validated. Bounded continuation advanced through the original-to-extension boundary and reached step 200,000 without entering the fixed validation slice.

## FineWeb continuation readiness

- `ManifestTokenDataset` validates and exposes the original training slice plus extension as one logical stream without rewriting either file.
- Cross-shard 257-token reads, including the production boundary, are verified.
- The block runner uses the manifest, fixed validation slice, global-step-derived offset, and sequential `shuffle=False` ordering.
- Step 150,000 offset: 1,228,800,000.
- Original-to-extension boundary: 1,245,891,252.
- Step 200,000 offset: 1,638,400,000 (extension offset 392,508,748).
- Capacity through step 200,000: verified, with 107,491,730 tokens remaining.
- Three isolated VASU-60M CUDA forward/backward checks passed; peak allocation was 1,376.42 MiB.
- Long-run continuation: completed to the authorized hard target of step 200,000 through one-process-at-a-time bounded blocks.

## Controlled FineWeb continuation integration

One explicitly bounded live block completed from global step 150,000 to 150,200. It consumed logical offsets `[1,228,800,000, 1,230,438,400)`, which remain inside the original unseen training slice; the extension shard was not reached.

- Train loss: 3.350210.
- Validation loss: not run because the block did not cross a 250-step validation milestone.
- Maximum temperature: 72°C.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_150200.pt`.
- Checkpoint reload and finite-state validation: passed.
- Status at that integration checkpoint: long-run continuation had not yet started; later continuation is recorded below.

## Bounded continuation to step 152,000

Nine separately observed 200-step executions continued from `block_final_step_150200.pt` to `block_final_step_152000.pt`. The hard target was reached exactly with 1,800 optimizer steps and 14,745,600 token positions consumed.

- Final logical offset: 1,245,184,000.
- Distance to extension boundary: 707,252 tokens.
- Extension reached: no.
- Final block train loss: 3.359692.
- Final validation loss: 3.364378.
- Maximum temperature: 73 C; thermal stops: none.
- Peak CUDA allocation: 1,376.37 MiB.
- Final checkpoint integrity and resume-state load: passed.
- Protected inputs and earlier checkpoints: unchanged.
- Automatic or long-running continuation: not started.

## Bounded manifest continuation to step 200,000

The boundary-crossing step-152,000 to step-152,200 block and the subsequent bounded continuation completed. The robust wrapper then continued from the validated step-152,400 state to the exact step-200,000 hard target.

- Final global step and logical offset: 200,000 and 1,638,400,000.
- Final extension offset: 392,508,748; remaining logical training tokens: 107,491,730.
- Final block train loss: 3.101754.
- Final validation loss: 3.356130.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Maximum recorded GPU temperature: 82°C; thermal stops: none.
- Final operational checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_200000.pt`.
- Preserved milestone: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Matching final SHA-256: `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f`.
- Final CPU validation: strict VASU-60M model load, optimizer state, scheduler state, `global_step=200000`, and finite stored tensors passed.
- Operational recovery: an external orchestration-host termination left a valid periodic step-179,880 checkpoint; continuation resumed from that internal step without replaying the completed token range.
- Logging: 238 completed block records cover 47,520 steps after 152,400; the recovered periodic checkpoint accounts for the additional 80 steps completed before the external host termination.
- Protected original shard, extension shard, manifest, tokenizer, and earlier milestone hashes remained unchanged.

Instruction tuning remains paused. Reaching step 200,000 is a base-pretraining milestone, not evidence that VASU-60M is assistant-ready.

## Masked Alpaca v3 historical preparation record

This section records the preparation state that preceded the now-completed masked-Alpaca-v3 experiment.

- Authoritative evaluated base: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Preserved recovery/history only: `fineweb_step_200400.pt` and `fineweb_step_200590.pt`; both validate as finite VASU-60M checkpoints but are not v3 initialization sources.
- Dataset and mask: unchanged masked-Alpaca-v2 token and mask files.
- Output isolation: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/` only.
- Comparison intent: hold the v2 formulation and hyperparameters fixed while changing only the FineWeb base checkpoint from step 150,000 to step 200,000.
- Dry run: strict base load, tokenizer/data/mask validation, loaders, optimizer, scheduler, and one forward-only masked batch passed with finite loss 2.146862 and 258 supervised targets.
- Dry-run writes: none; the v3 output directory was not created.
- Test status: 140 tests passed.
- Status at the time of this preparation record: training had not started. The completed v3 result and current UltraChat-v2 readiness are recorded above.
