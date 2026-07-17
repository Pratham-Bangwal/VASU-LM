# VASU Project Status

## Next-generation planning status

An evidence-based next-generation plan is recorded in `docs/VASU_NEXT_PLAN.md`. The recommended path is a capability-focused continuation of the preserved VASU-60M FineWeb step-200,000 **base** checkpoint, gated first by small data-mixture ablations. The planning target is approximately 1.2B additional new token positions at the existing 58,337,792-parameter architecture and 256-token context.

This is planning only. No architecture, runner, tokenizer, dataset, checkpoint, or training process has been changed or authorized. A future approximately 100M model remains conditional on proving the new data mixture and curriculum with the existing 60M model.

### Factual pilot source gate

The factual pilot now passes the metadata approval gate using the official
`wikimedia/wikipedia` distribution, configuration `20231101.en`, split
`train`, pinned at commit
`e6057dc557255a03c9c3c47ceab0eb44353b1bc5`. The source is approved only for
bounded acquisition and preparation under the recorded CC BY-SA/GFDL
attribution and redistribution obligations. No Wikimedia data has been
downloaded, prepared, tokenized, or used for training. The capability pilot
remains blocked by its mathematics, code, and reasoning source reviews.

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

### KV-cache inference status

- Typed, inference-only per-layer K/V cache implemented.
- Explicit causal prefill and one-token decode modes implemented.
- CPU and CUDA logit parity passed within `rtol=1e-4`, `atol=1e-5`.
- Greedy cached and uncached token IDs matched exactly for the required preferred-checkpoint prompts.
- EOS and maximum-context stopping parity passed.
- Full sampling history is retained for repetition penalty and sampling.
- Focused tests: 25 passed; full suite: 185 passed.
- Benchmark reports: `evaluation/kv_cache_benchmark_30_tokens.json` and `evaluation/kv_cache_benchmark_100_tokens.json`.
- Activation: `chat.py` exposes `USE_KV_CACHE = False`; uncached generation remains the default/reference path.

Measured CUDA performance showed lower cached peak allocation but no throughput improvement yet. At a 100-token limit, cached generation averaged 119.87 tokens/s versus 120.94 tokens/s uncached and used 236.89 MiB versus 258.86 MiB average peak allocation. A preallocated cache is the likely next optimization.

Compatibility remains unchanged: KV cache does not alter architecture parameters, trained weights, tokenizer, prompt templates, datasets, training behavior, model-state keys, or checkpoint schema.

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

## Deterministic automatic evaluation

The checkpoint comparison system now supports optional, task-specific automatic checks declared per prompt. The checks cover exact and accepted answers, keywords, formatting constraints, repetition, uncertainty/clarification behavior, and parseable Python constructs. Prompts without configured checks remain valid and are reported as not configured.

The expanded 40-prompt run automatically evaluates 36 prompts (61 checks) while preserving the existing manual-score fields:

- greedy Alpaca v3: 11/61 checks, prompt-average automatic score 0.167;
- greedy UltraChat v2: 13/61 checks, prompt-average automatic score 0.190;
- sampled Alpaca v3: 19/61 checks, prompt-average automatic score 0.292;
- sampled UltraChat v2: 20/61 checks, prompt-average automatic score 0.301.

These are heuristic task-compliance measurements, not general intelligence or reliability scores. Both checkpoints scored 0/5 automatic reasoning checks in both modes, and factual checks remained weak. Sampling improved several surface-form and repetition checks but does not establish improved correctness. Model architecture, checkpoints, datasets, tokenizer, prompt formatting, generation behavior, and manual-score semantics are unchanged.

Reports:

- `evaluation/checkpoint_comparison_expanded_40_greedy_auto.txt` and `.json`;
- `evaluation/checkpoint_score_summary_expanded_40_greedy_auto.json`;
- `evaluation/checkpoint_comparison_expanded_40_sampled_auto.txt` and `.json`;
- `evaluation/checkpoint_score_summary_expanded_40_sampled_auto.json`.

## Reproducible five-seed sampled evaluation

Sampled checkpoint comparison now supports a fixed single seed, a deterministic consecutive seed range, or an explicit ordered seed list. The official five-seed stability run used seeds 42-46 and produced 200 generations per checkpoint (400 total) without changing the sampling algorithm or its temperature/top-k/top-p settings.

- Alpaca v3: automatic prompt-average 0.279, automatic-score population standard deviation 0.371363, mean repetition ratio 0.356561, and 98/305 checks passed.
- UltraChat v2: automatic prompt-average 0.301, automatic-score population standard deviation 0.376426, mean repetition ratio 0.370466, and 106/305 checks passed.
- Both checkpoints: reasoning automatic average 0.000; factual-knowledge average 0.075; programming average 0.025.
- Relative category results: Alpaca was higher on definition and planning checks and had lower overall repetition. UltraChat was higher on conversation, instruction-following, formatting, and creative checks.

The standard deviations show substantial response-to-response variation. These task-specific automatic checks report stability and explicit constraint compliance, not general intelligence, factual correctness, or safety. Existing greedy reports, unseeded sampled reports, and single-seed flat JSON entries remain compatible.

Reports:

- `evaluation/checkpoint_comparison_expanded_40_sampled_5seed.txt`;
- `evaluation/checkpoint_comparison_expanded_40_sampled_5seed.json`;
- `evaluation/checkpoint_score_summary_expanded_40_sampled_5seed.json`.

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

## Bounded Wikimedia factual preparation pipeline

The approved `wikimedia/wikipedia` factual source now has a reusable,
interruption-safe pilot preparation pipeline. It is pinned to subset
`20231101.en`, split `train`, revision
`e6057dc557255a03c9c3c47ceab0eb44353b1bc5`, and one explicit Parquet shard.

Hard pilot limits are one shard, 10,000 inspected rows, 2,000 accepted
documents, 2,000,000 exact VASU-tokenizer tokens, and 1 GB downloaded. The
pipeline records provenance, explicit filter reasons, exact and bounded
near-duplicate checks, evaluation-prompt contamination evidence, hashes,
atomic progress, and deterministic resume/restart state. Generated raw,
interim, processed, and factual-manifest artifacts are ignored by Git.

Dry-run validation and all tests passed. A smaller acquisition smoke attempt
(100 rows, 20 accepted documents, 20,000 tokens) was attempted, but the pinned
file transfer remained at 0 bytes and was stopped. Therefore no Wikimedia
documents or tokens were prepared, the default pilot was not run, and no
training started. Cross-FineWeb document deduplication remains blocked because
the repository has no versioned document-level normalized-hash/signature
index; token binaries are not treated as a substitute.
