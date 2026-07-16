# VASU-LM Training Log

This log records verified milestones. It intentionally does not estimate unrecorded metrics or mark planned work as complete.

## Hardware

- NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM
- Intel Core i5-13420H
- 16 GB RAM
- Windows

## VASU-31M cycle

### TinyStories pretraining

- Objective: validate end-to-end language-model training and learn basic English/story structure.
- Best recorded validation loss: 2.2923.
- Result: coherent short story-like continuations emerged.

### FineWeb-Edu 100k

- Tokens: 126,516,958.
- Train loss: 4.7837.
- Validation loss: 4.2497.
- Result: broader vocabulary and web-style text, with limited knowledge.

### FineWeb-Edu 1M

- Tokens: 1,271,317,605.
- Train loss: 3.526017.
- Validation loss: 3.411819.
- Resulting checkpoint: `checkpoints/fineweb_1m/best.pt`.
- Result: improved document-style fluency; repetition, hallucination, and topic drift remained.

### Instruction experiments

| Stage | Train loss | Validation loss | Resulting checkpoint |
| --- | ---: | ---: | --- |
| Alpaca | 2.653548 | 2.539819 | `checkpoints/instruct_fineweb/best.pt` |
| Masked Alpaca | 2.902693 | 2.797211 | `checkpoints/instruct_fineweb_masked/best.pt` |
| Masked Alpaca v2 | 2.638911 | 2.686126 | `checkpoints/instruct_fineweb_masked_v2/best.pt` |
| UltraChat | 2.633709 | 2.487310 | `checkpoints/ultrachat_fineweb/best.pt` |

The final fixed-prompt manual baseline for `ultrachat_fineweb` is **2.225 / 5**. VASU-31M is the stable instruction-tuned fallback and its experiment cycle is complete.

## VASU-60M milestones

### Architecture selection

- Selected configuration: 58,337,792 parameters.
- Model dimension: 512; layers: 10; heads: 8; SwiGLU hidden dimension: 2,048.
- Vocabulary: 32,000; context: 256.
- VASU-31M defaults were preserved.

### CUDA smoke test

- Batch size: 2.
- Sequence length: 256.
- AMP forward/backward training step: successful.
- Conclusion: the realistic micro-batch fits the RTX 4050 Laptop GPU.

### Tiny real-data test

- Train loss: 20.909606.
- Validation loss: 20.169777.
- Peak allocated CUDA memory: 1,576.58 MiB.
- Checkpoint save and resume: successful; `global_step` restored.

### FineWeb base-pretraining milestones

| Global step | Train loss | Validation loss | Preserved checkpoint |
| ---: | ---: | ---: | --- |
| 500 | 7.594220 | 7.719598 | `checkpoints/vasu_60m/fineweb_warmup/final_step_500.pt` |
| 4,000 | 4.915869 | 5.047358 | Not recorded here |
| 5,000 | 4.771692 | 4.876656 | `checkpoints/vasu_60m/milestones/fineweb_step_5000.pt` |
| 10,060 | 4.298818 | 4.392632 | `checkpoints/vasu_60m/milestones/fineweb_step_10060.pt` |
| 54,060 | 3.616769 | 3.613814 | `checkpoints/vasu_60m/milestones/fineweb_step_54060.pt` |

At step 54,060, the maximum reported GPU temperature was 74°C and peak allocated CUDA memory was 1,376.37 MiB. Raw base continuations showed improved grammar and sentence structure, while repetition, factuality, semantic consistency, and long-range coherence remained weak.

Base pretraining for this cycle later completed at the preserved step-150,000 checkpoint. The first controlled VASU-60M instruction experiment is recorded below.

### Standard Alpaca instruction tuning

- Objective: run one standard full-loss Alpaca epoch before any masked or UltraChat experiment.
- Starting checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt`.
- Dataset: `data/processed/instruct/alpaca.bin`.
- Configuration: batch size 2, gradient accumulation 16, context 256, learning rate 5e-6, weight decay 0.01, AMP, gradient clipping 1.0.
- First session: safely stopped at 88°C and `global_step 150129`; resume checkpoint saved.
- Resume: restored successfully from `checkpoints/vasu_60m/alpaca/vasu.pt` and completed the same epoch.
- Final global step: 150,572.
- Final resumed-segment train loss: 2.743232.
- Validation loss: 2.674379.
- Best checkpoint: `checkpoints/vasu_60m/alpaca/best.pt`.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Maximum temperature during the successful completion session: 74°C.
- Training stability: stable after cooldown; thermal protection worked during the first session.
- Additional Alpaca epochs: not run.
- UltraChat: not started.

### Masked Alpaca v2 instruction tuning

- Objective: compare standard full-loss Alpaca against boundary-preserving assistant-only supervision from the same FineWeb step-150,000 base.
- Starting checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt`; the standard Alpaca checkpoint was not used as a starting point.
- Dataset format: fixed 257-token records yielding 256-token input/target sequences; complete examples end with supervised EOS, prompt/header and PAD tokens have mask 0, and response/EOS tokens have mask 1.
- Dataset records: 23,927 from 52,002 source rows.
- Dataset tokens: 6,149,239 total; 3,320,371 assistant-loss tokens; 1,492,629 prompt tokens; 1,336,239 padding tokens; 50,643 supervised EOS tokens.
- Data-quality accounting: 1,316 truncated responses without false EOS, 43 dropped examples, and 28 malformed examples.
- Configuration: one epoch, batch size 2, accumulation 16, context 256, learning rate 5e-6, weight decay 0.01, AMP, gradient clipping 1.0.
- Dry run: finite loss 2.341496, finite gradients, temporary checkpoint save/load and `global_step=150001` resume verification passed; peak CUDA memory 1,162.22 MiB.
- Final global step: 150,711.
- Train supervised-token loss: 2.852573.
- Validation supervised-token loss: 2.732960 across 166,527 validation supervised tokens.
- Training supervised tokens processed: 3,153,844; zero-supervision batches: 0.
- Best checkpoint: `checkpoints/vasu_60m/alpaca_masked_v2/best.pt`.
- Main resumable checkpoint: `checkpoints/vasu_60m/alpaca_masked_v2/vasu.pt`.
- Peak allocated CUDA memory: 1,374.88 MiB.
- Maximum GPU temperature: 86Â°C; no thermal stop occurred.
- Deterministic comparison: `evaluation/checkpoint_comparison_alpaca_masked_v2.txt` and `.json`; no manual score was assigned.
- Evaluation observation: outputs remain repetitive and factually unreliable. No improvement claim is made before manual review.
- Additional epochs: not run. UltraChat: not started.

## Operational lessons

- Unlimited periodic checkpoints filled disk space; bounded retention was introduced.
- Interrupted or incomplete checkpoint files were detected and skipped.
- Saves now write a temporary file and atomically promote it after successful serialization.
- Short block training and cooldowns made sustained laptop training thermally manageable.
- The block trainer stops automatically at 88°C.
- Milestone checkpoints are stored under `checkpoints/vasu_60m/milestones/`, outside operational retention.
- General trainer interruptions do not fully restore DataLoader sampler position; block training instead advances its FineWeb slice from `global_step`.

## FineWeb-Edu extension preparation

- Reason: at `global_step=150000`, the established offset is 1,228,800,000 tokens. The original 98% training region ends at 1,245,891,252, leaving only 2,086 unseen optimizer steps and making step 200,000 unreachable without more data.
- Original audit: `HuggingFaceFW/fineweb-edu`, implicit `default` config, `train`, streaming, no shuffle, first 1,000,000 rows. The retained raw rows match `CC-MAIN-2013-20` at the locally resolved revision, but the original script did not pin a commit.
- Original preprocessing: `text.strip()`, drop empty rows, append exact text `\n[EOS]\n`, tokenize with `assets/tokenizer.json`, and concatenate `uint16` IDs.
- Extension source: `HuggingFaceFW/fineweb-edu`, config `CC-MAIN-2025-26`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, split `train`, no shuffle.
- Deduplication: persistent SQLite normalized-text SHA-256 index, seeded from the retained original 1M-document JSONL. Exact original/extension and within-extension duplicates are rejected; semantic and near-duplicate overlap remains possible.
- 1M smoke: interruption/resume passed; 1,000,451 tokens, one exact cross-shard duplicate removed, validator passed.
- 10M smoke: 10,000,576 tokens, six exact cross-shard duplicates removed, validator passed.
- Final extension: 500,000,478 tokens, 1,000,000,956 bytes, SHA-256 `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e`.
- Documents: 379,450 seen; 379,247 written; 203 dropped as exact original-corpus duplicates; 0 malformed; 0 empty; 0 within-extension duplicates.
- Token range: 3 through 31,999. Tokenizer SHA-256: `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`.
- Manifest: `data/processed/pretrain/fineweb_manifest.json` preserves the original validation region `[1245891252, 1271317605)` and adds the extension as training-only.
- Capacity: step 200,000 maps to extension offset 392,508,748, leaving 107,491,730 logical training tokens after that step.
- Training status: not started. The training loader was intentionally not modified in this data-preparation task.

## FineWeb manifest-loader readiness validation

- Added `ManifestTokenDataset` as an additive, read-only multi-shard dataset; legacy `TextDataset` behavior remains available.
- Logical training order: original slice `[0, 1,245,891,252)` followed immediately by extension slice `[0, 500,000,478)`.
- Each sample advances by 256 logical tokens and reads 257 tokens for shifted next-token targets.
- The FineWeb block runner now constructs its training window from the manifest and uses `shuffle=False` for exact global-step progression.
- Validation remains isolated at `fineweb_1m.bin[1,245,891,252:1,271,317,605)`; the extension is never a validation source.
- Step 150,000 maps to logical offset 1,228,800,000 in the original shard.
- Step 200,000 maps to logical offset 1,638,400,000, extension offset 392,508,748.
- Available logical training tokens: 1,745,891,730; tokens remaining after step 200,000: 107,491,730.
- Production cross-boundary check read 100 original tokens followed by 157 extension tokens and matched manual memmap concatenation byte-for-byte.
- CUDA dry runs at an original-only location, across the shard boundary, and inside the extension all produced finite loss and finite gradients.
- Dry-run losses: 2.977941 original, 2.995886 boundary-crossing, and 3.285769 extension.
- Peak allocated CUDA memory: 1,376.42 MiB.
- Full test suite: 92 passed.
- Protected dataset, manifest, tokenizer, and step-150,000 checkpoint hashes remained unchanged.
- Status at readiness validation: no live continuation had started; the later one-block integration result is recorded below.

## FineWeb manifest live integration block: step 150,000 to 150,200

- Purpose: one controlled live integration test of manifest-aware sequential pretraining; no long-run wrapper was invoked.
- Resume state: operational `fineweb_blocks/step_150000.pt`, verified byte-identical to `milestones/fineweb_step_150000.pt` before execution.
- Starting global step: 150,000.
- Final global step: 150,200; exactly 200 optimizer steps completed.
- Starting logical offset: 1,228,800,000.
- Ending/next logical offset: 1,230,438,400.
- Tokens consumed: 1,638,400, entirely inside the remaining original training slice.
- Extension shard reached: no.
- Configuration: VASU-60M, sequence length 256, batch size 2, accumulation 16, learning rate 1e-4, weight decay 0.1, gradient clipping 1.0, AMP enabled.
- Train loss reported by the block runner: 3.350210.
- Validation: skipped because no 250-step validation milestone was crossed; no validation loss was invented.
- Final learning rate: 0.0001.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Maximum GPU temperature: 72°C; thermal stop did not occur.
- Elapsed process time: approximately 253.3 seconds.
- Resulting checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_150200.pt`.
- Checkpoint validation: reload, VASU-60M parameter count, optimizer, scheduler, global step, atomic completion, and finite-tensor checks passed.
- AMP scaler note: scaler state is recreated because it is not part of the established checkpoint schema.
- Protected milestone, tokenizer, manifest, original binary, and extension binary hashes remained unchanged.
- Automatic continuation: not started.

## Bounded FineWeb continuation: step 150,200 to 152,000

- Purpose: finish only the remaining authorized pre-boundary continuation in nine independently checked 200-step executions; the long-running PowerShell wrapper was not invoked.
- Starting checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_150200.pt`.
- Starting global step and logical offset: 150,200 and 1,230,438,400.
- Final global step and next logical offset: 152,000 and 1,245,184,000.
- Optimizer steps completed: exactly 1,800.
- Tokens consumed: exactly 14,745,600.
- Original-to-extension boundary: 1,245,891,252; the run stopped 707,252 tokens before it, so the extension shard was not used.
- Configuration remained unchanged: VASU-60M, sequence length 256, batch size 2, accumulation 16, learning rate 1e-4, weight decay 0.1, gradient clipping 1.0, and AMP.
- Final block train loss: 3.359692.
- Fixed validation losses recorded at block ends after milestone crossings: 3.348395 (step 150,400), 3.349373 (150,600), 3.354896 (150,800), 3.352015 (151,000), 3.345056 (151,400), 3.339874 (151,600), 3.342773 (151,800), and 3.364378 (152,000). Validation was correctly skipped at step 151,200.
- Maximum GPU temperature across the bounded executions: 73 C; no thermal stop occurred.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Combined Python-process elapsed time: approximately 1,459.4 seconds (24 minutes 19.4 seconds).
- Final checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_152000.pt`.
- Final checkpoint SHA-256: `7301ac3a697f11cdc52c01e04bcf13ba204c2f138f8b554a18fb71ab45f6df4a`.
- Validation: checkpoint schema, global step, strict model load, optimizer load, scheduler load, finite stored tensors, atomic completion, parameter count 58,337,792, and final learning rate 0.0001 all passed on CPU.
- Preservation: the step-150,000 milestone, step-150,200 integration checkpoint, tokenizer, manifest, original shard, and extension shard retained their verified hashes.
- Hard-stop behavior: the runner is capped at 152,000 and returns without training when that target is already reached.

## Manifest-aware FineWeb continuation: step 152,000 to 200,000

- Purpose: cross from the exhausted original training slice into the validated extension and reach an exact, authorized base-pretraining milestone without using the fixed original validation region for training.
- Starting operational state for robust orchestration: global step 152,400; earlier preserved milestones at 150,000, 152,000, and 152,200 remained outside retention.
- Configuration remained unchanged: VASU-60M, sequence length 256, batch size 2, accumulation 16, learning rate 1e-4, weight decay 0.1, gradient clipping 1.0, and AMP.
- Orchestration: one synchronous Python process per block, normally 200 optimizer steps, periodic checkpoints every 10 steps, and a 120-step final block from 199,880 to 200,000.
- Safety: CPU checkpoint validation, strict architecture/state checks, SHA-256 recording, atomic saves, corrupt-file filtering, bounded retention, 10 GiB disk floor, conflict detection, 88°C thermal stop, and positive-progress enforcement.
- Logs: per-block output under `logs/vasu_60m_200k/` and append-only summary records in `logs/vasu_60m_200k/run_summary.jsonl`.
- Recovery: the orchestration host was externally terminated during the 179,800-to-180,000 execution. The valid periodic `step_179880.pt` preserved 80 completed optimizer steps and passed CPU validation before resumption. No trainer survived the interruption.
- Completed block records after step 152,400: 238, representing 47,520 logged steps; the recovered periodic state accounts for the other 80 steps in the 47,600-step interval.
- Thermal stops: 0; maximum recorded GPU temperature: 82°C.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Final global step and logical offset: 200,000 and 1,638,400,000.
- Final extension offset: 392,508,748; tokens remaining after the target: 107,491,730.
- Final block train loss: 3.101754.
- Final validation loss: 3.356130.
- Final checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_200000.pt`.
- Preserved milestone: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Final checkpoint and milestone SHA-256: `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f`.
- Final validation: model/optimizer/scheduler state, internal `global_step=200000`, strict VASU-60M shapes, filename agreement, and finite tensors passed on CPU.
- Preservation: original and extension binaries, manifest, tokenizer, and earlier milestones retained their verified hashes; no `.tmp` checkpoint remained.
- Instruction status: no additional Alpaca epoch and no UltraChat training were started.

## Masked Alpaca v3 historical preparation from FineWeb step 200,000

- Date: 2026-07-16.
- Experiment: `vasu_60m_alpaca_masked_v3_from_200k`.
- Objective: controlled replication of masked Alpaca v2 with only the base checkpoint changed.
- Base checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`; internal step, VASU-60M shapes, complete state, and finite tensors validated.
- History checkpoints: step 200,400 and step 200,590 are preserved for recovery/history and are not used for this experiment.
- Dataset: `data/processed/instruct/alpaca_masked_v2.bin`, unchanged.
- Mask: `data/processed/instruct/alpaca_masked_v2_mask.bin`, unchanged.
- Records: 23,927 total; deterministic split of 22,730 train and 1,197 validation records.
- Mask semantics: stored current-token masks shift to `mask[1:]`; prompt/PAD targets are zero and assistant/EOS targets are one.
- Configuration: one epoch, batch size 2, accumulation 16, effective batch 32 records, context 256, learning rate 5e-6, weight decay 0.01, clipping 1.0, AMP, seed 42, CosineAnnealingLR with `T_max=1`, and saves every 100 optimizer steps.
- Output: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/`, isolated from all earlier Alpaca, FineWeb, milestone, and UltraChat checkpoints.
- Dry-run batch: input/target/mask shapes `(2, 256)`, 258 supervised targets, finite masked loss 2.146862.
- Dry-run side effects: no optimizer step, scheduler step, checkpoint, or output directory creation.
- Focused tests: 35 passed across v2/v3 masked-Alpaca coverage.
- Full suite: 140 passed.
- Status at this historical preparation point: the controlled epoch had not started. Its later completion is recorded in the next section.

## Masked Alpaca v3 completion and UltraChat masked v2 readiness

- Masked Alpaca v3 completed from the preserved FineWeb step-200,000 base.
- Best checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.
- Final global step: 200,711.
- Train assistant-token loss: 2.627628.
- Validation assistant-token loss: 2.522585.
- The best checkpoint passed internal-step, strict VASU-60M shape, complete-state, and finite-tensor validation.
- New UltraChat format: fixed 257-token records; canonical `User:`/`Assistant:` turns; prompt and padding mask 0; assistant response and complete-response EOS mask 1.
- Dataset: 19,844 records, 5,099,908 tokens, 3,133,959 supervised tokens, 5,803 supervised EOS tokens, and 14,711 responses truncated without false EOS.
- Split: 18,851 train records and 993 validation records.
- Dry run: one forward-only CUDA batch passed with finite loss 2.688121, 250 supervised targets, and approximately 470.43 MiB peak allocated memory.
- Calculated one-epoch length: 590 optimizer steps; target global step 201,301.
- Full test suite: 160 passed.
- Training status: UltraChat has not started; a separate explicit `--train` invocation is required.

## UltraChat masked v2 completed experimental epoch

- Experiment: `vasu_60m_ultrachat_masked_v2_from_alpaca_v3`.
- Starting checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.
- Starting global step: 200,711.
- Target/final global step: 201,301.
- Optimizer steps: 590.
- Resulting checkpoint: `checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt`.
- Role: completed experimental branch; preserved for comparison and rollback, not promoted as the default assistant.
- Selection result: expanded 40-prompt greedy and sampled comparisons favored Alpaca v3 for repetition control and several task categories.
- Default assistant checkpoint remains `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.
- No additional Alpaca or UltraChat epoch is authorized.
