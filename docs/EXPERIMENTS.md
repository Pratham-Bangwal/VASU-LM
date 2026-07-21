# VASU Experiments

Unknown or unavailable values are explicitly marked rather than inferred.

## VASU-60M manifest-aware FineWeb loader validation

- Objective: prove safe sequential continuation from the original FineWeb training slice into the validated 500M-token extension without concatenating files or entering validation data.
- Starting checkpoint used for dry runs: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt` (read-only).
- Data configuration: `data/processed/pretrain/fineweb_manifest.json`, uint16 memmaps, sequence length 256, batch size 2 for dry runs.
- Logical strategy: sample `i` starts at `logical_start + i * 256` and reads 257 tokens for shifted targets.
- Boundary result: a 257-token read from logical offset 1,245,891,152 contained exactly 100 original tokens and 157 extension tokens and matched manual slices.
- Original-shard dry-run loss: 2.977941; finite gradients.
- Boundary-crossing dry-run loss: 2.995886; finite gradients.
- Extension-shard dry-run loss: 3.285769; finite gradients.
- Peak allocated CUDA memory: 1,376.42 MiB.
- Validation isolation: passed; only the fixed original validation slice is exposed.
- Test result: 92 passed.
- Resulting checkpoint: none; the validation saved no checkpoint and performed no optimizer step.
- Conclusion: loader and step-200,000 capacity are technically ready.
- Decision: do not start continuation training in this implementation task.

## VASU-60M FineWeb step-150,200 live integration

- Objective: validate one complete live optimizer/checkpoint cycle using the manifest-aware loader without starting long-running continuation.
- Starting checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt`; the operational step-150,000 file used by the runner was byte-identical.
- Configuration: VASU-60M, batch size 2, accumulation 16, sequence length 256, learning rate 1e-4, weight decay 0.1, gradient clipping 1.0, AMP, 200-step hard target.
- Starting step/offset: 150,000 / 1,228,800,000.
- Final step/offset: 150,200 / 1,230,438,400.
- Train loss: 3.350210.
- Validation loss: unavailable; validation was correctly skipped because no 250-step milestone was crossed.
- Peak allocated CUDA memory: 1,376.37 MiB.
- Maximum temperature: 72°C; no thermal stop.
- Resulting checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_150200.pt`.
- Checkpoint result: global step, model, optimizer, scheduler, finite tensors, and atomic finalization validated.
- Data result: all 1,638,400 consumed tokens came from the original training slice; extension use remains untested in live optimizer training.
- Decision: stop after one block and require a separate manual continuation decision.

## VASU-31M FineWeb-Edu 1M pretraining

- Objective: build a stronger general-language base model.
- Configuration: VASU-31M, 256-token context, FineWeb-Edu 1M processed corpus.
- Starting checkpoint: earlier VASU-31M pretrained state; exact path not recorded here.
- Resulting checkpoint: `checkpoints/fineweb_1m/best.pt`.
- Train loss: 3.526017.
- Validation loss: 3.411819.
- Conclusion: document-style fluency improved; factuality and repetition remained weak.
- Decision: proceed to controlled instruction tuning.

## VASU-31M Alpaca

- Objective: teach single-turn instruction following.
- Starting checkpoint: `checkpoints/fineweb_1m/best.pt`.
- Resulting checkpoint: `checkpoints/instruct_fineweb/best.pt`.
- Train loss: 2.653548.
- Validation loss: 2.539819.
- Conclusion: assistant-like answers emerged but remained unreliable.
- Decision: preserve as the main Alpaca checkpoint and test masking/chat tuning.

## VASU-31M masked Alpaca

- Objective: apply loss only to assistant-response targets.
- Starting checkpoint: `checkpoints/fineweb_1m/best.pt`.
- Resulting checkpoint: `checkpoints/instruct_fineweb_masked/best.pt`.
- Train loss: 2.902693.
- Validation loss: 2.797211.
- Conclusion: mask alignment was correct, but generation quality was worse than unmasked Alpaca.
- Decision: retain for diagnostics; do not promote.

## VASU-31M masked Alpaca v2

- Objective: use masked Alpaca as cleanup after standard Alpaca.
- Starting checkpoint: `checkpoints/instruct_fineweb/best.pt`.
- Resulting checkpoint: `checkpoints/instruct_fineweb_masked_v2/best.pt`.
- Train loss: 2.638911.
- Validation loss: 2.686126.
- Conclusion: some structure improved, but factual errors and semantic drift remained.
- Decision: compare with UltraChat rather than continue Alpaca blindly.

## VASU-31M UltraChat

- Objective: improve assistant tone and conversational response patterns.
- Starting checkpoint: `checkpoints/instruct_fineweb/best.pt`.
- Resulting checkpoint: `checkpoints/ultrachat_fineweb/best.pt`.
- Train loss: 2.633709.
- Validation loss: 2.487310.
- Conclusion: most assistant-like checkpoint in the fixed comparison, though factuality and reasoning remained weak.
- Decision: select as the stable VASU-31M assistant fallback.

## VASU-31M manual evaluation baseline

- Objective: establish a repeatable instruction-model baseline.
- Checkpoint: `checkpoints/ultrachat_fineweb/best.pt`.
- Evaluation: eight fixed prompts scored for relevance, factuality, instruction following, fluency, and repetition control.
- Result: 2.225 / 5.
- Decision: future VASU-60M instruction checkpoints must be compared against this baseline.

## VASU-60M architecture feasibility

- Objective: select a model near 60M parameters that is realistic on 6 GB VRAM.
- Configuration: dim 512, 10 layers, 8 heads, hidden dimension 2,048, context 256.
- Parameter count: 58,337,792.
- Starting/resulting checkpoint: not applicable.
- Conclusion: selected as the opt-in VASU-60M configuration while preserving VASU-31M defaults.

## VASU-60M CUDA smoke test

- Objective: verify realistic forward/backward feasibility.
- Configuration: batch size 2, sequence length 256, AMP enabled.
- Result: successful.
- Train/validation loss: not used as model-quality metrics for this smoke test.
- Decision: proceed to a tiny real-data save/resume test.

## VASU-60M tiny real-data test

- Objective: verify FineWeb loading, training, validation, saving, and resume.
- Resulting checkpoint: `checkpoints/vasu_60m/tiny_test/vasu.pt`.
- Train loss: 20.909606.
- Validation loss: 20.169777.
- Peak CUDA memory: 1,576.58 MiB.
- Conclusion: real-data training and resume succeeded.
- Decision: proceed to controlled warmup and block training.

## VASU-60M block pretraining

- Objective: sustain FineWeb pretraining within laptop thermal and disk constraints.
- Configuration: batch size 2, accumulation 16, context 256, learning rate 1e-4, weight decay 0.1, AMP, 100-step blocks.
- Starting checkpoint: `checkpoints/vasu_60m/fineweb_warmup/final_step_500.pt` for the initial block run; later runs resume the latest valid block checkpoint.
- Resulting checkpoints: operational checkpoints under `checkpoints/vasu_60m/fineweb_blocks/`, with selected milestones copied manually to the milestone directory.
- Conclusion: thermal stopping, atomic saves, corruption filtering, retention, and global-step-based data progression are operational.
- Decision: continue toward step 100,000 in bounded sessions.

## VASU-60M step-5,000 evaluation

- Checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_5000.pt`.
- Train loss: 4.771692.
- Validation loss: 4.876656.
- Output: `evaluation/vasu_60m_base_step_5000.txt`.
- Conclusion: English-like raw continuations emerged but remained incoherent and repetitive.
- Decision: continue base pretraining.

## VASU-60M step-10,060 evaluation

- Checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_10060.pt`.
- Train loss: 4.298818.
- Validation loss: 4.392632.
- Output: `evaluation/vasu_60m_base_step_10060.txt`.
- Conclusion: base quality remained insufficient for an instruction-tuning decision.
- Decision: continue base pretraining.

## VASU-60M step-54,060 evaluation

- Checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_54060.pt`.
- Train loss: 3.616769.
- Validation loss: 3.613814.
- Sampled output: `evaluation/vasu_60m_base_step_54060.txt`.
- Optional greedy output: `evaluation/vasu_60m_base_step_54060_greedy.txt`.
- Conclusion: grammar and sentence structure improved; repetition, factuality, semantic consistency, and long-range coherence remain weak.
- Decision: continue base pretraining to step 100,000 before considering instruction tuning.

## VASU-60M standard Alpaca

- Objective: perform the first controlled instruction-tuning experiment using the standard full-loss Alpaca path.
- Configuration: one epoch, batch size 2, accumulation 16, context 256, learning rate 5e-6, weight decay 0.01, AMP, gradient clipping 1.0.
- Starting checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt`.
- Dataset: `data/processed/instruct/alpaca.bin`; masked Alpaca files were not used.
- Resumable checkpoint: `checkpoints/vasu_60m/alpaca/vasu.pt`.
- Resulting checkpoint: `checkpoints/vasu_60m/alpaca/best.pt`.
- First session: thermal protection stopped safely at 88°C and global step 150,129.
- Resume: successful; the same epoch continued from the saved Alpaca checkpoint.
- Final global step: 150,572.
- Final resumed-segment train loss: 2.743232.
- Validation loss: 2.674379.
- Peak CUDA memory: 1,376.37 MiB.
- Maximum temperature during completion: 74°C.
- Conclusion: the one-epoch standard Alpaca run completed stably after one protected thermal interruption.
- Initial evaluation concern: the first comparison was suspected of using plain prompts, but the evaluator's generation helper was implicitly applying `User: {instruction}\nAssistant:`.
- Verified training template: `User: {instruction}`, optional input on the next line, then `Assistant: {response}\n`; no BOS, EOS, `###` headers, or special separators were serialized.
- Evaluation defects found: `--checkpoints` was ignored, the VASU-60M FineWeb step-150,000 baseline was absent, and sampling was stochastic.
- Corrected evaluation: shared prompt formatting, explicit per-checkpoint prompt formats, functional checkpoint filtering, and greedy deterministic mode.
- Corrected reports: `evaluation/checkpoint_comparison_corrected.txt`, `evaluation/checkpoint_comparison_corrected.json`, and `evaluation/checkpoint_score_summary_corrected.json`.
- Corrected-output observation: the Alpaca checkpoint remains repetitive and often factually or instructionally incorrect under the verified template. No new manual score was assigned.
- Next decision: review corrected outputs before considering UltraChat; do not add another Alpaca epoch automatically. VASU-60M UltraChat has not started.

## VASU-60M masked Alpaca v2

- Objective: test assistant-only loss with explicit example boundaries and supervised response EOS, starting from the same base as standard Alpaca.
- Starting checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_150000.pt`.
- Resulting checkpoint: `checkpoints/vasu_60m/alpaca_masked_v2/best.pt`.
- Dataset: `alpaca_masked_v2_record_packed_eos_v1`, fixed 257-token records, 23,927 records, 6,149,239 stored tokens, and 3,320,371 assistant-loss tokens.
- Packing: multiple complete examples may share a record only behind supervised EOS; overlong responses are truncated without false EOS; PAD is masked.
- Data quality: 1,316 truncated, 43 dropped, and 28 malformed examples out of 52,002 source rows.
- Configuration: one epoch, batch size 2, accumulation 16, context 256, learning rate 5e-6, weight decay 0.01, AMP, clipping 1.0.
- Final global step: 150,711.
- Train supervised-token loss: 2.852573.
- Validation supervised-token loss: 2.732960.
- Peak CUDA memory: 1,374.88 MiB; maximum temperature: 86Â°C; thermal stop: none.
- Resume: temporary dry-run checkpoint restored successfully; the full epoch completed without interruption. Hardened resume remains available through `vasu.pt`.
- Evaluation: deterministic four-checkpoint reports are `evaluation/checkpoint_comparison_alpaca_masked_v2.txt`, `.json`, and `evaluation/checkpoint_score_summary_alpaca_masked_v2.json`.
- Conclusion: engineering and mask-alignment objectives passed, but generations remain repetitive and unreliable. No manual score or improvement claim was assigned.
- Decision: manually review/score before considering any separately approved UltraChat experiment. Do not run a second Alpaca epoch automatically.

## VASU-60M FineWeb-Edu extension

- Objective: add enough compatible unseen training data to make a controlled step-150,000 to step-200,000 continuation possible without entering the fixed original validation region.
- Original capacity finding: step 150,000 maps to token 1,228,800,000; the original training boundary is 1,245,891,252; only 2,086 additional optimizer steps remained.
- Source audit: the original unshuffled default stream's first million rows are inside `CC-MAIN-2013-20`. Its script did not record a revision, although retained rows match the locally resolved pinned repository state.
- Extension source: `HuggingFaceFW/fineweb-edu`, `CC-MAIN-2025-26`, `train`, pinned at `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Compatibility: identical tokenizer, `str.strip()` normalization, empty-row filtering, literal `\n[EOS]\n` separator, and `uint16` serialization.
- Exact deduplication: normalized-text SHA-256 hashes persisted in SQLite and seeded from the retained original JSONL. This removes exact matches but cannot prove absence of semantic or near-duplicate overlap across crawls.
- 1M smoke result: 1,000,451 tokens; controlled interruption and deterministic resume passed; validation passed.
- 10M smoke result: 10,000,576 tokens; validation passed.
- Final result: 500,000,478 tokens across 379,247 accepted documents; 203 exact original-corpus duplicates rejected; 0 malformed, empty, or within-extension duplicate documents.
- Resulting shard: `data/processed/pretrain/fineweb_extension_500m.bin`.
- Metadata: `data/processed/pretrain/fineweb_extension_500m_metadata.json`.
- Manifest: `data/processed/pretrain/fineweb_manifest.json`.
- Output SHA-256: `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e`.
- Capacity conclusion: step 200,000 maps to extension offset 392,508,748 and is within the logical training capacity; original validation remains fixed.
- Decision: preserve the validated data artifacts. Do not train until the multi-shard loader is separately implemented and reviewed.

## VASU-60M bounded pre-boundary continuation to step 152,000

- Objective: continue from the validated step-150,200 integration checkpoint to an exact hard target without entering the extension shard.
- Starting checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_150200.pt`.
- Resulting checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_152000.pt`.
- Configuration: batch size 2, accumulation 16, context 256, learning rate 1e-4, weight decay 0.1, gradient clipping 1.0, AMP, 200-step executions, and periodic saves every 10 steps.
- Scope: nine manually invoked and individually checked processes; no long-running PowerShell loop.
- Result: exactly 1,800 optimizer steps and 14,745,600 token positions, ending at logical offset 1,245,184,000.
- Shard result: all reads remained in the original training region; 707,252 tokens remained before the extension boundary.
- Final block train loss: 3.359692.
- Final validation loss: 3.364378.
- Operational result: no thermal stops, maximum 73 C, peak allocated CUDA memory 1,376.37 MiB, and approximately 1,459.4 seconds combined process time.
- Checkpoint result: CPU strict model load, optimizer/scheduler restoration, finite-state scan, schema validation, and SHA-256 recording passed.
- Conclusion: the hard target and pre-boundary data progression are correct; this run does not test a cross-shard training batch.
- Decision: stop at 152,000. A step-152,000 to step-152,200 boundary-crossing block requires separate authorization.

## VASU-60M manifest continuation to step 200,000

- Objective: complete the authorized unseen-token continuation across the original-to-extension boundary while preserving the fixed original validation slice.
- Starting checkpoint for robust long orchestration: validated operational state at step 152,400; preserved step-150,000, step-152,000, and step-152,200 milestones were not modified.
- Resulting operational checkpoint: `checkpoints/vasu_60m/fineweb_blocks/block_final_step_200000.pt`.
- Resulting milestone: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Configuration: batch size 2, accumulation 16, context 256, learning rate 1e-4, weight decay 0.1, clipping 1.0, AMP, 200-step bounded executions, and periodic saves every 10 steps.
- Data result: final logical offset 1,638,400,000, equal to extension offset 392,508,748; the original validation slice remained isolated.
- Final block train loss: 3.101754.
- Final validation loss: 3.356130.
- Operational result: 0 thermal stops, 82°C maximum recorded temperature, and 1,376.37 MiB peak allocated CUDA memory.
- Recovery result: one external host termination was recovered from valid periodic step 179,880; CPU validation proved finite complete state before resuming.
- Integrity result: final operational and milestone checkpoints share SHA-256 `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f` and load with complete model, optimizer, and scheduler state at internal step 200,000.
- Protected artifacts: both token shards, the manifest, tokenizer, and prior milestone checkpoints retained their verified hashes.
- Conclusion: bounded manifest-aware pretraining reached step 200,000 exactly and preserved checkpoint/data compatibility.
- Decision: stop base continuation at the authorized milestone. Do not infer assistant quality or begin UltraChat from this operational result alone.

## VASU-60M masked Alpaca v3 historical preparation from step 200,000

- Objective: isolate the effect of a stronger base checkpoint by replicating masked Alpaca v2 and changing only the initialization source.
- Starting checkpoint: authoritative evaluated base `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Recovery/history checkpoints: preserved steps 200,400 and 200,590; neither is eligible for v3 initialization.
- Dataset/mask: unchanged `alpaca_masked_v2.bin` and `alpaca_masked_v2_mask.bin` with format `alpaca_masked_v2_record_packed_eos_v1`.
- Configuration: identical to v2—one epoch, batch size 2, accumulation 16, context 256, learning rate 5e-6, weight decay 0.01, clipping 1.0, AMP, seed 42, 95/5 deterministic record split, CosineAnnealingLR `T_max=1`, and 100-step checkpoint cadence.
- Isolation: all possible v3 writes are constrained to `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/`; resume discovery searches only that directory.
- Base validation: step 200,000, finite tensors, strict VASU-60M state shape/key match, tokenizer vocabulary 32,000, and strict model load passed.
- Dataset validation: token/mask lengths, 257-token records, binary masks, prompt/PAD masking, assistant target presence, supervised EOS count, metadata aggregates, and tokenizer IDs passed.
- Dry-run result: finite masked loss 2.146862 on a forward-only batch with 258 supervised targets; optimizer/scheduler were constructed but not stepped.
- Resulting checkpoint: none; the dry run wrote no files.
- Test result: 35 focused tests and 140 full-suite tests passed.
- Conclusion: the experiment is technically ready as a controlled base-checkpoint comparison.
- Decision at preparation time: do not claim training results and do not start UltraChat. The later completed v3 result is recorded in the UltraChat-v2 preparation entry below.

## VASU-60M UltraChat masked v2 preparation

- Objective: prepare an EOS-supervised, boundary-aware UltraChat comparison starting only from the completed masked-Alpaca-v3 checkpoint.
- Starting checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`, global step 200,711; train loss 2.627628 and validation loss 2.522585 from the completed v3 run.
- Source: deterministic prefix of `data/raw/instruct/ultrachat.jsonl`; source revision was not recorded locally and remains `null` in metadata.
- Format: each user/assistant pair uses the shared `User: ...\nAssistant:` prompt boundary, response tokens are supervised, and a real complete response ends with supervised EOS.
- Packing: fixed 257-token records. Complete turns may share a record only across supervised EOS. Overlong responses start fresh, are truncated to record capacity, receive no false EOS, and terminate the conversation contribution.
- Dataset result: 19,844 records / 5,099,908 tokens; 18,851 train and 993 validation records; 3,133,959 assistant/EOS loss tokens; 5,803 EOS tokens; 14,711 truncated turns; 357,921 padding tokens.
- Validation: SHA-256, aggregate metadata, mask values, PAD masking, semantic response boundaries, EOS/truncation counts, target shifting, and split isolation passed.
- Training configuration prepared: one epoch, batch 2, accumulation 16, context 256, learning rate 2e-6, weight decay 0.01, clipping 1.0, AMP, thermal stop 88 C, atomic saves, bounded retention, and 10 GiB disk floor.
- Dry run: finite forward-only loss 2.688121; no optimizer step and no checkpoint write.
- Calculated target: 590 optimizer steps, from global step 200,711 to 201,301.
- Test result: 20 focused tests and 160 full-suite tests passed.
- Decision: do not start UltraChat automatically. Review the high truncation rate and obtain explicit authorization for the one-epoch run.

## Final VASU-60M assistant comparison: Alpaca v3 vs UltraChat masked v2

- Objective: select the default VASU-60M assistant checkpoint after both controlled instruction branches completed.
- Checkpoints compared:
  - `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`;
  - `checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt`.
- Evaluation coverage: the same expanded 40-prompt suite under greedy decoding and sampled decoding.
- Repetition: Alpaca v3 had the lower average repetition under both decoding modes.
- Category behavior: Alpaca v3 was relatively stronger in reasoning, programming, planning, uncertainty, and instruction following. This is a relative comparison, not evidence of general correctness.
- UltraChat behavior: the branch added some conversational variation but regressed in deterministic stability and several evaluated task categories.
- Reliability finding: neither checkpoint is dependable for factual knowledge, arithmetic, exact formatting, uncertainty handling, reasoning, or programming correctness.
- Selection: promote Alpaca masked v3 as the default experimental VASU-60M assistant checkpoint; retain UltraChat masked v2 as an experimental comparison only.
- Additional UltraChat epoch: rejected because the completed branch did not improve the principal benchmark weaknesses and showed stability/category regressions. More training could amplify those behaviors without evidence of a likely benefit.
- Current decision: no further Alpaca or UltraChat training is authorized. Continue with stabilization, evaluation, documentation, and next-generation planning.

## Expanded 40-prompt automatic-check baseline

- Objective: add reproducible task-specific checks to the existing greedy and sampled checkpoint comparison without replacing manual review.
- Compared checkpoints: Alpaca masked v3 from FineWeb step 200,000 and UltraChat masked v2 from Alpaca v3.
- Coverage: 36 of 40 prompts have meaningful deterministic checks, totaling 61 checks; four subjective prompts remain unscored automatically.
- Greedy result: Alpaca v3 passed 11/61 checks (prompt-average 0.167); UltraChat v2 passed 13/61 (0.190).
- Sampled result: Alpaca v3 passed 19/61 checks (prompt-average 0.292); UltraChat v2 passed 20/61 (0.301).
- Greedy category observations: Alpaca led definition checks (5/8 versus 4/8); UltraChat led formatting (3/8 versus 1/8) and instruction-following checks (3/8 versus 2/8). Both passed 0/5 reasoning, 0/5 factual-knowledge, and 0/7 programming checks.
- Sampled category observations: Alpaca led definition (7/8 versus 5/8), creative (2/3 versus 1/3), and programming (1/7 versus 0/7); UltraChat led conversation (4/6 versus 1/6), formatting (2/8 versus 1/8), and instruction following (4/8 versus 3/8). Both again passed 0/5 reasoning checks.
- Interpretation: the automatic metrics measure explicit prompt constraints and simple lexical/syntactic properties. They do not measure general intelligence, factual reliability, semantic quality, safety, or usefulness.
- Reports: `evaluation/checkpoint_comparison_expanded_40_greedy_auto.{txt,json}`, `evaluation/checkpoint_score_summary_expanded_40_greedy_auto.json`, `evaluation/checkpoint_comparison_expanded_40_sampled_auto.{txt,json}`, and `evaluation/checkpoint_score_summary_expanded_40_sampled_auto.json`.
- Decision: retain the earlier manually reviewed Alpaca-v3 default selection. Automatic results are supplementary evidence and do not authorize training or checkpoint promotion.

## Expanded 40-prompt five-seed sampled stability run

- Objective: replace dependence on one lucky or unlucky sampled pass with a reproducible view across seeds 42, 43, 44, 45, and 46.
- Configuration: unchanged sampled decoding (`temperature=0.45`, `top_k=20`, `top_p=0.8`, 60 new-token limit), two checkpoints, 40 prompts, and five samples per prompt.
- Scale: 200 generations per checkpoint and 400 total generations.
- Alpaca v3 result: automatic average 0.279, population standard deviation 0.371363, mean repetition 0.356561, and 98/305 automatic checks passed.
- UltraChat v2 result: automatic average 0.301, population standard deviation 0.376426, mean repetition 0.370466, and 106/305 automatic checks passed.
- Category comparison: Alpaca led definitions (0.825 versus 0.750) and planning (0.467 versus 0.367), while UltraChat led conversation (0.633 versus 0.367), instruction following (0.550 versus 0.500), formatting (0.333 versus 0.258), and creative checks (0.150 versus 0.100). Factual knowledge was tied at 0.075, programming at 0.025, uncertainty at 0.150, and reasoning at 0.000.
- Score-instability examples: `unknown_person` had score standard deviation 0.489898 for both checkpoints; `short_story` reached 0.400 for both. Alpaca `education_3_benefits` and UltraChat `two_programming_languages` also reached 0.400.
- Repetition-instability examples: Alpaca `comparison_reasoning` had a 0.871 repetition-ratio range; UltraChat `basic_arithmetic` had a 0.5714 range. A stable automatic score of zero on these prompts does not imply stable response quality.
- Interpretation: this run measures sampling stability and explicit heuristic checks. It does not validate factuality, semantic quality, safety, or general capability, and it does not override the manually reviewed Alpaca-v3 default selection.
- Reports: `evaluation/checkpoint_comparison_expanded_40_sampled_5seed.{txt,json}` and `evaluation/checkpoint_score_summary_expanded_40_sampled_5seed.json`.
- Decision: preserve both checkpoints and current training state; no new training is authorized by this evaluation.

## VASU-60M instruction-quality Batch 002

- Objective: perform a small masked instruction-quality refinement from the
  preserved Batch 001 best checkpoint and determine whether it produces a clear
  semantic improvement.
- Parent checkpoint:
  `checkpoints/vasu_60m/instruction_quality_batch_001_from_ultrachat_v2/best.pt`,
  global step 201304, SHA-256
  `4f377fc49fef2382c531b5bb28731fa9664bac217256b8a76dcf1427f4d39d8e`.
- Dataset: 500 approved source examples packed into 95 fixed records: 90
  training records and five validation records. The release contains 24,415
  tokens and 10,338 supervised response/EOS tokens.
- Training: one epoch, batch size 2, gradient accumulation 16, learning rate
  5e-7, weight decay 0.01, AMP, and gradient clipping 1.0.
- Completion: three optimizer updates, final global step 201307, train
  supervised-token loss 3.018085, and validation supervised-token loss
  3.189959 over 442 validation tokens.
- Selected artifact:
  `checkpoints/vasu_60m/instruction_quality_batch_002_from_batch_001/best.pt`,
  SHA-256
  `5025c1690035f0f0139bb2cff4044fd4a0ef6df0e480eee71d96a12e61272262`.
- Automatic comparison: format compliance, repetition, diversity, and Alpaca
  validation loss improved slightly. UltraChat validation loss was effectively
  unchanged.
- Human review: Batch 001 was preferred on eight prompts, Batch 002 on six,
  with 46 ties across the 60-prompt stratified review.
- Decision: preserve Batch 002 as a completed experiment but do not promote it.
  Batch 001 remains the preferred instruction-quality checkpoint because Batch
  002 did not establish a clear semantic improvement.
