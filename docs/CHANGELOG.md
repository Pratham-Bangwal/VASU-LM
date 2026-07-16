# Changelog

## Unreleased

### Added

- Opt-in VASU-60M configuration with 58,337,792 parameters.
- CPU, CUDA, realistic-memory, and tiny real-data smoke-test tooling.
- Resumable VASU-60M FineWeb block pretraining.
- Automatic thermal-stop protection and cooldown-aware long-run orchestration.
- Low-disk protection and bounded checkpoint retention.
- Raw base-model milestone evaluation with sampled and optional greedy decoding.
- Dedicated resumable VASU-60M standard-Alpaca training with thermal, disk, atomic-save, corruption-filtering, and bounded-retention safeguards.
- Architecture-aware checkpoint entries in the comparison evaluator.
- Shared, verified standard-Alpaca prompt formatting for dataset preparation and inference.
- Deterministic greedy checkpoint-comparison mode and a VASU-60M FineWeb step-150,000 baseline entry.
- Boundary-preserving masked-Alpaca-v2 preparation, metadata, alignment validation, dry-run, and hardened one-epoch training tooling.
- Named evaluation-report outputs that preserve earlier checkpoint comparisons.
- A resumable FineWeb-Edu extension preparer with pinned-source streaming, bounded `uint16` writes, atomic promotion, persistent SQLite exact deduplication, and complete metadata.
- FineWeb extension validation and logical multi-shard manifest tooling.
- Additive manifest-aware token dataset with lazy read-only memmaps and cross-shard sequence reads.
- Production manifest mapping validator with explicit three-location CUDA dry-run mode.
- Exact hard-target support for bounded FineWeb continuation, with regression coverage for a step-152,000 cap and preservation of the step-150,200 integration checkpoint.
- CPU-only checkpoint inspection and a bounded step-200,000 orchestration wrapper with per-block logs, JSONL summaries, SHA-256 validation, exact progress checks, and atomic milestone preservation.
- Isolated VASU-60M masked-Alpaca-v3 runner prepared from the authoritative FineWeb step-200,000 base, with strict preflight validation and a no-write forward-only dry-run mode.
- Boundary-aware UltraChat masked-v2 preparation, read-only validation, supervised EOS metadata, isolated one-epoch runner, and explicit no-write dry-run mode.
- Expanded 40-prompt greedy and sampled benchmark for final VASU-60M checkpoint selection.
- Typed inference-only KV cache with validated per-layer tensor state, explicit causal prefill, and one-token decode modes.
- Cached-versus-uncached preferred-checkpoint benchmark with token parity, stage timing, throughput, and CUDA peak-memory reporting.

### Changed

- VASU-60M training now operates in short 100-optimizer-step blocks with checkpoints every 10 steps.
- Preserved milestones are stored separately from retained operational checkpoints.
- Project documentation now distinguishes the completed VASU-31M cycle from active VASU-60M base pretraining.
- VASU-60M completed one standard full-loss Alpaca epoch from the preserved step-150,000 base checkpoint; UltraChat remains pending.
- Checkpoint entries may explicitly declare plain or Alpaca prompt formatting while legacy entries retain their previous behavior.
- Masked Alpaca v2 uses fixed records, assistant/EOS-only supervision, masked padding, and explicit truncation accounting without changing existing datasets.
- The FineWeb block runner now reads sequential logical windows from the manifest with `shuffle=False`; batch size, accumulation, optimizer, AMP, checkpoint schema, and validation sample count are unchanged.
- Authorized manifest-aware FineWeb continuation now uses 200-step single-process executions and an exact shortened final block at the hard step-200,000 ceiling.
- Masked Alpaca v3 reuses the v2 dataset, target-shift mask semantics, supervised EOS behavior, split, optimizer, scheduler, seed, AMP, and one-epoch limit; only base provenance and output isolation differ.
- Alpaca masked v3 from FineWeb step 200,000 is now the default experimental VASU-60M assistant checkpoint.
- `chat.py` now uses VASU-60M and Alpaca v3 instead of the legacy VASU-31M UltraChat checkpoint, with explicit Alpaca prompt formatting and the validated sampled-decoding settings.
- UltraChat masked v2 is retained as an experimental branch and is not promoted.
- Generation accepts opt-in `use_kv_cache`; the existing uncached full-history path remains the default reference implementation.
- `chat.py` exposes `USE_KV_CACHE = False` pending a separate activation decision.

### Fixed

- Checkpoints are saved through temporary files and atomically promoted after serialization.
- Corrupt and incomplete checkpoints are ignored during resume selection.
- Checkpoint retention prevents unbounded periodic saves from filling the disk.
- The long-run runner validates internal checkpoint `global_step` values rather than trusting filenames alone.
- The comparison evaluator now honors one or more `--checkpoints` IDs and rejects unknown IDs clearly.
- Corrected evaluation reports preserve earlier comparison history instead of overwriting it.
- FineWeb extension resume validates tokenizer/source identity and reconciles temporary output against committed SQLite progress rather than appending blindly.
- FineWeb resume positioning now remains exact across partial blocks because the specialized block loader uses deterministic sequential ordering.
- The earlier pre-boundary controlled runner cannot advance beyond step 152,000 and exits without training when that historical target is already reached.
- Step-200,000 orchestration conflict detection excludes its own ancestor shell and matches actual training scripts/runners rather than unrelated commands containing FineWeb path text.
- Interactive chat loading now strictly validates the checkpoint model-state keys and tensor shapes against the VASU-60M configuration.
- Cached multi-token prompt prefill now uses causal attention instead of allowing prompt tokens to attend to future prompt positions.
- Cached decode now enforces a populated synchronized cache, one-token queries, correct RoPE offsets, and the model context limit.

### Documented

- VASU-60M step-54,060 milestone: train loss 3.616769 and validation loss 3.613814.
- VASU-60M base pretraining completed for this cycle at the preserved step-150,000 checkpoint.
- The first VASU-60M instruction stage used standard Alpaca for exactly one epoch; UltraChat has not started.
- VASU-60M Alpaca result: final resumed-segment train loss 2.743232, validation loss 2.674379, peak CUDA memory 1,376.37 MiB, and verified resume after a safe thermal stop.
- Prompt audit result: the standard Alpaca binary and prior evaluator both used `User:/Assistant:` formatting; the suspected plain-prompt mismatch was not the cause of poor generations. Corrected deterministic outputs remain unscored, and VASU-60M UltraChat has not started.
- VASU-60M masked Alpaca v2 completed one epoch at global step 150,711 with train supervised-token loss 2.852573 and validation supervised-token loss 2.732960. Deterministic outputs remain unscored and unreliable; UltraChat has not started.
- The original FineWeb training region ends at global-step capacity 152,086 under the established offset mapping.
- A validated `CC-MAIN-2025-26` FineWeb-Edu extension contains 500,000,478 tokens; step 200,000 capacity is available without altering the original validation slice. No continuation training has started.
- Manifest-aware production checks passed at the original shard, cross-shard boundary, and extension shard with finite CUDA loss/gradients and unchanged protected-file hashes. Full continuation training has not started.
- One live manifest integration block completed from step 150,000 to 150,200 with train loss 3.350210, 72°C maximum temperature, 1,376.37 MiB peak allocated CUDA memory, and a validated final checkpoint. No long-running continuation or extension-shard training was started.

- A bounded nine-block continuation completed from step 150,200 to the exact step-152,000 hard target. It consumed 14,745,600 tokens, remained 707,252 tokens before the extension boundary, ended with train loss 3.359692 and validation loss 3.364378, reached 73 C maximum temperature, and produced a fully validated `block_final_step_152000.pt`. No extension-shard or long-running training was started.
- Manifest-aware continuation reached the exact step-200,000 target at logical offset 1,638,400,000. The final block reported train loss 3.101754 and validation loss 3.356130; maximum recorded temperature was 82°C, peak allocated CUDA memory was 1,376.37 MiB, and no thermal stop occurred.
- `block_final_step_200000.pt` and preserved `milestones/fineweb_step_200000.pt` passed CPU model/optimizer/scheduler, architecture, finite-state, internal-step, and SHA-256 validation. Instruction tuning remained paused and UltraChat was not started.
- FineWeb step 200,000 remains the authoritative evaluated v3 base; steps 200,400 and 200,590 are preserved as validated recovery/history checkpoints only.
- Masked Alpaca v3 preparation passed a finite forward-only loss check (2.146862), 35 focused tests, and the 140-test full suite. No v3 checkpoint was written, instruction training was not started, and UltraChat remains paused.
- Masked Alpaca v3 subsequently completed at step 200,711 with train assistant-token loss 2.627628 and validation loss 2.522585. UltraChat masked v2 preparation produced 5,099,908 tokens and passed a finite no-write dry run; UltraChat training has not started.
- UltraChat masked v2 subsequently completed 590 optimizer steps from global step 200,711 to 201,301. The 40-prompt comparison selected Alpaca v3 as the default and preserved UltraChat v2 as experimental; neither model is documented as generally reliable.
- KV-cache CPU/CUDA logits, greedy token IDs, EOS stopping, and context-limit parity passed. Dynamic caching lowered measured peak CUDA allocation but was approximately 0.9% slower on the 100-token benchmark, so it remains disabled by default.

## v0.5

- Added inference sampling and a chat interface.

## v0.4

- Added TensorBoard logging, validation, callbacks, and memory-mapped datasets.

## v0.3

- Added AMP, gradient clipping, and checkpointing.

## v0.2

- Added RMSNorm, RoPE, and weight tying.

## v0.1

- Initial GPT-style implementation.
