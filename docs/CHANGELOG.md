# Changelog

## Unreleased

### Added

- Added the pre-commit implementation proposal for the VASU-140M detached v2
  authorization gate. It preserves the accepted builder identity while adding
  canonical external-envelope loading, exact runtime/gate binding, exclusive
  locking, conservative stale-lock recovery, direct v2 receipt provenance, and
  adversarial transaction coverage. GPT-5.5 independently accepted the exact
  pre-commit implementation; post-commit identity review remains pending. No
  envelope or protected production artifact was created.

- Added the specification-only VASU-140M one-build authorization protocol and
  independent review packet. The selected design separates the accepted
  implementation anchor from an exact clean runtime commit and uses a detached,
  self-hashed, single-use envelope to avoid circular Git identity. It defines
  fail-closed validation, concurrency, crash, review, and human-approval
  boundaries. GPT-5.5 independently accepted the design. Its audit records a
  non-blocking fixture-path typo without changing the hash-bound review packet.
  No envelope, publication, or training authority was created.

- Added the implementation proposal for the VASU-140M two-phase production
  release builder. It provides read-only full-source qualification, complete
  serialized mask audits, exact expiring one-build authorization validation,
  atomic publication, incomplete-publication quarantine, consumed receipts,
  and adversarial Windows/path/disk/mutation/reuse coverage. Frozen evidence
  contains hashes and counts only. GPT-5.5 independently accepted the exact
  implementation; post-commit qualification identity review remains required.
  No production release or authorization was created.

- Added versioned post-commit qualification evidence for accepted implementation
  commit `c014716`. Source, assignment, logical-record, token, and mask hashes
  remain unchanged; only commit-bound manifest and qualification identities
  changed. GPT-5.5 independently accepted the exact transition. The evidence
  remains non-authorizing.

- Corrected the post-commit qualification smoke to accept documentation-only
  descendants of `c014716` while still pinning the exact implementation-file
  SHA-256. This changes no builder or qualification identity.

- Added the specification-only VASU-140M production release-builder design and
  independent review packet. The recommended architecture separates read-only
  full-source qualification from exact, one-build-authorized publication and
  defines atomicity, recovery, path, lineage, reproducibility, and mask gates.
  GPT-5.5 independently accepted the design for a separately reviewed
  implementation proposal. No production builder, release artifact, or
  training authority was created by the design milestone.

- Added a fixture-only VASU-140M transactional release constructor. It binds
  the accepted plan and independent decision, reuses the frozen 513-token
  contract, rejects production paths and fixtures above 30 examples, publishes
  complete directories atomically, removes failed staging attempts, and
  validates every artifact hash and layout. GPT-5.5 independently accepted the
  exact layer as fixture-only construction evidence. This is test
  infrastructure only; it creates no production release or training authority.

- Added the specification-only VASU-140M instruction seed release plan. It
  binds 1,000 human-approved CC0 examples from instruction-quality Batches 001
  and 002, performs global review/deduplication checks, pins 2,618 evaluation
  prompts across ten authoritative files, quarantines four exact overlaps, and
  freezes 996 eligible examples into deterministic 898/48/50 planned splits.
  GPT-5.5 independently accepted the exact plan for future release
  construction review. Qualification is read-only:
  every planned production output is absent, no compatible base checkpoint is
  selected, and release construction and training remain unauthorized.

- Added the frozen, source-agnostic VASU-140M 513-token record and shifted-mask
  contract. It validates the unchanged tokenizer, complete-example packing,
  prompt/PAD/EOS/cross-example mask behavior, split isolation, fixed
  `uint16[513]`/`uint8[513]` layouts, record-local training views,
  deterministic fixture rebuild, lineage-report tamper rejection, and exact
  frozen-report identity pinning. GPT-5.5 independently accepted the remediated
  contract for reference by a future, separately reviewed source-specific
  release plan. No production dataset or training authorization was created.

- Added VASU-140M family-aware exact-resume qualification. The first immutable
  run exposed DataLoader iterator consumption of global PyTorch RNG, which
  shifted dropout after resume. Dedicated loader generators now isolate model
  RNG. The corrected v2 synthetic CPU run matched model, AdamW, scheduler,
  scaler, sampler, partial gradients, sample order, progress, and RNG state;
  real training remains unauthorized.

- Added the formal Candidate D matched-control scientific decision record.
  Candidate D control completed its 10M-token non-arithmetic continuation from
  Candidate A final, reached 112/1000 arithmetic development exact accuracy
  versus Candidate A's 117/1000, and preserved/slightly improved broad-language
  and factual measures. It is accepted as the matched-control comparison
  checkpoint only; Candidate D treatment remains unauthorized, Candidate A
  final remains the preferred continued-pretraining base, and masked Alpaca v3
  remains the preferred instruction-tuned assistant.

- Added the formal Candidate A scientific decision record. Candidate A's final
  arithmetic treatment checkpoint is the preferred continued-pretraining base:
  it achieved 117/1000 development and 124/1000 held-out arithmetic exact
  accuracy while retaining broad-language and factual gains. Its arithmetic
  capability is narrow and repetition remains a limitation; masked Alpaca v3
  remains the preferred instruction-tuned assistant, Candidate C remains the
  successful control, and Candidate B remains unauthorized.

- Added the formal Candidate A parent-checkpoint scientific decision record.
  Candidate A remains the parallel FineWeb-step-200,000 treatment branch for
  the Candidate C control comparison, preserving the matched approximately
  20M-token design and isolating the 5% verified-arithmetic-v2 source.
  Candidate C `final.pt` remains the preferred practical continued-pretraining
  base; Candidate A and Candidate B remain unauthorized. The record is
  `docs/CAPABILITY_CPT_A_FACTUAL_20M_V2_PARENT_CHECKPOINT_DECISION.md`.

- Added the formal Candidate C scientific decision record. Candidate C's
  completed 20M-token control improved FineWeb and Wikimedia validation loss
  and normalized cloze performance, while arithmetic remained neutral and
  repetition behavior remained mixed. Candidate C is a continued-pretraining
  base candidate, not an automatic replacement for the Alpaca v3 assistant.
  Candidate A remains eligible for separate authorization review and Candidate
  B remains unauthorized.

- Added the opt-in production Candidate C capability-CPT runtime with
  successful-update interval validation, separate FineWeb/Wikimedia/arithmetic
  best checkpoints, durable validation-event resume, explicit recovery
  selection, atomic verified checkpointing, corruption filtering, bounded
  retention, disk guards, and thermal abort monitoring.
- Added deterministic, hash-bound, resumable full verified-arithmetic-v2
  development/evaluation tooling with strict exact-answer outcome categories
  and per-example atomic progress.
- Added a scientific authorization packet, authorization-record template, and
  human-readable launch/resume/evaluation/rollback runbook. Candidate C's
  authorized run is complete; Candidate A and Candidate B remain unauthorized.

- Added deterministic verified arithmetic v2 with 34,000 exact-answer logical
  examples, 3,261 unique packed training records, held-out templates/ranges,
  split-contamination checks, transactional serialization, and standalone
  validation.
- Added versioned per-source replay-safety gates, scoped justified overrides,
  A/B/C v2 schedules, CUDA AMP scheduled-mixture smoke validation, and CUDA
  exact-resume coverage across accumulation, source, and replay boundaries.

- Added the generalized deterministic N-source scheduled-mixture layer,
  transactional A/B/C schedule releases, packed-mask and standard-source
  batching, schedule-aware resume identity, launch-config validation, and
  bounded non-optimizing real-model smoke tooling. The legacy 85/15 mixture is
  unchanged and all new training configurations remain unauthorized.

- Added the canonical hash-bound `verified_arithmetic_v1` serializer and
  standalone validator. The release preserves deterministic logical
  train/development/evaluation splits, uses the unchanged VASU tokenizer,
  packs one unique training pass into fixed 257-token records, and validates
  exact answers, terminal EOS, PAD-tail masks, cross-example boundaries,
  artifact hashes, split isolation, and capability-v1 exclusion.
- Added transactional staging and overwrite protection, dry-run and compact
  JSON summaries, strict utilization gating, deterministic second-build
  checks, and direct `PackedInstructionDataset` compatibility tests. Generated
  binaries remain ignored and training remains unauthorized.

- Added the versioned, reproducible VASU internal capability-evaluation
  framework with canonical checkpoint loading, greedy/seeded generation,
  atomic resumable result files, objective structural scorers, explicitly
  labelled heuristic metrics, and human-review promotion blocking.
- Added deterministic, versioned promotion-gate enforcement for continued
  pretraining, instruction, and conversation comparisons. Promotion reports
  now preserve category-level calculations, explicit blockers, structured
  human-review decisions, and gate hashes for resume compatibility.
- Added unauthorized VASU-60M capability-CPT A/B/C mixture plans and a
  deterministic verified-arithmetic smoke generator. The plans cap the
  approved Wikimedia release at its no-replacement capacity; no training data
  mixture or model run was created.

- Added compact deterministic mid-epoch resume support to the general Trainer.
  Additive checkpoint metadata now preserves sampler, partial-gradient, AMP
  scaler, and RNG state; legacy checkpoints remain loadable with an explicit
  non-exact-resume warning.
- Hardened the general Trainer checkpoint path with atomic saves, explicit
  non-finite-gradient skip handling, phase-consistent callback timing, and
  defined scheduler-horizon behavior for extended epoch targets.
- Added explicit standard, foreach, and CUDA-fused AdamW backend selection plus
  a bounded synthetic CUDA benchmark. Standard AdamW remains the default.
- Added bounded real-data pipeline and disposable checkpoint-I/O profiling
  tools. The audit retained Windows-safe worker-zero defaults because VASU-60M
  was compute-bound, and changed general Trainer validation to
  `torch.inference_mode()` after loss-parity validation.

- Completed VASU-60M instruction-quality Batch 002 from the Batch 001 best
  checkpoint. The isolated three-update run finished at global step 201307 with
  train/validation supervised-token losses of 3.018085/3.189959.
- Added deterministic Batch 001-versus-Batch 002 evaluation on the frozen
  216-prompt benchmark and a completed 60-prompt semantic review. Human
  preferences were eight for Batch 001, six for Batch 002, and 46 ties.
- Preserved Batch 002 checkpoint SHA-256
  `5025c1690035f0f0139bb2cff4044fd4a0ef6df0e480eee71d96a12e61272262`
  as a rejected promotion candidate. Batch 001 remains preferred.


- Repaired instruction-quality batch 001 after its first 100-example human gate
  failed (77 approved, 15 fact checks, eight rewrites). Replaced all 125 broad
  factual references with claim-specific sources and corrected list filler,
  ambiguous wording, beginner grammar, and transformation fidelity.
- Added a hash-bound repair audit recording 159 changed and 341 unchanged
  records plus 38 stale gate-v1 decisions. No historical decision was reused or
  copied into the empty production review file.
- Added deterministic gate-v2 sampling and a blank 100-example review packet
  with the original category distribution, old gate hashes where available,
  new record hashes, changed flags, and automatic findings. Training,
  tokenization, release building, and automatic approval remain disabled.

- Authored deterministic instruction-quality batch 001 with exactly 500
  purpose-written, unreviewed production candidates and the configured seven-
  capability distribution. Added stable batch IDs, source/config manifests,
  automatic validation and token-length reports, and a complete grouped human-
  review packet.
- Extended the existing v1 validator additively for namespaced batch IDs,
  verified HTTP(S) factual references, exact numbered-item counts, exact word
  counts, and labelled-field structures. Existing demo IDs and constraints
  remain compatible.
- Verified zero invalid records, exact duplicates, near-duplicate candidates,
  demo collisions, and truncations. No production token, mask, release, model,
  checkpoint, or training artifact was created; authorization remains false.

- Added the versioned `vasu_instruction_quality_v1` data-engineering pilot:
  strict source schema, transparent validation and quality scoring, layered
  exact/near deduplication, hash-bound human decisions, deterministic
  capability-stratified splitting, and atomic masked-release construction.
- Added a 21-record demonstration fixture spanning all seven planned
  capabilities, validation/source/review manifests, a readable review packet,
  CLI validation/review/release commands, and focused regression tests.
- Reused the established masked Alpaca-v3 prompt, response/EOS supervision,
  and padding-mask behavior without changing tokenizer, checkpoints, model, or
  training code. The fixture remains unreviewed and `training_authorized=false`;
  no release or training run has been authorized.

- Added the frozen 216-prompt UltraChat promotion benchmark, checkpoint audit,
  dual held-out masked-loss evaluation, EOS-aware degeneration metrics, strict
  and relaxed format checks, and a blank 60-prompt stratified review packet.
- Evaluated the historically completed UltraChat masked-v2 branch without
  optimizer updates. UltraChat loss improved, but format compliance,
  repetition, and coherence heuristics regressed; Alpaca v3 remains the main
  instruction checkpoint and no additional training stage is authorized.

- Closed the completed FineWeb/Wikimedia factual-CPT branch as preserved but
  rejected: losses improved, while frozen cloze and multiple-choice evidence
  did not establish a meaningful factual gain. Further CPT and instruction
  tuning from that candidate are unauthorized.
- Added a reproducible Alpaca-v3 parent-baseline artifact for the masked
  UltraChat comparison, derived from the same preserved 40-prompt greedy and
  sampled reports and recording masked validation loss plus response-level
  comparison heuristics.
- Verified that the isolated UltraChat runner strictly names masked Alpaca v3
  as its parent and contains no factual-CPT parent reference. The already
  completed UltraChat outputs remain preserved as an experimental branch;
  future reruns still require explicit `--train` authorization.

- Added the frozen 300-example factual-CPT v2 benchmark: 100 cloze examples,
  100 direct conditional-likelihood multiple-choice examples, 50 general
  continuations, and 50 qualitative prompts, with seeded bootstrap intervals.
- Added parent/best/latest factual-CPT evaluation with benchmark, configuration,
  tokenizer, and checkpoint hashes plus greedy and controlled-sampling
  degeneration metrics that respect response boundaries.
- Completed the approved 1,221-step FineWeb/Wikimedia factual-CPT run. FineWeb
  validation improved from 3.356130 to 3.317529 and held-out Wikimedia from
  3.393557 to 3.286295 at the selected checkpoint.
- Preserved the factual-CPT checkpoint as an evaluation candidate rather than
  promoting it: exact cloze and MC accuracy did not improve, and the two-point
  normalized-cloze increase was not distinguishable from benchmark noise.

- Prepared an authorization-gated VASU-60M factual continued-pretraining pilot
  from FineWeb step 200,000: 85% unseen FineWeb-Edu, 15% approved Wikimedia,
  seed 42, and a 10-million-token cap. No optimizer update was performed.
- Added deterministic parent-level Wikimedia splitting and isolated `uint16`
  train/validation tokenization with exact source/tokenizer hashes. The split
  contains 382 training and 20 validation parents with zero overlap.
- Added reusable fixed-record pretraining-mixture preparation. The 39,062-record
  artifact contains 9,999,872 supervised positions and prevents cross-source
  target transitions. A clean rebuild reproduced binary SHA-256
  `60a06cc54f0c77b977db733829584786edbe518eda38eab83964887d9a12bc4b`
  and schedule SHA-256
  `f426c1e3a301bad4b451c2b6d8b1bea2fe567dd262d493fadf5c25e368fef6b0`.
- Added an explicit factual-CPT evaluation plan and six category-level prompts
  covering science, history, geography, technology, biography, and definitions.
  The prompts are not copied from the Wikimedia release.
- Added a versioned, SHA-bound Wikimedia quarantine release transformation.
  It excludes only the 12 exact chunks rejected by the frozen human gate,
  preserves retained records byte-for-byte and in order, publishes atomically,
  and validates source immutability, IDs, provenance, parent counts, chunk
  counts, and token counts.
- Published the approved quarantine release at SHA-256
  `6aa10d73669ca90ad20f867f14a6368d2191b38094f02aa1679ebaf122962de7`:
  402 parents, 3,597 chunks, and 1,993,564 tokens after quarantining 12 chunks
  and 6,133 tokens. Existing-policy audit results are zero automatic rejects
  and zero unexplained findings. No review sample or training run was started.
- Added a named, dataset-wide Wikimedia defect scanner covering every defect
  family recorded in completed manual-review archives, with bounded evidence,
  automatic-versus-review dispositions, and zero-unexplained-match enforcement.
- Added the immutable Wikimedia release policy: zero automatic precheck
  failures, zero human critical rejects, noted harmless minor issues, one final
  remediation cycle, and hash-qualified approved artifacts.
- Froze the final global-audit candidate at SHA-256
  `ffcbc25f4863f519744212f809ee600bdc7f4a0d5c2d02a0833e1bc4cec6014d`
  with 3,609 chunks, 1,999,697 tokens, a zero-unexplained global audit, and an
  84-item pending deterministic review. No Wikimedia training was started.

- Added deterministic, SHA-bound Wikimedia manual-review tooling with local
  seeded sampling, overlapping-reason preservation, bounded start/end
  previews, automatic quality prechecks, atomic status updates, and explicit
  pending/pass/minor-issue/reject validation.
- Generated the initial 79-chunk pending review sample across 62 parent
  articles. It includes all requested bounded sampling groups, 21 forced
  reference-type sections, 2 suspicious-metadata selections, and 14 removed
  duplicate selections; no record was automatically approved and no training
  was started.
- Added a deterministic, resumable, atomic FineWeb-extension source-ID recovery
  smoke using the official Dataset Viewer exact-ID filter with per-request
  pinned-revision enforcement and no full-scan fallback.
- Added 25 network-free recovery tests covering historical evidence, spread
  sampling, exact hashing, classifications, revision/config protection,
  resume, atomic writes, report safety, and scalability blocking.
- Completed the bounded 100-ID smoke: 100 IDs found, 100 exact
  `SHA-256(text.strip())` matches, zero failures, 498,312 accepted text bytes,
  and no retained source text. Full reacquisition and extension indexing remain
  unexecuted.
- Added a 1,000-ID recovery benchmark with serial, bounded concurrency 2/4/8,
  documented OR batches 5/10/25, isolated cold/warm caches, rate limiting,
  separate connect/read timeouts, Retry-After, exponential retry, seeded
  jitter, HTTP accounting, atomic resume, and production estimates.
- Selected OR batch 25 with concurrency 1 after all seven strategies produced
  1,000/1,000 exact historical hashes. It used 40 requests in 37.33 seconds;
  full recovery, extension indexing, and training remain unstarted.

- Added a versioned SQLite FineWeb document-index abstraction with exact
  normalized SHA-256 lookup, deterministic word-5-gram MinHash/LSH candidates,
  provenance fields, atomic promotion, bounded builds, and resumable progress.
- Added Wikimedia cross-source exact/near rejection and ambiguous-overlap
  review flags under normalization version
  `vasu_cross_source_nfc_casefold_ws_v1`.
- Added FineWeb artifact inventory/build/check CLIs and network-free tests. No
  training run was started.
- Built and validated the original FineWeb document index: 999,992 unique
  documents, eight duplicate hashes, 7,999,936 LSH buckets, original-only
  coverage, and explicit missing-extension status.
- Added an extension recovery audit and source-ID reacquisition plan without
  downloading or rebuilding extension data.

### Changed

- The first deterministic Wikimedia manual review failed because it exposed
  systematic reference appendix leakage, 18 below-minimum samples, malformed
  overlap boundaries, list-heavy content, and missing source/template values.
  Its SHA-bound reports remain archived for evidence.
- Wikimedia v4 preparation now excludes canonical structured and embedded
  reference headings; merges or rejects subminimum chunks; records boundary
  types and overlap characters; omits unsafe mid-sentence overlaps; and runs
  configurable list-density plus conservative malformed-source checks before
  deduplication and accounting.
- Regenerated and validated the default pilot at 304 parents, 3,480 chunks,
  and 1,999,700 tokens, with an observed 128–1,023 token range and zero retained
  reference headings, replacement/mojibake markers, or token fallbacks. The
  new 61-chunk review is fully pending and has zero automatic precheck failures.
- Replaced Wikimedia preparation's ambiguous accepted-document limit with
  independent `max_accepted_parent_documents` and `max_accepted_chunks`
  limits. The legacy key remains a deprecated chunk-limit alias and cannot be
  combined with the explicit chunk field.
- Versioned Wikimedia progress and preparation manifests now report raw
  examples, unique accepted parents, accepted chunks, rejected items, and VASU
  tokens independently. Unsafe legacy progress requires an explicit restart.
- Regenerated the bounded default Wikimedia pilot under the explicit limits:
  262 parent documents, 3,751 chunks, and 1,999,974 tokens, ending at the token
  ceiling with a validated v3 manifest and no replacement characters. One
  high-confidence FineWeb near overlap was rejected.
- Training-oriented Wikimedia preparation excludes reference-like sections by
  default; broad-review mode continues to flag them.
- Default factual preparation requires a completed compatible FineWeb index.
  Smoke and broad-review modes may proceed only with an explicit blocked status.

- Added a bounded, resumable Wikimedia factual-pilot preparation package and
  CLI pinned to the approved `20231101.en` revision and one explicit Parquet
  shard.
- Added deterministic filtering with reason accounting, exact and bounded
  near-duplicate detection, evaluation-prompt contamination checks, exact VASU
  token measurement, provenance-rich JSONL, atomic progress, manifests, and
  output validation.
- Added 57 network-free Wikimedia preparation tests covering limits, source
  approval, pinned acquisition, filtering, deduplication, contamination,
  resume/restart, hashes, validation, path independence, and Git-ignore rules.
- Added conservative reversible mojibake repair, Unicode quality reason codes,
  boundary-preserving reference/citation/template cleanup, and exact-token
  article chunking.
- Added the `wikimedia_pilot_v2` chunk schema with parent provenance, chunk
  IDs/index/count, section metadata, repair status, warnings, and chunk hashes.
- Added deterministic broad-review row selection, row-group-bounded reads,
  review-suffixed outputs, a five-chunk-per-parent diversity cap, reference
  section keep/flag/exclude behavior, and diversity/warning summaries.
- Added bounded retries around atomic replace to tolerate transient Windows
  file locks without weakening atomic progress semantics.

- Pinned primary-source registry evidence and an approved preparation plan for the factual pilot's official `wikimedia/wikipedia` `20231101.en` snapshot at commit `e6057dc557255a03c9c3c47ceab0eb44353b1bc5`, including exact published size/example counts, licensing obligations, Windows-safe paths, and quality/deduplication controls. No data was downloaded or training started.
- Source-registry command-line validation for mixture readiness via `python -m vasu.data.sources.registry`.

- Reproducible sampled checkpoint comparison with `--seed`, consecutive `--num-samples`, and explicit ordered `--seeds` support.
- Multi-seed per-prompt score/repetition aggregates, per-check pass rates, checkpoint/category summaries, and grouped seed-specific text output.
- Five-seed expanded 40-prompt stability reports for Alpaca v3 and UltraChat v2.
- Deterministic, task-specific automatic response checks for exact/accepted answers, keywords, formatting, repetition, uncertainty, clarification, and parseable Python code.
- Optional per-prompt `checks` metadata and automatic per-prompt/checkpoint summary fields in text and JSON evaluation reports.
- Expanded 40-prompt automatic-check reports for greedy and sampled Alpaca-v3/UltraChat-v2 comparisons.
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

- Multi-seed evaluation resets Python, PyTorch CPU, and CUDA RNG state before each generation; it does not enable deterministic kernels or alter sampling semantics.
- Greedy, legacy unseeded sampling, and exactly-one-seed evaluation retain the existing flat generation-result schema.
- Checkpoint comparison summaries now include automatic pass counts and averages alongside unchanged manual-score fields; prompts without checks remain backward compatible.
- Response statistics and automatic repetition checks now share one canonical repetition calculation.
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

- Bounded benchmark task submission so a failed request cannot drain an
  uncommitted pre-submitted queue, and added bounded atomic-promotion retries
  for transient Windows scanner locks.

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

- Wikimedia pilot limits are one shard, 10,000 raw rows, 2,000 accepted parent
  documents, 4,000 accepted chunks, 2,000,000 tokens, and 1 GB downloaded. The
  corrected smoke retained 20 chunks and 17,237 tokens with a 1,022-token
  maximum. The earlier 2,000-chunk/1,036,527-token default artifact remains
  valid but limit-bound; the refactored default artifact reached 1,999,974
  tokens. Model training was not started.
- Smoke mojibake was traced to incompatible Windows display decoding rather
  than corrupted Parquet/JSONL bytes. The real empty-string inline-cleanup
  boundary risk was fixed and regression-tested.
- The review-only Wikimedia sample retained 50 chunks from 14 articles and
  26,435 tokens; it is not an authorized training artifact.
- FineWeb cross-source deduplication is explicitly blocked without a versioned
  document-level normalized-hash/signature index; existing token binaries are
  not sufficient.

- Added an evidence-based next-generation planning decision: first validate a factual/math/code/reasoning data mixture through small ablations, then conditionally continue the existing VASU-60M base for approximately 1.2B new tokens. No implementation or training is authorized by the plan.
- Compared continued VASU-60M training, a same-size v2, an approximately 100M–120M scale-up, and a smaller pipeline-validation model, including hardware, runtime, compatibility, data, and failure-risk trade-offs.
- Five-seed results describe sampling stability rather than a single sampled outcome; the automatic metrics remain heuristic and are not general-intelligence scores.
- Automatic evaluation scores are heuristic task-compliance indicators, not measures of general intelligence or model reliability; manual review remains required.
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
