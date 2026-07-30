# VASU Project Status

## Capability-CPT v2 status

Candidate C (`capability_cpt_c_control_20m_v2`) completed its authorized
20,004,864-token, 2,442-update control run from the FineWeb step-200,000
parent. It used 91% FineWeb replay and 9% approved Wikimedia factual data,
with no arithmetic training.

Candidate C improved FineWeb and Wikimedia validation loss and produced a
statistically distinguishable normalized cloze improvement. Arithmetic exact
accuracy remained 0/1000, capability-v1 showed no measurable objective
regression, and repetition behavior was mixed. The formal decision is recorded
in `docs/CAPABILITY_CPT_C_CONTROL_20M_V2_DECISION.md`.

Candidate A (`capability_cpt_a_factual_20m_v2`) completed its matched
20,004,864-token, 2,442-update arithmetic treatment from the FineWeb
step-200,000 parent. It used 86% FineWeb replay, 9% approved Wikimedia factual
data, and 5% verified arithmetic v2. Candidate A achieved 117/1000 arithmetic
development and 124/1000 held-out exact accuracy, while Candidate C and the
parent both scored 0/1000 on development.

Candidate A `final.pt` is the preferred continued-pretraining base checkpoint.
It retains the broad-language and factual gains relative to the parent with
only operationally very small likelihood costs relative to Candidate C. Its
arithmetic improvement is narrow, concentrated in comparisons and numeric
properties; repetition remains a limitation. Candidate A is not the preferred
interactive assistant: masked Alpaca v3 remains assistant-specific. Candidate
C remains the successful control and historical comparison checkpoint. Candidate
B remains unauthorized and conditional. The formal decision is recorded in
`docs/CAPABILITY_CPT_A_FACTUAL_20M_V2_DECISION.md`.

Candidate D control (`capability_cpt_d_control_10m_from_a_v1`) completed its
matched 10,002,432-token non-arithmetic continuation from Candidate A `final.pt`.
Its selected `final.pt` has SHA-256
`3f513727ed0ea63a9b4aaf963c736f30b409e6901caddb50bf85c1db6922c384`.
It processed 39,072 records in 19,536 microbatches and 1,221 optimizer updates
with 91% FineWeb, 9% Wikimedia, and 0% arithmetic. It reached 112/1000
arithmetic development exact accuracy versus Candidate A's 117/1000, consistent
with no meaningful arithmetic improvement from non-arithmetic continuation.
FineWeb/Wikimedia losses improved to 3.299042/3.264998 and normalized cloze
rose to 0.120. The control is scientifically accepted as the matched-control
comparison checkpoint, not the preferred production base. It supports
consideration of, but does not authorize, Candidate D treatment; Candidate D
treatment remains `training_authorized: false` and Candidate B remains
unauthorized. Candidate A final remains the preferred continued-pretraining
base until treatment evaluation is complete, and masked Alpaca v3 remains the
preferred instruction-tuned assistant. See
`docs/CAPABILITY_CPT_D_CONTROL_10M_FROM_A_V1_DECISION.md`.

Candidate D treatment (`capability_cpt_d_arithmetic_10m_from_a_v1`) completed
its matched 10,002,432-token continuation from Candidate A final, including
explicit safe resumes after disk and thermal safeguards. Its final checkpoint
is `checkpoints/vasu_60m/capability_cpt_d_arithmetic_10m_from_a_v1/final.pt`
with SHA-256 `c17f9a481d6ae235f4264cbe9ca2775529291da303ff1e38a2d1d3ac8258aaa2`.
Promotion is rejected: arithmetic development was 112/1000 and held-out
arithmetic was 121/1000, below Candidate A's 124/1000 and far below the
predeclared +5-point requirement. Core arithmetic operations remained at 0%
held-out exact accuracy. FineWeb/Wikimedia retention stayed within the
matched-control limits, but retention cannot compensate for the failed primary
capability objective. Candidate A final remains the preferred continued-
pretraining base; Candidate D treatment artifacts are retained only as a
rejected experimental record. `training_authorized` is false and Candidate B
remains unauthorized. See
`docs/CAPABILITY_CPT_D_ARITHMETIC_10M_FROM_A_V1_DECISION.md`.
The follow-up read-only operation-family analysis is recorded in
`docs/CANDIDATE_D_ARITHMETIC_TREATMENT_ERROR_ANALYSIS.md`.
The subsequent read-only training-signal audit and non-authorizing next
hypothesis are recorded in
`docs/CANDIDATE_D_TRAINING_SIGNAL_AUDIT_AND_NEXT_HYPOTHESIS.md`.

Candidate E is the proposed matched step-supervision follow-up. Its
non-authorizing preparation package includes paired target/mask compilation,
immutable release tooling, split and budget validation, frozen evaluation
gates, and an independent-review packet. No Candidate E production dataset,
schedule, training configuration, authorization, checkpoint, or training run
exists. See `docs/CANDIDATE_E_INDEPENDENT_REVIEW_PACKET.md`.

## VASU-140M 513-token data contract

The generic VASU-140M fixed-record and shifted-mask contract is frozen and
fixture-qualified. It binds the exact family/configuration, unchanged
32,000-token tokenizer, `uint16[513]` records, `uint8[513]` stored masks, and
trainer derivation `tokens[:-1]`, `tokens[1:]`, `stored_mask[1:]`. Validators
fail closed on boundary merges, truncation, PAD content, wrong EOS
supervision, cross-example targets, split/semantic leakage, invalid dtypes,
and nondeterministic rebuilds.

The frozen fixture report is
`evaluation/fixtures/vasu_140m_513_record_spec_v1.json`. Primary engineering
review is complete. A first independent pass found that the generic validator
did not pin the exact frozen report identity; engineering added an exact
canonical-identity gate and a modified-and-rehashed regression test. The
remediated contract was independently accepted by GPT-5.5 on 2026-07-30 for
use by a future, separately reviewed source-specific release plan. No
production VASU-140M dataset, source selection, schedule, training config,
authorization, checkpoint, or training run exists. See
`docs/VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md`. The requirement-level
completion audit is
`docs/VASU_140M_513_TOKEN_MILESTONE_AUDIT_20260730.md`.

## VASU-140M instruction seed release plan

The first source-specific plan is frozen and engineering-qualified without
constructing a release. It references the two existing human-approved,
purpose-written CC0 instruction-quality batches: 1,000 examples across seven
capabilities. Global source review and deduplication checks pass.

An initial three-file contamination inventory was rejected during engineering
review as incomplete. The remediated comparison covers 2,618 prompts across
ten hash-pinned evaluation files and found four exact question overlaps. The
plan quarantines those IDs, leaving 996 eligible
examples with zero remaining full-prompt or eight-word-fragment matches. It
preserves all 48 source-native validation examples as development, selects 50
evaluation examples deterministically by source/capability strata and seed
140513, and leaves 898 training examples. The assignment SHA-256 is
`59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394`.

This is an instruction-stage plan, not a base-pretraining corpus. No compatible
VASU-140M base checkpoint is selected, all planned outputs are absent, and
release construction and training remain unauthorized. GPT-5.5 independently
accepted the exact plan for future release construction review; the next gate
is a separately reviewed, non-overwriting construction implementation.

The next gate now has an engineering-complete fixture-only implementation. It
accepts at most 30 caller-supplied logical examples, rejects the planned
production paths, writes token/mask/manifest files into an isolated staging
directory, and atomically publishes only the complete directory. Tests prove
deterministic replay, mask layout, overwrite rejection, identity rejection,
tamper detection, and cleanup after injected failure. No production data was
constructed. GPT-5.5 independently accepted the exact fixture layer as
construction evidence on 2026-07-30. The acceptance is non-authorizing; a
separately designed and reviewed production builder remains required before
any production release construction.

A specification-only production-builder design now proposes separate read-only
qualification and explicitly authorized publication phases. It defines
immutable lineage, one-build authorization, atomic staging, incomplete-
publication quarantine, Windows path defenses, deterministic full-source dry
runs, and complete mask audits. GPT-5.5 independently accepted the design on
2026-07-30 for a separately reviewed implementation proposal. The decision is
not production release or training authority.

The accepted design now has an engineering-complete implementation proposal.
Read-only qualification compiled and decoded all 996 eligible examples twice
with identical evidence, reproduced the frozen 898/48/50 assignment, and
performed complete serialized mask and boundary audits. The publisher requires
an exact self-hashed, expiring one-build authorization plus a clean reviewed
commit, and its temporary-root tests cover disk, Windows junction/open-handle,
mutation, atomicity, incomplete publication, receipt, and reuse failures.
GPT-5.5 independently accepted the implementation on 2026-07-30. The reviewed
qualification is intentionally pre-commit evidence; after the accepted package
is committed, qualification must be regenerated against that exact clean
commit and independently identity-reviewed. No authorization record or
production output exists.

The accepted implementation is committed as `c014716`. A clean-commit,
read-only qualification reproduced twice and preserves the exact source,
assignment, logical-record, token, and mask identities. Its new qualification
SHA-256 is
`ed6f64b9d5cb97925bc68186b4ec403de6e25dc484e33cb103ed05ee977f5cd6`.
Only commit-bound manifest and qualification identities changed. GPT-5.5
independently accepted the exact transition on 2026-07-30. This acceptance is
non-authorizing: no authorization record exists, and publication and training
remain prohibited.

The post-commit smoke remains valid on documentation-only descendants: it
requires `c014716` to be an ancestor and independently rechecks the exact
implementation-file SHA-256. This avoids invalidating reviewed code merely by
committing its accepted evidence.

The VASU-140M one-build protocol was independently accepted on 2026-07-30. The
one-build protocol uses a detached, self-hashed authorization envelope that
separately binds the accepted implementation anchor and an exact clean runtime
commit. This removes Git self-reference while preserving fail-closed,
single-use publication. It still requires a separately reviewed implementation,
a future explicit human approval, and a separate explicit publication
instruction. The accepted review packet contains a non-blocking fixture-path
typo documented in its audit; the reviewed packet remains immutable. No
envelope or production artifact exists.

The separately isolated v2 authorization gate is now implemented with canonical
detached-envelope loading, exact runtime and code identities, exclusive
single-use locking, conservative stale-lock recovery, direct v2 receipt
binding, and adversarial transaction tests. The accepted production builder
remains byte-identical. GPT-5.5 independently accepted the exact pre-commit
implementation on 2026-07-30. A clean post-commit qualification and independent
identity review remain mandatory. The accepted implementation is committed as
`050d1fa`; its clean post-commit qualification preserves all code and release
identities, with only the expected commit-bound qualification transition.
GPT-5.5 independently accepted that exact transition on 2026-07-30. The
implementation creates no envelope and grants no publication or training
authority.

The v2 post-commit identity smoke is documentation-descendant-safe: it requires
`050d1fa` in ancestry, rechecks exact gate/test/smoke hashes, and compares the
normalized commit-bound report to the accepted frozen identity.

The final runtime-eligibility procedure is now prepared. Its read-only smoke
requires a clean worktree and index, emits the exact runtime commit plus all
accepted identities, and creates no repository artifact. The independent
runtime decision must remain detached from Git to avoid changing the commit it
reviews.

Pratham Sharma subsequently authorized one production publication through the
detached envelope
`vasu-140m-instruction-seed-v1-20260730-001`. The transactional publication
completed once from runtime commit `5cf4188`: 996 examples in exact 898/48/50
splits, with a consumed receipt and `training_authorized=false`.
`validate_published_release` passes and publication status is `complete`.
GPT-5.5 independently accepted the completed publication on 2026-07-31. The
review packet date-path erratum is documented in the publication audit. No
checkpoint or training plan is selected.

## Capability-CPT schedule status

The generalized deterministic N-source schedule layer and Candidates A/B/C
remain hash-bound. Candidate C and Candidate A are complete; Candidate B
remains unauthorized. Candidate A and Candidate C share the FineWeb step-200,000
parent, 20,004,864-token budget, 78,144 records, 39,072 microbatches, 2,442
updates, 2,442-step scheduler, 49-update warmup, 256-token sequence length,
batch size 2, and accumulation 16; their intended experimental difference is
Candidate A's 5% verified-arithmetic-v2 allocation. Source validation, mixed
masked/unmasked batches, validation isolation, schedule-aware exact-resume
metadata, and 2,442-update accounting remain preserved.

## Capability evaluation framework

The additive `evaluation.framework` internal capability suite is available for
checkpoint-comparable greedy or fixed-seed sampled evaluation. It records
versioned suite, checkpoint, tokenizer, and generation metadata while keeping
objective structural metrics, heuristic signals, and human-review requirements
separate. It does not authorize further training or promote a checkpoint by
itself.

## FineWeb extension recovery and coverage

The 2026-07-17 bounded source-ID smoke selected 100 evenly spread historical
extension IDs and retrieved them through the official Dataset Viewer exact-ID
filter pinned by the response revision header. All 100 IDs were found and all
100 `SHA-256(text.strip())` values matched; all failure categories were zero.
The method inspected 498,312 text bytes and downloaded 645,735 JSON response
bytes. This smoke is historical evidence; it was superseded by the completed
production recovery described below.

The follow-up 1,000-ID benchmark passed exact correctness for serial,
concurrency 2/4/8, and documented OR batches of 5/10/25. Batch 25 with
concurrency 1 is the selected production design: 40 requests and 37.33 seconds
for 1,000 IDs, with zero 429s, transient errors, or retries. A conservative
2 RPS production ceiling is recommended. This benchmark selected the production
strategy used for the completed full recovery.

Production recovery completed for all 379,247 retained extension source IDs
against the pinned `HuggingFaceFW/fineweb-edu` `CC-MAIN-2025-26` revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. All 379,247 recovered
`text.strip()` hashes matched the historical fingerprints, with zero missing
IDs, hash mismatches, duplicate accepted IDs, malformed records, or unrecovered
retrieval errors. The compressed recovered artifact is
`data/interim/pretrain/fineweb_extension_recovered.jsonl.gz`, size
797,342,910 bytes, SHA-256
`c90f9e21d9b73324b9165cf1fb7ffbc274fbba5ac22b7cbe48abb6d7f1e`.

The extension document index is complete at
`data/manifests/pretrain/fineweb_extension_document_index.sqlite3`, with
379,247 indexed documents, 3,033,976 LSH buckets, zero rejected records, zero
duplicate hashes, SQLite integrity `ok`, and SHA-256
`d093770179204b45af1a5a824d5a5826a6f0626b3c8825aa81ac8d3557acf16e`.
Combined coverage is recorded in
`data/manifests/pretrain/fineweb_combined_coverage.json`, covering both
`fineweb_original` and `fineweb_extension` with no missing coverage and
`training_ready: true`. No model training was started by the recovery or
indexing workflows.

## FineWeb document-level deduplication status

The production original-source index is complete. It contains 999,992 unique
documents from 1,000,000 JSONL records, collapses eight normalized duplicates,
and contains 7,999,936 LSH bucket entries. The 4,876,034,048-byte SQLite file
has SHA-256 `07508a0fe83023cfe0a624ae1af21cba4c2e6dffe62f83b7078656bb168888cd`.
All indexed records use stable line-based IDs and explicitly incomplete
provenance because the source JSONL retained only `text`.

The FineWeb extension document text was recovered and indexed separately from
the protected token binary. The combined coverage manifest now allows the
factual preparation pipeline to use both original and extension FineWeb
coverage under normalization version `vasu_cross_source_nfc_casefold_ws_v1`.

Read-only comparison of the 50-chunk Wikimedia broad-review artifact against
the combined original-plus-extension FineWeb indexes found no exact,
high-confidence near, or ambiguous overlaps. The report is
`data/manifests/factual/wikimedia_fineweb_combined_overlap.json`. This result
does not make the review artifact training data.

## Next-generation planning status

An evidence-based next-generation plan is recorded in `docs/VASU_NEXT_PLAN.md`. The recommended path is a capability-focused continuation of the preserved VASU-60M FineWeb step-200,000 **base** checkpoint, gated first by small data-mixture ablations. The planning target is approximately 1.2B additional new token positions at the existing 58,337,792-parameter architecture and 256-token context.

This is planning only. No architecture, runner, tokenizer, dataset, checkpoint, or training process has been changed or authorized. A future approximately 100M model remains conditional on proving the new data mixture and curriculum with the existing 60M model.

### Factual pilot source gate

The factual pilot now passes the metadata approval gate using the official
`wikimedia/wikipedia` distribution, configuration `20231101.en`, split
`train`, pinned at commit
`e6057dc557255a03c9c3c47ceab0eb44353b1bc5`. The source is approved only for
bounded acquisition and preparation under the recorded CC BY-SA/GFDL
attribution and redistribution obligations. One pinned 420,296,449-byte shard
has been acquired for a bounded preparation smoke test; no Wikimedia data has
been used for training. The capability pilot remains blocked by its
mathematics, code, and reasoning source reviews.

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

The 2026-07-30 read-only CPU FineWeb loader measurement used three matched
replicates, batch size 2, sequence length 256, eight warm-up batches, and 40
measured iterations. `workers=0` without pinning had median throughput
12,711,994 tokens/s. Requested pinning had no meaningful CPU-only effect
(-0.88%; CUDA was unavailable), while `workers=2` reduced median throughput by
86.10%. The local CPU decision is to retain `workers=0`, reject the two-worker
variant, and defer KV-cache adoption pending a matched CUDA benchmark. See
`docs/PERFORMANCE_PHASE3_CPU_LOADER_20260730.md` and the retained artifacts in
`evaluation/results/performance_phase3_cpu_loader_20260730/`.

Compatibility remains unchanged: KV cache does not alter architecture parameters, trained weights, tokenizer, prompt templates, datasets, training behavior, model-state keys, or checkpoint schema.

The Phase 4 architecture decision and implementation-readiness boundary are
documented in `docs/VASU_140M_FAMILY_PROPOSAL.md` and
`docs/VASU_140M_IMPLEMENTATION_READINESS.md`. The isolated
`vasu_140m_v1` configuration (768 width, 12 layers, 12 heads, 3072 SwiGLU
width, 512-token context) is now registered under a canonical config/family
fingerprint and its exact 137,841,408-parameter contract passes meta-device
construction. VASU-31M/60M defaults, checkpoint containers, tokenizer, data,
masks, training, and resume behavior are unchanged. Existing checkpoints are
not tensor-compatible with this family.

The bounded VASU-140M CPU qualification passed on 2026-07-30. One synthetic
FP32 batch of shape `[1, 8]` produced finite logits, loss, and gradients for
all 110 parameter tensors without creating an optimizer or updating weights.
Dynamic and preallocated KV caches matched uncached logits across prompt
prefill and three decode steps with maximum absolute error `5.72e-06`. See
`docs/VASU_140M_CPU_QUALIFICATION_20260730.md`. CUDA,
exact-resume, 513-token data, and frozen evaluation gates remain open.

The additive VASU-140M model-only checkpoint gate also passed on 2026-07-30.
An atomic 551,408,411-byte temporary checkpoint strictly restored all 111
state tensors bit-for-bit into a fresh model and preserved weight tying.
Wrong declared-family and wrong destination-config cases were rejected before
state loading, and the temporary checkpoint was removed with disk space fully
restored. Legacy checkpoint containers are unchanged. See
`docs/VASU_140M_CHECKPOINT_QUALIFICATION_20260730.md`. Exact resume and CUDA
checkpoint I/O remain unqualified; training remains unauthorized.

The first VASU-140M synthetic exact-resume run then exposed a real stochastic
resume defect: recreating a DataLoader iterator consumed global PyTorch RNG,
so dropout caused model and AdamW divergence despite correct sampler and
partial-gradient restoration. The failed artifact is preserved. Training and
validation loaders now use dedicated generators, isolating loader bookkeeping
from model RNG. The corrected v2 run matched model, AdamW, scheduler, scaler,
sampler, sample order, progress, partial gradients, and Python/NumPy/PyTorch
RNG state after resuming a 1,102,823,453-byte mid-accumulation checkpoint.
All temporary artifacts were removed. See
`docs/VASU_140M_EXACT_RESUME_QUALIFICATION_20260730.md`. This qualifies only
the frozen synthetic CPU workload; CUDA/AMP, data, evaluation, and training
authorization gates remain closed.

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

Hard pilot limits are one shard, 10,000 inspected rows, 2,000 accepted parent
documents, 4,000 accepted chunks, 2,000,000 exact VASU-tokenizer tokens, and
1 GB downloaded. Parent-document and chunk limits are independent; the former
ambiguous `max_accepted_documents` setting historically counted accepted
chunks and is now accepted only as a deprecated legacy chunk-limit alias. The
pipeline records provenance, explicit filter reasons, exact and bounded
near-duplicate checks, evaluation-prompt contamination evidence, hashes,
atomic progress, and deterministic resume/restart state. Generated raw,
interim, processed, and factual-manifest artifacts are ignored by Git.

Manual review of the first smoke artifact exposed apparent mojibake, possible
joined words, and oversized full-article records. Code-point tracing proved
the pinned Parquet and UTF-8 JSONL contained correct Unicode; mojibake appeared
only when the JSONL was displayed through an incompatible Windows decoder.
The old citation-only cleanup still had a real boundary risk because it used
empty-string replacement and did not handle references or templates.

The v2 pipeline now performs conservative reversible encoding repair only
when corruption markers demonstrably decrease, rejects replacement/control or
low-confidence corruption, preserves inline word boundaries, and chunks before
deduplication, contamination checks, and token accounting. Chunk settings are
768 target, 1,024 maximum, 128 minimum, and 32 overlap tokens.

The corrected smoke inspected 2 rows and retained 20 distinct chunks with
17,237 tokens; the maximum was 1,022 tokens. It recorded 0 encoding repairs,
0 quality rejections, no replacement characters, and no detected joined-word
regressions. Output validation passed. The earlier default artifact retained
2,000 chunks and 1,036,527 tokens; it remains valid historical evidence but
was limited by the ambiguous chunk counter rather than by the intended token
budget.

The refactored default pilot completed with status `token_limit` after 279 raw
examples, 262 accepted parent documents, 3,751 accepted chunks, and 1,999,974
VASU tokens. Its maximum chunk is 1,024 tokens, replacement-character count is
zero, and output SHA-256 is
`cdccaff4d305dc4cfd273c3e94f5376c349773c317c5720227db97745f1d646a`.
One high-confidence near overlap with FineWeb was rejected before accounting.
The v3 manifest and output validator passed. This is a prepared pilot artifact;
no factual-pilot model training has started. Cross-FineWeb document
deduplication uses the combined original-plus-extension coverage manifest and
indexes.

Deterministic manual-review tooling now verifies the immutable input SHA-256
before selecting bounded previews. Seed 42 selected 79 unique chunks: 20
random, 10 shortest, 10 longest, 10 nearest the 1,024-token ceiling, 10 evenly
spaced, 10 from distinct parents, all 21 reference-type sections, and 2
suspicious-metadata records; 14 overlapping selections were deduplicated. The
sample represents 62 parent articles. All 79 classifications remain pending,
so the factual pilot is not yet manually approved and no training has started.

That first manual-quality gate failed: it exposed 21 retained reference-type
chunks, 18 below-minimum chunks, malformed overlap starts, list-dominated
content, and missing source/template values. Its reports are preserved under
the old `cdccaff4d305dc4` dataset hash.

The corrected v4 pipeline now excludes canonical structured or embedded
reference appendices, merges or rejects chunks below 128 tokens, records
paragraph/sentence/word-fallback boundary metadata, omits unsafe mid-sentence
overlaps, and rejects list-dominated, missing-value, malformed-source, and
low-information chunks before deduplication or accounting. The regenerated
  artifact reached `token_limit` with 330 raw examples, 304 parents, 3,480
  chunks, and 1,999,700 tokens. Its observed range is 128–1,023 tokens,
  SHA-256 is
  `4c21e6b54cf747a82769951e97222cbe161296e7c0556ee14125df93cc1a4fd0`,
and validation found zero retained reference headings, replacement/mojibake
markers, or token-fallback boundaries.

The regenerated seed-42 review contains 61 pending chunks from 53 parents,
including one warning-forced record. It has zero reference-forced records,
zero below-minimum failures, and zero automatic precheck failures. Human
classification is still required; no Wikimedia training has started.

### Final global-audit release candidate

All defect families recorded across the completed hash-bound reviews are now
covered by a reusable global scan. The final permitted remediation cycle
produced 402 parent documents, 3,609 chunks, and 1,999,697 tokens with dataset
SHA-256
`ffcbc25f4863f519744212f809ee600bdc7f4a0d5c2d02a0833e1bc4cec6014d`.
Output validation passed. The final global audit has zero automatic-reject
findings and zero unexplained matches; its remaining 29 findings are explicitly
manual-review cases rather than silently accepted defects.

The frozen seed-42 review contains 84 pending chunks and zero automatic
precheck failures. It is immutable for this candidate and may be replaced only
if human review finds a critical reject. Promotion follows
[`WIKIMEDIA_RELEASE_POLICY.md`](WIKIMEDIA_RELEASE_POLICY.md): manual review must
finish with zero critical rejects, and any harmless minor issue must carry a
note. The candidate is not approved and Wikimedia training has not started.

The frozen human decisions subsequently identified 12 exact critical rejects.
They were excluded through a hash-bound quarantine transformation without
changing the source candidate, review decisions, chunk IDs, ordering, or
quality heuristics. The approved quarantine release has SHA-256
`6aa10d73669ca90ad20f867f14a6368d2191b38094f02aa1679ebaf122962de7`,
402 parents, 3,597 chunks, and 1,993,564 tokens. It quarantines 6,133 tokens
across 9 parents and removes no parent completely. Deterministic output and
quality validation passed with zero automatic rejects and zero unexplained
findings. No new review sample was generated, and no Wikimedia training has
started or been authorized.

### Deterministic broad-review sample

A separate review-only mode now selects 500 unique, shard-spanning row indices
using evenly spaced positions with a seed-local deterministic offset. It reads
only the required Parquet row groups, retains at most 5 chunks per parent, and
writes review-suffixed artifacts without touching smoke/default outputs.

The validated review inspected 15 selected rows and retained 50 chunks from 14
parent articles (26,435 tokens). Chunk tokens were 43 minimum, 470.5 median,
528.7 mean, and 938 maximum. The largest article contributed 10%; 3
reference-section chunks were flagged. Encoding repairs, quality rejections,
contamination matches, and exact/near duplicates were all zero. Validation and
bounded UTF-8 preview review passed. One malformed date boundary was traced to
the pinned Parquet source itself, so uncertain factual text was not silently
rewritten. This artifact is for review, not training; the full pilot remains
unauthorized.

### Wikimedia factual continued-pretraining pilot prepared

An isolated VASU-60M continued-pretraining experiment is prepared from
`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`. It uses an immutable
10,000,000-token cap with 85% sequential FineWeb-Edu and 15% Wikimedia,
seed 42, and no replacement sampling. The 256-token alignment yields 39,062
records and 9,999,872 supervised tokens: 8,499,968 FineWeb tokens and
1,499,904 Wikimedia tokens (about 0.783 of the Wikimedia training split).

Wikimedia was split by parent before tokenization: 382 training parents and 20
validation parents, with zero overlap. The resulting training and validation
streams contain 1,915,008 and 92,944 tokens. The mixed binary SHA-256 is
`60a06cc54f0c77b977db733829584786edbe518eda38eab83964887d9a12bc4b`;
the source schedule SHA-256 is
`f426c1e3a301bad4b451c2b6d8b1bea2fe567dd262d493fadf5c25e368fef6b0`.
A clean temporary rebuild reproduced both hashes.

The experiment uses a new optimizer and cosine scheduler rather than falsely
resuming the parent's optimizer clock. Provenance remains
`parent_global_step=200000`, while experiment-local step starts at zero. Batch
size 2, accumulation 16, AMP, 1e-5 peak learning rate, 1e-6 minimum, 25 warmup
steps, weight decay 0.1, and clipping 1.0 are configured. Expected duration is
1,221 optimizer steps. `training_authorized` is false; no optimizer update or
new checkpoint has been produced.

### Factual CPT completed and evaluated

The experiment later completed all 1,221 optimizer steps and consumed exactly
9,999,872 supervised tokens. The selected checkpoint is
`checkpoints/vasu_60m/factual_cpt_wikimedia_15pct_from_200k/best.pt` at
experiment step 1,200; `latest.pt` records the final partial update at step
1,221. FineWeb validation improved from 3.356130 to 3.317529 and held-out
Wikimedia improved from 3.393557 to 3.286295.

The initial six-prompt sampled check was insufficient. The frozen 300-example
factual-CPT v2 benchmark therefore measures 100 cloze items, 100 direct-
likelihood multiple-choice items, 50 continuations, and 50 qualitative prompts
under greedy and seeded sampling. Raw cloze stayed at 6%, normalized cloze
moved from 7% to 9%, and raw/length-normalized multiple choice remained 41%/
36%. Bootstrap intervals do not distinguish the factual changes from benchmark
noise. The branch is preserved as an evaluation candidate but is not promoted.
No further CPT or instruction tuning is authorized by this result.

Closeout metadata records `completed_experiment=true`,
`promotion_status=rejected`, `recommended_for_further_cpt=false`,
`recommended_for_instruction_tuning=false`, and
`artifact_retention=preserve` beside the selected checkpoint. The experiment
configuration has been returned to `training_authorized=false`, preventing an
accidental repeat while preserving every checkpoint and evaluation artifact.

The approved Wikimedia quarantine release remains an immutable data artifact;
rejecting this training branch does not revoke or alter that dataset approval.
The authoritative instruction lineage remains
`checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt` (SHA-256
`c5da8e1f95f84ad391338548ab777d2aabf3f931f7f6c5aff54c040caef63c43`).
It does not use the factual-CPT checkpoint.

## Instruction-quality pilot pipeline

The interactive-quality audit found inference, tokenizer round-tripping,
prompt alignment, and response-mask alignment correct. The primary limitation
is instruction-data quality and coverage; model capacity and limited factual
knowledge remain secondary constraints. Immediate retraining is not
authorized.

The `vasu_instruction_quality_v1` engineering pipeline is implemented with a
21-example demonstration fixture covering all seven planned capabilities. All
21 records pass automatic schema and constraint validation, with zero exact
duplicate groups and zero near-duplicate candidates at threshold 0.85. Human
review subsequently approved all 21, and the demonstration release produced
five packed records containing 1,285 tokens. Automatic scoring did not approve
any example; the preserved human decisions remain hash-bound to source records.

The planned production pilot remains 2,000-5,000 human-approved examples with
the configured capability distribution. The source and review manifests are
hash-bound, `training_authorized` remains `false`, and no training or optimizer
update has occurred. Existing Alpaca, UltraChat, FineWeb, Wikimedia, tokenizer,
checkpoint, and training artifacts remain unchanged by this pilot.

### Instruction-quality production batch 001

The 21-example demonstration completed human review and release successfully:
21 approved examples, five packed records, and 1,285 packed tokens. Its source,
decisions, and release are frozen and were not modified by batch 001.

Batch 001 is the first production-candidate authoring tranche. It contains 500
purpose-written, unreviewed records: 125 short factual answers, 100 beginner
explanations, 100 exact-format examples, 75 rewrites, 50 structured lists, 25
strict JSON examples, and 25 uncertainty fallbacks. The difficulty mix is 350
easy and 150 medium, with no hard examples.

All 500 records pass the existing schema and format validator. Exact duplicate,
near-duplicate, demo-leakage, prompt-marker, truncation, JSON, format, and
factual-verification findings are zero. The longest complete token audit is 65
tokens under the unchanged 257-token record capacity. All records remain
`unreviewed`, the separate review file is empty, no production `.bin` or mask
exists, and `training_authorized` remains `false`. The next gate is human review.

The first deterministic 100-example human gate did not pass: 77 examples were
approved for that gate, 15 needed fact checking, and eight needed rewriting.
Reviewers identified broad factual pages that did not bind directly to claims,
generic list filler, one ambiguous measurement question, one awkward beginner
prompt, and two transformation-fidelity failures.

Batch 001 has therefore been rebuilt without transferring any gate decision.
All 125 factual records now cite claim-specific pages; all source URLs were
replaced. Twenty beginner prompts, eleven exact-format/filler examples, and
three rewriting examples were also corrected. In total, 159 source records
changed and 341 remained byte-equivalent at the canonical record level. Of the
100 gate-v1 decisions, 38 are stale because their underlying records changed.

The repaired source again validates at 500/500 with zero exact or near
duplicates, zero demo collisions, and zero truncations. A fresh deterministic
gate-v2 packet contains the same 25/20/20/15/10/5/5 category distribution and
no decisions. Human review remains pending; training is still unauthorized.

The masked UltraChat experiment had already completed before this closeout, so
the historical “blocked pending baseline” gate is no longer a current-state
description. Its preserved output directory is therefore intentionally not
clean. The parent baseline is now recorded reproducibly at
`evaluation/results/vasu_60m_ultrachat_parent_alpaca_v3_baseline.json`, derived
from the identical 40-prompt greedy and sampled reports used after UltraChat.
Any future rerun is unauthorized unless explicitly launched with the isolated
runner's `--train` flag.

### UltraChat promotion evaluation

The completed UltraChat branch has now been evaluated against masked Alpaca v3
with a frozen 216-prompt benchmark, identical greedy and controlled-sampling
settings, and the held-out UltraChat and Alpaca masks. UltraChat validation loss
improved from 2.693622 to 2.581431, while Alpaca validation loss moved from
2.522585 to 2.543987 (+0.85%). Empty-output and premature-EOS rates remained
zero.

Promotion is rejected. UltraChat reduced strict format compliance from 7.87%
to 0% under greedy decoding and from 16.20% to 4.17% under controlled sampling.
Mean repetition increased from 0.7261 to 0.7717 greedy and from 0.3424 to
0.3922 sampled; the obvious-incoherence heuristic also worsened in both modes.
The 60-prompt stratified human-review form remains blank, so no clear semantic
or conversational improvement has been established. Masked Alpaca v3 remains
the main instruction-tuned VASU-60M checkpoint. No new training stage is
authorized.

Artifacts: `evaluation/benchmarks/ultrachat_promotion_v1.json`,
`evaluation/results/ultrachat_checkpoint_audit_v1.json`,
`evaluation/results/ultrachat_promotion_v1.json`, and
`evaluation/results/ultrachat_promotion_v1_manual_review.txt`.

### Instruction-quality Batch 002 closeout

Batch 002 completed as an isolated masked refinement from the Batch 001 best
checkpoint. It processed 90 training records in three optimizer updates,
finishing at global step 201307. Train supervised-token loss was 3.018085 and
validation supervised-token loss was 3.189959 over 442 validation tokens.

The resulting checkpoint is preserved at
`checkpoints/vasu_60m/instruction_quality_batch_002_from_batch_001/best.pt`
with SHA-256
`5025c1690035f0f0139bb2cff4044fd4a0ef6df0e480eee71d96a12e61272262`.

All automatic safety gates passed. Batch 002 showed small improvements in
repetition, diversity, controlled-sampling format compliance, and Alpaca
validation loss, while UltraChat loss remained effectively stable. However,
the 60-prompt semantic review preferred Batch 001 eight times and Batch 002 six
times, with 46 ties.

Promotion is rejected because Batch 002 did not demonstrate a clear semantic
improvement. Batch 001 remains the preferred instruction-quality checkpoint.
Batch 002 and its evaluation artifacts are retained for reproducibility; no
additional training is authorized by this result.

### Training-system reliability: exact general Trainer resume

The general shuffled `Trainer` now supports deterministic mid-epoch resume.
New additive checkpoint metadata preserves the sampler seed/epoch/next-batch
position, partial gradient-accumulation state, AMP scaler state, and process
RNG states. Older checkpoints remain loadable with an explicit warning because
they cannot establish an exact mid-epoch position. No model, tokenizer,
processed dataset, or existing checkpoint tensor key changed.

### Capability CPT ablation preparation

Three matched, unauthorized 20M-nominal capability-CPT plans are prepared in
`configs/data/mixtures/`. They retain the step-200k parent, use no replacement
sampling, and cap the approved Wikimedia share at 9% because the release has
only 1,915,008 train tokens. Canonical verified-arithmetic v1 is now tokenized
and validated: 80 unique training examples, nine fixed records, 2,081 real
tokens, 232 PAD tokens, and isolated 20-example development and evaluation
splits. It remains ignored generated data with `training_authorized=false`.
The generalized three-source scheduled-mixture builder remains absent, so
Candidates A/B/C are not executable.
