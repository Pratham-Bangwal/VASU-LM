# VASU Next-Generation Training Plan

## Purpose and decision boundary

This document plans the next VASU training generation from measured repository evidence. It does not authorize implementation, dataset creation, checkpoint conversion, or training.

Authoritative starting points:

- base model: VASU-60M, 58,337,792 parameters;
- base checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`;
- preferred assistant: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`;
- experimental conversation branch: `checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt`;
- hardware: RTX 4050 Laptop GPU (6 GB), Intel i5-13420H, 16 GB RAM, Windows;
- tokenizer: existing 32,000-token `assets/tokenizer.json`;
- context: 256 tokens.

The current VASU-60M uses a decoder-only, pre-norm Transformer with RoPE, RMSNorm, SwiGLU, tied embeddings, bias-free linear layers, and PyTorch scaled-dot-product attention. Its configuration is dimension 512, 10 layers, 8 heads, and SwiGLU hidden dimension 2,048.

## Evidence baseline

The five-seed 40-prompt sampled comparison is the strongest current stability evidence:

| Checkpoint | Automatic average | Mean repetition | Reasoning | Factual knowledge | Programming |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alpaca masked v3 | 0.279 | 0.356561 | 0.000 | 0.075 | 0.025 |
| UltraChat masked v2 | 0.301 | 0.370466 | 0.000 | 0.075 | 0.025 |

These are heuristic task-compliance scores, not intelligence scores. They show that neither instruction branch reliably solves simple factual, arithmetic, or programming tasks. UltraChat improves mainly conversation and formatting, while Alpaca remains the default because it has lower repetition and stronger relative behavior in several non-conversational categories.

The base checkpoint has processed approximately 1.6384 billion token positions under the established mapping (`200000 × 2 × 16 × 256`). The current binary training stream has only about 107.5 million unused tokens after step 200,000, so a meaningful continuation needs new, provenance-recorded data rather than another long replay of the same stream.

The measured pre-boundary run took about 1,459 seconds for 1,800 optimizer steps, or roughly 0.81 seconds per step before orchestration and cooldown overhead. Current training uses about 1.38 GiB peak allocated CUDA memory, but sustained laptop temperature—not VRAM—has historically been the tighter operational limit.

## Root-cause analysis

### Factual knowledge

Likely contributors, in descending practical importance:

1. **Data composition.** FineWeb-Edu provides broad educational web text, but it is not a controlled factual corpus. The dataset's own card recommends complementing it with curated sources such as Wikipedia. The current extension also changes crawl period rather than adding a distinct factual domain.
2. **Insufficient stable factual exposure.** Approximately 1.64B token positions is meaningful for 60M parameters, but repeated exposure to web-style text does not guarantee coverage or retention of elementary facts.
3. **Capacity.** A 60M model cannot store or robustly retrieve the breadth of facts expected from modern assistants. More targeted data can improve common facts but will not make it an authority.
4. **Context length.** A 256-token context limits document-level integration and reading comprehension. It does not explain failures on isolated facts such as capitals or planets.
5. **Instruction-stage forgetting.** Assistant-only tuning can shift distributions, and UltraChat caused category regressions. However, the base and earlier instruction evaluations were already weak, so catastrophic forgetting is a contributor rather than the sole cause.

### Reasoning and arithmetic

- The pretraining mixture contains no measured, controlled proportion of arithmetic, code, or programmatically verified reasoning examples.
- Alpaca and UltraChat are broad instruction/conversation corpora, not verified arithmetic curricula.
- A 60M model should be capable of learning frequent, short arithmetic patterns and simple algorithm templates. Consistent zero scores therefore indicate data/curriculum absence as well as a capacity ceiling.
- Exact arithmetic needs programmatically checked examples and held-out templates. More unverified chain-of-thought text may teach plausible-looking errors.
- Scaling alone is not a sufficient diagnosis: a 100M model trained on the same mixture could reproduce the same failure pattern at higher cost.

### Repetition

The likely causes are cumulative:

- limited model capacity and a 256-token window;
- duplicated or low-entropy patterns in web and instruction data;
- instruction examples truncated to fixed records;
- one-epoch UltraChat data containing many truncated responses;
- over-representation of templated answers;
- weak EOS/termination behavior in older datasets, although masked v3 and UltraChat v2 now supervise EOS correctly;
- sampling configuration, which changes visible repetition but cannot repair the learned distribution.

Overfitting is plausible in narrow instruction stages, but it has not been proven with a train/validation divergence study. Repetition should be tracked by category and seed at every stage rather than attributed to architecture by default.

### Instruction following

- Prompt-template consistency and response-mask alignment are now validated, so the poor results are not explained by a formatting bug.
- Masked assistant-only loss and supervised EOS improved the formulation but did not create missing capabilities.
- Existing instruction data lacks enough verified exact-format, arithmetic, uncertainty, and code tasks.
- The UltraChat pipeline has a high truncation count, which weakens conversation endings and multi-turn consistency.
- The curriculum moves quickly from broad single-turn instruction data to conversation. A better sequence is general instruction, verified task skills, then a small conversational stage with replay from earlier stages.

## Option comparison

### Option A — Continue VASU-60M with a capability-focused mixture

| Dimension | Assessment |
| --- | --- |
| Expected quality improvement | Moderate and most directly targeted. Best chance of improving common facts, elementary arithmetic, code vocabulary, and instruction readiness without discarding learned language ability. Still bounded by 60M capacity. |
| Training duration | About 35–60 wall-clock hours for 1.2B additional tokens, including block startup, validation, and cooldown overhead. |
| VRAM feasibility | Proven at batch 2, sequence 256, accumulation 16, AMP. No architecture-memory increase. |
| RAM/storage | 16 GB RAM is adequate with streaming/memmap preparation. Expect roughly 2.4 GB for 1.2B `uint16` tokens, plus metadata, source/cache staging, and 2–4 GB of retained checkpoints. Reserve 10–20 GB incremental workspace. |
| Complexity | Medium: new data auditing, mixture manifests, deterministic sampling, and stage-aware evaluation; little model-code work. |
| Failure risk | Medium. Main risks are bad mixture quality, domain forgetting, duplicate contamination, licensing errors, and thermals. |
| Dataset requirement | Approximately 1.2B new, deduplicated tokens across general, factual, code, math, and verified reasoning domains. |
| Checkpoint compatibility | Full model-state compatibility with the step-200,000 base. Optimizer state can load, but a new explicitly defined LR schedule should not silently inherit the old constant-LR assumptions. |
| Tokenizer compatibility | Full. Measure code/math fragmentation before accepting the mixture. |
| Evaluation | Existing 40-prompt greedy and five-seed suite plus held-out factual, arithmetic, code, and reading-comprehension sets. |
| Scalability | Good as a data/curriculum validation step. The same mixture can later train a 100M model. |

### Option B — Train VASU-60M v2 at approximately the same size

A plausible research configuration is dimension 480, 12 layers, 8 heads, hidden dimension 1,920, and context 512 (approximately 60M parameters). It trades width for depth, but there is no VASU evidence that this trade improves quality.

| Dimension | Assessment |
| --- | --- |
| Expected quality improvement | Uncertain. Better schedule/data may help, but those gains do not require new tensor shapes. Context 512 helps longer examples, not isolated factual/arithmetic failures. |
| Training duration | Roughly 50–90 hours for a fresh 2–3B-token run, depending on the 512-token batch that fits. |
| VRAM feasibility | Likely feasible only after a CUDA smoke test; attention activation cost grows with sequence length. Batch 1 with accumulation 32 may be needed. |
| RAM/storage | 16 GB RAM should remain adequate with streaming; expect 4–8 GB token binaries plus staging/cache and current-size checkpoints. |
| Complexity | High: new config, data records, smoke tests, checkpoint isolation, context-aware evaluation, and a full restart. |
| Failure risk | Medium-high because the width/depth change is unvalidated and the run cannot fall back to the current weights. |
| Dataset requirement | At least 2–3B well-mixed tokens to compare fairly. |
| Checkpoint compatibility | Width/depth changes break model and optimizer compatibility. A context-only increase can load weights because RoPE buffers are non-persistent, but fixed 257-token instruction records and loaders require review. |
| Tokenizer compatibility | Full if vocabulary remains 32,000. |
| Evaluation | All current tests plus 512-token recall/reading tasks and matched-token ablations against Option A. |
| Scalability | Moderate. Useful only if controlled ablations justify the architectural changes. |

### Option C — Scale to approximately 100M–120M parameters

A conservative candidate is dimension 640, 12 layers, 10 heads, hidden dimension 2,560, and context 512, approximately 100M parameters. Extending to 15 layers approaches 120M.

| Dimension | Assessment |
| --- | --- |
| Expected quality improvement | Highest eventual capacity, but only with a substantially better dataset. It will not automatically fix reasoning or factuality. |
| Training duration | Approximately 100–200 wall-clock hours for 2.5–3.5B tokens on this laptop, subject to smoke-test throughput and cooldowns. |
| VRAM feasibility | Plausible but unproven. Start with batch 1, accumulation 32, AMP, and possibly activation checkpointing. A realistic 512-token backward test is mandatory. |
| RAM/storage | 16 GB RAM is tight but workable with streaming and zero full-corpus loads. Expect roughly 1.1–1.5 GB per full training checkpoint and at least 15–30 GB free for data, cache, and retention. |
| Complexity | High: new architecture, memory work, longer orchestration, and full retraining. |
| Failure risk | High due to thermal duration, wall-clock interruptions, scheduler errors, and an unproven data mixture. |
| Dataset requirement | At least 2.5–3.5B high-quality mixed tokens; more is desirable for the capacity. |
| Checkpoint compatibility | Incompatible with VASU-60M tensors and optimizer states. |
| Tokenizer compatibility | Full if kept at 32,000. |
| Evaluation | Same suite plus scale-matched learning curves and long-context tests. |
| Scalability | Best long-term path after the data curriculum proves itself. |

### Option D — Scale down temporarily

| Dimension | Assessment |
| --- | --- |
| Expected quality improvement | Low for the final assistant; useful for fast pipeline and mixture ablations. |
| Training duration | Approximately 15–35 hours for a thorough 1.5–2B-token 31M–40M run. Small 10M–100M-token ablations complete much faster. |
| VRAM feasibility | Very safe. Enables larger batches or more frequent experiments. |
| RAM/storage | 16 GB RAM is ample for the model with streaming data. Checkpoints are smaller, but dataset storage is unchanged for a matched-token comparison. |
| Complexity | Low if reusing VASU-31M, medium for a new 40M shape. |
| Failure risk | Low operationally, high risk of spending time below the capacity needed for the target quality. |
| Dataset requirement | Same quality requirements; reduced scale does not excuse poor data. |
| Checkpoint compatibility | Existing VASU-31M is reusable; a new 40M model is not. |
| Tokenizer compatibility | Full. |
| Evaluation | Excellent for data-mixture ranking, weak evidence for final 100M capability. |
| Scalability | Valuable as a short ablation tool, not the main next generation. |

## Recommendation: Option A with an ablation gate

Continue the **base** VASU-60M step-200,000 checkpoint for approximately **1.2B new tokens**, targeting about **2.84B total pretraining token positions**. Do not continue from an instruction-tuned checkpoint. Before the full continuation, validate the mixture with short 20M–50M-token runs; these may use the current 60M checkpoint and do not require a separate small model.

This is the largest realistic near-term improvement because it addresses the best-supported causes—data composition and curriculum—while retaining a proven model, memory profile, checkpoint system, and tokenizer. It also produces a reusable mixture for a later 100M model. Option C becomes the next candidate only if Option A shows measurable skill gains without unacceptable forgetting.

## Recommended target configuration

| Field | Recommendation |
| --- | --- |
| Parameters | 58,337,792 (unchanged) |
| Architecture | Existing VASU-60M; no tensor-shape changes |
| Context | 256 for the continuation; evaluate 512 separately after the capability-repair run |
| Tokenizer | Reuse `assets/tokenizer.json`; audit fragmentation by domain |
| Additional pretraining | 1.2B new token positions |
| Total exposure target | Approximately 2.84B token positions |
| Batch/accumulation | Batch 2, accumulation 16, effective 32 sequences / 8,192 tokens per optimizer step |
| Pretraining LR | 500–1,000-step warmup to 5e-5, cosine decay toward 5e-6; validate scheduler resume exactly |
| Instruction LR | 3e-6 to 5e-6 for general/skill SFT; 1e-6 to 2e-6 for the final conversation stage |
| Operational checkpoints | Atomic every 50–100 optimizer steps with bounded retention |
| Milestones | Preserve every 25,000 optimizer steps; evaluate at each milestone |

An additional 1.2B tokens is about 146,485 optimizer steps at the current effective token batch. This is a planning estimate, not an authorized target in any runner.

## Proposed pretraining mixture

All source revisions, licenses, attribution requirements, hashes, filters, and removal policies must be recorded before preparation.

| Category | Share | Purpose | Quality filters | Main risks |
| --- | ---: | --- | --- | --- |
| General web text | 20% | Preserve broad language coverage and avoid narrow-domain forgetting. | Language confidence, deduplication, boilerplate/PII/toxicity filters, perplexity and length bounds. | Web noise, duplication, unstable facts. |
| High-quality educational text | 25% | Clear explanations, grammar, and basic concepts. | FineWeb-Edu quality threshold, document integrity, cross-source deduplication. | Classifier bias and repeated educational templates. |
| Wikipedia-style factual text | 18% | Stable entities, geography, science, history, and Indian/global coverage. | Current pinned dump, article integrity, attribution/provenance, remove lists/tables where serialization is poor. | Share-alike/attribution obligations, outdated facts, demographic bias. |
| Open/public-domain long-form prose | 8% | Longer discourse, topic retention, and varied style. | License verified per work and jurisdiction, chapter integrity, deduplication, OCR quality. | Copyright ambiguity and archaic language. Do not ingest merely because a text is downloadable. |
| Programming code and explanations | 10% | Syntax, variables, functions, loops, and code explanation. | Allowlisted permissive licenses, provenance, secret/PII scanning, near-deduplication, parsability. | Per-file license obligations, vulnerable/malicious code, tokenizer fragmentation. |
| Mathematics | 8% | Mathematical vocabulary and worked solutions. | Clear problem/solution boundaries, notation preservation, answer consistency, deduplication. | Incorrect web solutions and LaTeX/tokenization damage. |
| Verified synthetic reasoning | 4% | Arithmetic, sequences, comparisons, and constrained transformations. | Programmatic generation and answer verification; held-out templates and values. | Template memorization and low entropy. |
| Evidence-linked QA material | 4% | Connect factual text to concise questions and answers. | Derive from licensed source text, retain evidence IDs, reject unsupported answers. | Leakage into evaluation and hallucinated synthetic answers. |
| Indian and broader global knowledge | 3% | Reduce geographic imbalance and improve locally relevant common knowledge. | Balanced topic/language-source audit, factual provenance, deduplication against Wikipedia portion. | Tokenizer inefficiency outside English, cultural bias, uneven coverage. |

The FineWeb-Edu card identifies an ODC-By 1.0 release subject to Common Crawl terms. FineMath is also ODC-By 1.0 and is a candidate only after a pinned-subset audit. The factual pilot now selects the official `wikimedia/wikipedia` Parquet distribution, configuration `20231101.en`, pinned at commit `e6057dc557255a03c9c3c47ceab0eb44353b1bc5`. Its registry review records CC BY-SA 3.0/GFDL attribution and redistribution obligations, exact provider-published size/example counts, sequential-shard preparation, and required contamination/cross-FineWeb deduplication. Approval authorizes only later bounded acquisition and preparation; no Wikimedia data has been downloaded or used for training. Cosmopedia is Apache-2.0 but synthetic and must be quality sampled rather than treated as ground truth. The Stack v2 is gated, contains per-file licenses and provenance obligations, and requires ongoing removal updates; it should not be adopted wholesale. Stanford Alpaca is CC BY-NC 4.0 and research-only, so it must not become the foundation of a future commercially reusable model.

Primary source references:

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/v1.0.0/README.md)
- [FineMath dataset card](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- [Pinned Wikimedia Wikipedia dataset commit](https://huggingface.co/datasets/wikimedia/wikipedia/commit/e6057dc557255a03c9c3c47ceab0eb44353b1bc5)
- [Wikimedia licensing terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
- [Cosmopedia dataset card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- [The Stack v2 dataset card and terms](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [Stanford Alpaca repository and data license notice](https://github.com/tatsu-lab/stanford_alpaca)

## Training curriculum

### Stage 0 — mixture ablation

- Prepare small, provenance-complete candidate shards.
- Run 20M–50M new tokens per mixture candidate from the same step-200,000 base.
- Compare validation by domain and the existing 40-prompt suite.
- Select a mixture only if it improves at least two targeted categories without worsening repetition or broad-language validation materially.

### Stage 1 — broad capability-preserving continuation (about 700M tokens)

- Use the whole mixture with at least 45% general/educational replay.
- Evaluate every 25,000 optimizer steps.
- Preserve exact data-position and mixture-sampler state.

### Stage 2 — factual and educational emphasis (about 250M tokens)

- Increase Wikipedia-style, educational, evidence-linked QA, and regional material.
- Keep at least 30% broad replay to reduce domain forgetting.
- Evaluate factual accuracy on held-out entities and dates, not only training-style QA.

### Stage 3 — code, mathematics, and verified reasoning (about 250M tokens)

- Increase code/math/verified reasoning while retaining at least 40% broad/factual replay.
- Use held-out templates and numeric ranges to distinguish pattern memorization from transfer.
- Stop if general validation or language quality degrades consistently.

### Stage 4 — general instruction tuning

- Start again from the best new **base** milestone.
- Use boundary-preserved, EOS-supervised, assistant-only loss.
- Prefer a permissively licensed or internally generated instruction set; retain existing Alpaca only for research comparisons because of its non-commercial license.
- One controlled epoch, then evaluate.

### Stage 5 — verified reasoning, formatting, and uncertainty tuning

- Mix exact-format tasks, programmatically checked arithmetic, small code tasks, evidence-based factual QA, and refusal/clarification examples.
- Replay 25%–40% general instruction examples to avoid becoming a narrow solver.
- Keep hidden test templates separate from generated training templates.

### Stage 6 — limited conversational tuning

- Use a small, high-quality, low-truncation conversation set only after the capability gate passes.
- Replay at least 30% Stage 4–5 examples.
- Use a lower LR and stop immediately if reasoning, factual, programming, or repetition metrics regress.

## Architecture decisions

| Change | Decision and evidence | Files eventually affected | Impact | Checkpoint compatibility | Dataset compatibility |
| --- | --- | --- | --- | --- | --- |
| Context 256 → 512 | Defer during Option A. Useful for comprehension and longer SFT, but not the likely cause of simple fact/arithmetic failures. | `vasu/config.py`, data/training scripts, RoPE/cache/generation tests, fixed-record preparation. | Roughly higher attention/activation cost; may force batch 1. | Model weights can likely load because RoPE buffers are non-persistent, but optimizer/run compatibility requires validation. | Raw token streams remain compatible; fixed 257-token record datasets need repacking/review. |
| Grouped-query attention | Reject for now. It mainly reduces KV/inference cost; cache memory is not the quality bottleneck at context 256. | Config, attention projections, cache, model tests. | Lower cache size, uncertain quality at this scale. | Breaking tensor shapes. | Compatible. |
| SDPA/Flash backend work | Keep current PyTorch SDPA. Profile backend selection later; do not treat it as a quality change. | Attention/performance tests and environment configuration. | Potential speed/memory improvement only. | Compatible if math is unchanged. | Compatible. |
| Preallocated KV cache | Later inference optimization. Dynamic cache is correct but slightly slower; this does not affect training quality. | `vasu/cache.py`, generation, inference tests. | Lower allocation overhead if successful. | Fully compatible. | Fully compatible. |
| Width/depth/head changes | Reject for Option A. Consider only for an isolated 100M generation after data validation. | Config, model/attention/MLP tests, all architecture-specific runners. | More capacity and compute. | Breaking. | Token streams compatible. |
| SwiGLU hidden dimension | Keep 2,048; no evidence that a ratio change is the limiting factor. | Config/MLP/tests if changed. | Parameter/compute change. | Breaking. | Compatible. |
| RoPE scaling | Reject. No need at 256 and no evidence for extrapolation beyond trained context. | RoPE/config/tests. | Long-context experiment risk. | State dict may load, behavior changes. | Raw streams compatible; records may not be. |
| Dropout | Keep 0.1 until controlled ablation. | Config only plus tests/runners. | Regularization change. | Tensor-compatible, behavior-changing. | Compatible. |
| Bias usage | Keep disabled; adding bias has no evidence-backed benefit here. | Config and linear layers. | Small parameter change. | Breaking keys/shapes. | Compatible. |
| Normalization placement | Keep pre-norm RMSNorm; training is stable. | Blocks/model/tests. | High-risk optimization behavior change. | Breaking behavior/possibly keys. | Compatible. |
| Weight tying | Keep enabled; it saves about one vocabulary embedding matrix and is appropriate under 6 GB VRAM. | Model/embedding/load tests if changed. | Untying adds substantial parameters. | Breaking semantics and state loading. | Compatible. |

## Evaluation plan

### Existing suite

At every milestone run:

- greedy 40-prompt comparison;
- five-seed sampled comparison with seeds 42–46;
- automatic check and repetition summaries;
- manual review of all exact-answer failures and a fixed qualitative subset.

### Required expansion

| Category | Recommended evaluation |
| --- | --- |
| Arithmetic | Programmatically generated held-out addition, subtraction, multiplication, fractions, and sequences; exact-match accuracy by difficulty. |
| Common facts | Versioned QA from held-out Wikipedia/Wikidata-derived material with evidence and a date cutoff. |
| Reading comprehension | Short passages within 256 tokens with answerable and unanswerable questions. |
| Code generation | Small executable Python functions with unit tests; never score only by keywords. |
| Code explanation | Explain fixed snippets; manual rubric plus required-concept checks. |
| Formatting | JSON, exact lists, line counts, schemas, and constrained transformations. |
| Safety/uncertainty | Insufficient-information, medical/legal/financial uncertainty, and clarification requests. |
| Repetition | Mean, range, and worst-case repetition across fixed seeds and generation lengths. |
| Conversation | Multi-turn consistency and recovery from corrections; only after a longer context path exists. |
| Long-context recall | Deferred 512-token suite with key-value recall and passage QA; not used to judge the 256-token model. |

### Gates and stop conditions

- **Ablation gate:** no full continuation unless a 20M–50M-token mixture trial improves targeted held-out loss or task accuracy without a material broad-domain regression.
- **Milestone cadence:** every 25,000 optimizer steps during pretraining; after every SFT stage.
- **Quality stop:** stop/review after two consecutive milestones with no targeted improvement, or any persistent broad validation regression.
- **Repetition stop:** review if mean repetition rises by more than 0.05 absolute from the stage-start checkpoint or worst-case loops materially increase.
- **Numerical stop:** immediate safe stop for NaN/Inf loss, non-finite gradients, or checkpoint validation failure.
- **Operational stop:** retain the existing 88°C thermal stop and low-disk protection.
- **SFT promotion gate:** do not promote on average score alone; require no regression in reasoning/factual/programming categories and manual review.

## Compatibility strategy

### Compatible changes

- continuing the existing 58,337,792-parameter architecture;
- reusing `assets/tokenizer.json` after domain-fragmentation audit;
- reusing existing raw token streams as the replay portion of a new manifest;
- adding new token shards without rewriting existing binaries;
- using the existing evaluation suite and automatic metrics;
- using atomic checkpointing, corruption filtering, retention, disk, and thermal protections;
- adding mixture metadata and data-position state to a new specialized runner without changing old checkpoint payloads.

### Breaking changes

- changing vocabulary invalidates embeddings, LM head, tokenizer IDs, and every processed binary dataset;
- changing dimension, layer count, head projection layout, MLP width, bias keys, or weight tying invalidates current model/optimizer states;
- changing prompt templates without regenerating matching instruction records invalidates evaluation/training alignment;
- changing fixed record length requires new instruction binaries/masks and validation;
- a new scheduler cannot safely pretend that the old scheduler state has equivalent semantics;
- GQA changes K/V projections and cache structure and therefore breaks model-state compatibility.

## Operational estimates and risks

### Runtime and disk

- 1.2B additional tokens at 8,192 tokens/optimizer step: about 146,485 optimizer steps.
- Pure measured-step estimate: about 33 hours; realistic block/cooldown/validation range: **35–60 hours**.
- New `uint16` token data: about 2.4 GB, excluding raw downloads and preparation cache.
- Retained full checkpoints: about 0.7 GB each at current schema; operational retention plus milestones can add 2–4 GB.
- Recommended free space before data preparation: at least 20 GB, with streaming cleanup after artifact validation.

### Largest risks

1. A weak or contaminated data mixture consumes many hours without improving transfer.
2. Factual/math/code subsets may carry licensing, provenance, answer-quality, or secret/PII problems.
3. Narrow continuation may cause broad-language forgetting.
4. The 60M capacity ceiling may prevent robust generalization despite better data.
5. Laptop thermal cycling and long wall-clock time increase interruption risk.
6. Existing 256-token context limits document-level learning and later conversation quality.
7. Heuristic evaluation may reward keywords and formatting while missing semantic errors.

## Final recommendation

```text
Recommended path: Option A — capability-focused continuation of the existing VASU-60M base, gated by small mixture ablations.
Why: It targets the evidence-backed data and curriculum gaps, reuses the validated 60M system, and produces a reusable mixture before a much more expensive scale-up.
Rejected alternatives: Option B has unproven shape/context benefits and requires substantial retraining; Option C is the long-term candidate but is too costly before data validation; Option D is useful only for short mixture ablations and has a lower final capability ceiling.
Target parameter count: 58,337,792 for the next controlled generation; approximately 100M only after the data/curriculum gate passes.
Target context length: 256 for the continuation; separately evaluate 512 after capability repair.
Tokenizer: Reuse assets/tokenizer.json; do not retrain unless a measured domain-fragmentation audit fails.
Pretraining token target: 1.2B additional new token positions, approximately 2.84B total exposure.
Estimated training stages: mixture ablation; broad continuation; factual/educational emphasis; code/math/reasoning emphasis; general SFT; verified skill SFT; limited conversation SFT.
Estimated hardware feasibility: Proven for the 60M/256-token continuation at batch 2 and accumulation 16; approximately 35–60 wall-clock hours with thermal-safe blocks.
Checkpoint compatibility: Full for Option A model weights; preserve the step-200,000 milestone and isolate all new checkpoints. Scheduler semantics require an explicit new plan.
Dataset compatibility: Existing token streams and tokenizer remain usable; approximately 1.2B new provenance-recorded tokens and a new mixture manifest are required.
First implementation task: Design a non-executing mixture-manifest schema and deterministic domain sampler with exact resume state; do not change the model.
First data task: Build only a small licensed/provenance-audited pilot covering Wikipedia-style facts, FineMath, permissive code, and verified synthetic arithmetic, then validate tokenizer fragmentation and deduplication.
First training experiment: Two or more isolated 20M–50M-token continuation ablations from the same step-200,000 base, with matched optimizer settings and evaluation.
Go/no-go criteria: Proceed only if a candidate improves at least two targeted held-out categories, preserves broad validation/language quality, does not increase mean repetition by more than 0.05, remains thermally stable, and passes artifact/license/provenance checks.
```

## Wikimedia factual-pilot implementation status

The first data-preparation component of this plan is implemented but has not
produced training data. The Wikimedia preparer is pinned to the approved
source revision and refuses unbounded execution. Its default limits are one
Parquet shard, 10,000 inspected examples, 2,000 retained documents, 2,000,000
VASU tokens, and 1 GB downloaded. A smoke mode reduces those caps to 100 rows,
20 documents, and 20,000 tokens.

Configuration/approval validation, deterministic filtering, exact and bounded
near deduplication, prompt contamination checks, exact token measurement,
atomic resume/restart, provenance manifests, and output validation are covered
by tests. The first pinned-file smoke transfer stalled at 0 bytes and was
stopped, so there are no pilot statistics and no factual continuation is
authorized. Cross-FineWeb deduplication still requires a document-level index.
