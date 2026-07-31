# VASU-140M Base-Model Evaluation Suite v2 Design

Status: **design-only; no suite is frozen and no model evaluation is run.**

## Purpose

Define the evaluation contract that must be frozen before the first
`vasu_140m_v1` optimizer update. The suite evaluates a decoder-only base model
through raw continuation and likelihood interfaces. It must not use assistant
roles, instruction templates, or instruction-tuned checkpoints as direct
controls.

## Frozen identity boundary

Every suite manifest must bind its schema/version, repository commit, model
family/configuration, tokenizer SHA-256, prompt and dataset files, scorer code,
generation configuration, seeds, and canonical manifest SHA-256. Result
manifests additionally bind the checkpoint file SHA-256, environment, device,
precision, command, timestamps, and every output shard.

The initial VASU-140M checkpoint has no trained parent. Comparisons may include
random initialization and VASU-60M base references only as labeled calibration
points; they are not matched scientific parents and cannot establish promotion.

## Required dimensions

1. **Likelihood:** full-loss development and held-out loss/perplexity on the
   future base-data release, reported by source and overall with token counts.
2. **Factual retention:** direct likelihood multiple choice and normalized
   cloze using frozen, source-attributed prompts. Generation is secondary.
3. **Arithmetic:** verified development and held-out items with exact parsing,
   operation-family breakdowns, and bootstrap confidence intervals.
4. **Repetition and degeneration:** raw continuation prompts measuring repeated
   n-grams, unique-token ratios, looping, empty output, EOS behavior, and length.
5. **Robustness:** meaning-preserving prompt perturbations, whitespace and case
   variants, continuation-prefix variants, and paired consistency statistics.
6. **Manual review:** a deterministic stratified sample covering coherence,
   factual support, contradiction, degeneration, and unsafe certainty. Human
   judgments remain separate from automatic metrics.

## Split and contamination policy

Development prompts support diagnostics. Held-out prompts remain sealed until
the experiment plan's declared decision point. No prompt, accepted answer,
eight-word fragment, semantic duplicate, or parent document may enter training
data. Every prompt inventory is hash-bound and included in source-admission
contamination scanning before data splitting.

## Runtime protocol

Greedy generation is the primary deterministic generation mode. A separately
reported sampled mode uses a frozen seed list. Maximum new tokens, EOS/PAD IDs,
temperature, top-k/top-p, repetition penalty, batching, precision, and KV-cache
behavior are fixed per task family. Interrupted evaluation resumes only when
suite, checkpoint, tokenizer, scorer, and runtime identities match exactly.

Likelihood tasks must score choices or target continuations directly and must
not infer likelihood from generated text. Generation parsers retain raw text,
parsed value, parse status, truncation status, and token counts.

## Statistical reporting

Report numerator/denominator and point estimates for every metric. Paired tasks
use paired differences. Accuracy and rates include deterministic bootstrap
95% confidence intervals with frozen seeds and resample counts. Source,
operation, prompt-family, and length strata are reported without collapsing
them into a single capability score. Zero-versus-zero results are explicitly
described as no measurable difference under that test.

## Fail-closed conditions

Evaluation is invalid if any required inventory, hash, scorer, split, seed,
raw response, or checkpoint identity is missing; if held-out prompts were
opened early; if an instruction prompt format is used; if results are silently
overwritten; or if metric dimensions are combined into one promotion score.

## Implementation sequence

1. Define a strict suite/result schema and compatibility validator.
2. Build small synthetic fixtures for each task type and adversarial tests.
3. Freeze versioned development and held-out inventories with provenance.
4. Produce a read-only qualification report without a model checkpoint.
5. Obtain independent implementation and frozen-identity reviews.

## Compatibility and non-authorization

This design changes no model, tokenizer, dataset, mask, checkpoint, or existing
evaluation result. Existing evaluation utilities may be adapted additively.
It does not authorize prompt generation from protected held-out sources, model
execution, data construction, configuration, checkpoint creation, or training.
