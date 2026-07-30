# VASU Roadmap

## VASU-140M instruction-seed release

- Completed: accepted record contract, release plan, fixture constructor,
  production-builder design, implementation, and post-commit identity review.
- Completed: independently accepted detached two-identity one-build
  authorization protocol.
- Completed: pre-commit protocol loader, exact runtime identity gate, exclusive
  lock, stale-lock policy, direct v2 receipt binding, and adversarial evidence.
- Completed: independent acceptance of the exact pre-commit implementation.
- Completed: commit the accepted package as `050d1fa` and freeze its clean
  post-commit qualification.
- Completed: independent acceptance of the post-commit identity.
- Next: commit the read-only runtime-eligibility tooling, emit the exact clean
  runtime report, and obtain a detached GPT-5.5 decision.
- Completed: detached runtime acceptance, one exact human-approved envelope,
  and one transactional production publication.
- Completed: independent validation of the published release and consumed
  receipt.
- Completed: checkpoint-selection audit found no compatible VASU-140M parent;
  VASU-60M artifacts are explicitly rejected as wrong-family checkpoints.
- Completed: GPT-5.5 independently accepted the fail-closed selection
  decision; no VASU-140M instruction-stage parent is selected.
- Completed: GPT-5.5 independently accepted the non-authorizing VASU-140M
  base-pretraining readiness protocol with CUDA, data, evaluation, resume,
  review, and authorization gates.
- Next: separately design and review the bounded CUDA/AMP operational
  qualification. Training remains blocked pending every readiness gate, a
  qualified 140M base checkpoint, and separate authorization.

## Capability-CPT v2 decision gate

Arithmetic diversity remediation and CUDA validation are complete. The next
decision is experimental authorization—not additional source replay or an
automatic launch.

## Capability continuation candidates

- **Completed:** canonical verified-arithmetic release and deterministic
  N-source schedules for factual A, balanced B, and non-arithmetic control C.
- **Completed:** bounded non-optimizing model smoke checks and schedule/config
  integrity validation.
- **Blocked:** authorization and scientific review of arithmetic replay.
- **Pending:** separately authorize at most one candidate, train in an
  isolated directory, and evaluate against the parent with versioned gates.

## FineWeb extension document recovery

- [x] Audit 379,247 retained source IDs and historical fingerprints.
- [x] Implement deterministic, atomic, revision-pinned bounded reacquisition.
- [x] Pass a 100-ID spread smoke with 100 exact historical hash matches.
- [x] Benchmark 1,000 IDs across serial, concurrency 2/4/8, and OR batches
  5/10/25; select batch 25, concurrency 1, and a conservative 2 RPS ceiling.
- [x] Run the resumable full reacquisition for all 379,247 retained extension
  source IDs with 379,247 exact historical text-hash matches.
- [x] Build and validate compatible extension coverage under
  `vasu_cross_source_nfc_casefold_ws_v1`.
- [x] Re-run the factual-pilot readiness gate after complete coverage.

The completed recovery and indexing work did not start model training or
authorize the default factual pilot.

## FineWeb cross-source deduplication gate

- Completed: versioned, resumable FineWeb document-index schema, exact lookup,
  word-5-gram MinHash/LSH candidate lookup, shared normalization, synthetic
  validation, and Wikimedia enforcement integration.
- Completed: real original-source build covering 1,000,000 inputs and 999,992
  unique indexed documents, with SQLite integrity and output-hash validation.
- Completed: production extension source-ID recovery, compatible extension
  document index, combined original-plus-extension coverage manifest, and a
  Wikimedia broad-review overlap check with zero candidates.
- Completed: refactored default factual-pilot preparation reached the token
  ceiling with 262 parent documents, 3,751 chunks, and 1,999,974 tokens; its
  v3 manifest and output validation passed.
- Not started: factual-pilot model training.

## Next-generation planning update

The completed VASU-60M cycle is now followed by a planning gate rather than immediate scale-up.

Planned sequence (not implemented):

1. Specify a provenance-aware multi-domain mixture manifest and deterministic resumable sampler.
2. Prepare only small licensed pilot shards for factual, math, permissive-code, and verified synthetic reasoning data.
3. Run matched 20M–50M-token continuation ablations from `fineweb_step_200000.pt`.
4. If the gate passes, continue the existing VASU-60M base for approximately 1.2B new tokens using broad replay and staged domain emphasis.
5. Rebuild instruction tuning as general instruction → verified reasoning/formatting → limited conversation, evaluating after every stage.
6. Consider an approximately 100M model only after the mixture demonstrates measurable improvement without unacceptable forgetting or repetition.

The detailed option comparison, compatibility analysis, stop conditions, and go/no-go criteria are in `docs/VASU_NEXT_PLAN.md`. None of these milestones is marked complete.

Source-governance progress:

- completed: pinned primary-source review and registry approval for the
  `wikimedia/wikipedia` `20231101.en` factual source at commit
  `e6057dc557255a03c9c3c47ceab0eb44353b1bc5`;
- completed: bounded, resumable, one-shard Wikimedia pilot preparation with
  provenance, deterministic filtering, contamination checks, exact/near
  deduplication, exact tokenizer measurement, and atomic progress;
- completed: corrected v2 smoke preparation with exact-token chunking, safe
  inline-markup spacing, conservative Unicode handling, 20 distinct chunks,
  17,237 tokens, and a 1,022-token observed maximum;
- completed: deterministic broad review across the shard, producing 50 chunks
  from 14 parent articles and 26,435 tokens with a 938-token maximum;
- completed: split the ambiguous accepted-document counter into independent
  parent-document and chunk limits, then regenerate the default pilot to
  1,999,974 tokens with zero replacement characters and one rejected FineWeb
  near overlap;
- completed: add deterministic SHA-bound manual-review sampling with bounded
  previews; the current 79-chunk sample covers random, size-extreme,
  near-maximum, evenly spaced, distinct-parent, warning, reference, and
  suspicious-metadata groups;
- completed: treat the first 79-chunk review as failed evidence, then correct
  reference appendix leakage, tiny fragments, malformed boundaries,
  list-dominated content, and missing source values before regenerating the
  pilot under the v4 schema;
- completed: validate the corrected 1,999,700-token artifact and regenerate a
  61-chunk pending review with zero reference, below-minimum, or automatic
  precheck failures;
- next: manually classify all 61 corrected samples and decide whether the
  pilot can support a controlled factual continuation experiment;
- unchanged: factual-pilot model training has not started;
- available: cross-FineWeb document deduplication through the combined
  original-plus-extension coverage manifest and compatible indexes;
- blocked: capability-pilot mathematics, code, and reasoning approvals.

## Historical roadmap snapshot

The sections below preserve an earlier step-54,060-to-100,000 roadmap snapshot. They are retained as project history and are superseded for current planning by the completed step-200,000 cycle and `docs/VASU_NEXT_PLAN.md`.

## Completed

- VASU-31M architecture and training pipeline;
- TinyStories and FineWeb pretraining experiments;
- VASU-31M Alpaca and UltraChat instruction-tuning experiments;
- masked-response training experiments;
- VASU-31M fixed-prompt comparison and 2.225 / 5 manual baseline;
- VASU-60M architecture planning and opt-in configuration;
- VASU-60M CPU and CUDA smoke tests;
- VASU-60M tiny real-data and save/resume tests;
- resumable, thermal-safe FineWeb block pretraining;
- atomic checkpoint saving and corrupt-checkpoint filtering;
- bounded checkpoint retention and low-disk protection;
- base-generation milestone evaluation through step 54,060.

## In progress

- Continue VASU-60M FineWeb base pretraining from the preserved step-54,060 milestone toward global step 100,000.

VASU-60M instruction tuning has not started.

## Next

1. Preserve and evaluate the step-100,000 VASU-60M checkpoint.
2. Compare grammar, repetition, topic retention, coherence, and factuality against earlier base milestones.
3. Decide whether more base pretraining is required.
4. If the base-model gate passes, run controlled VASU-60M Alpaca tuning.
5. Evaluate before any controlled UltraChat tuning.
6. Compare the resulting VASU-60M instruction model against the VASU-31M baseline of 2.225 / 5.
7. Expand evaluation coverage and investigate generation repetition controls.
8. Complete the blocked VASU-60M capability-CPT readiness gate: tokenize the
   verified arithmetic corpus, add a validated three-source fixed-record
   mixture builder, then run matched 20M-token candidates only after explicit
   authorization.

## Later

- KV-cache optimization;
- longer context support;
- improved learning-rate scheduling experiments;
- complete data-sampler resume state;
- broader language-model and factuality evaluation;
- improved instruction and safety data;
- quantization and inference optimization;
- conversational memory, tools, and possible Jarvis-style assistant layers only after the language model is stable.

## Guiding principle

Promote a checkpoint only through reproducible evaluation. The current preferred assistant remains the evaluated VASU-60M Alpaca v3 checkpoint; experimental branches must exceed it through matched evaluation before promotion.


KV-cache v1:
Correct dynamic cache implementation
Fully parity-tested
Lower peak memory
Not enabled because performance regressed slightly

KV-cache v2:
Preallocated storage
Future optimization task
Must preserve all existing parity tests
