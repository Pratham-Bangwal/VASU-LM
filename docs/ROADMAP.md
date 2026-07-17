# VASU Roadmap

## FineWeb cross-source deduplication gate

- Completed: versioned, resumable FineWeb document-index schema, exact lookup,
  word-5-gram MinHash/LSH candidate lookup, shared normalization, synthetic
  validation, and Wikimedia enforcement integration.
- Completed: real original-source build covering 1,000,000 inputs and 999,992
  unique indexed documents, with SQLite integrity and output-hash validation.
- Blocked: production coverage until a full compatible index is built and the
  extension's document-level text is reacquired from an authoritative source.
- Not started: default factual-pilot preparation or training.

## Next-generation planning update

The completed VASU-60M cycle is now followed by a planning gate rather than immediate scale-up.

Planned sequence (not implemented):

1. Specify a provenance-aware multi-domain mixture manifest and deterministic resumeable sampler.
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
- next: review broader bounded samples before deciding whether to authorize the
  default 2M-token pilot;
- unchanged: the default pilot and factual-pilot training have not started;
- blocked: cross-FineWeb document deduplication until a versioned normalized
  document-hash and compatible word-shingle signature index exists;
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

Promote a checkpoint only through reproducible evaluation. Keep the VASU-31M assistant checkpoint as the stable fallback until a VASU-60M instruction checkpoint measurably exceeds it.


KV-cache v1:
Correct dynamic cache implementation
Fully parity-tested
Lower peak memory
Not enabled because performance regressed slightly

KV-cache v2:
Preallocated storage
Future optimization task
Must preserve all existing parity tests
