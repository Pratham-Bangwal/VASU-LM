# VASU-140M Source Admission v4 Candidate Construction Audit

Status: two review candidates constructed; neither source is admitted.

Review-ready packages:

- FineWeb: `configs/data/admissions/candidates/fineweb_edu_extension_2025_26.v4.candidate.json`
  with package identity
  `0b178c31742249c926eb9dd0b19cedadffe51c351d936a51e6821a204fc0b910`.
- Wikimedia: `configs/data/admissions/candidates/wikipedia_en_20231101.v4.candidate.json`
  with package identity
  `06bfd284b4cc3e64554ec42925e1c3b272f6051ffcf90591934b0cc8c2968446`.

Both packages bind all ten accepted prompt inventories, the accepted prompt
matrix decision, existing v3 legal/acquisition/lineage/deduplication policy,
and false training authorization. FineWeb additionally binds the reviewed
2,659-document quarantine as mandatory by file and embedded canonical identity.
Wikimedia has no quarantine because the independent scan found no candidates.

Both decisions remain `pending`. Likelihood remains deferred until independent
source admission establishes final document-level exclusions. No source bytes,
dataset, tokenizer, mask, checkpoint, evaluation result, optimizer, schedule,
configuration, or training state changed.
