# VASU-140M Private Curator Intake

Date: 2026-08-03

Status: validator implemented and fixture-qualified; production held-out inputs
remain absent.

## Purpose

This package validates externally curated held-out inputs without copying or
printing their plaintext. It reads only the five explicitly named JSONL files
from a directory outside the repository and emits counts, file hashes, byte
counts, and aggregate identity counts. It never reads the Age private key,
decrypts ciphertext, creates an evaluation authorization, or starts training.

## Required private files

The default private directory is `D:\VASU_PRIVATE_HELDOUT`. Final files are:

- `factuality-heldout.jsonl`: 200 records;
- `arithmetic-heldout.jsonl`: 1,000 records;
- `repetition-heldout.jsonl`: 120 records;
- `robustness-heldout.jsonl`: 120 records;
- `manual-review-heldout.jsonl`: 60 records.

The existing `.schema-sample.jsonl` files are formatting examples only and are
rejected as final evidence. Every final record must use schema
`vasu_140m_sole_curator_draft_v1`, status
`curator_approved_for_sealing`, its exact dimension, and globally unique
`item_id`, `semantic_family_id`, and `parent_document_id` values.

Every record requires this provenance object:

- `source_name`, `source_url`, `license_name`, `license_url`;
- `source_revision`, `citation`, and `authored_by`;
- both URLs must use HTTPS.

Dimension-specific fields follow the schema examples. Arithmetic is restricted
to the auditable template `Calculate exactly: INTEGER OP INTEGER = ?`, integer
answers, and exact division. Manual-review rubrics must be exactly `coherence`,
`factual_support`, and `degeneration` in that order.

## Validation

Run from `D:\VASU`:

```powershell
python scripts\validate_vasu_140m_private_curator_intake.py
```

Success produces JSON containing only aggregate counts and hashes. The
validator rejects wrong counts, schema-sample status, malformed provenance,
duplicate identities or prompts, incorrect arithmetic, linked inputs, private
directories inside the repository, and exact normalized overlap with the
public assistant-authored development suite.

The fixture qualification is reproducible with:

```powershell
python scripts\smoke_vasu_140m_private_curator_intake.py
```

## Scientific boundary

Passing intake proves structural validity and exact development-prompt
non-overlap only. It does not prove independent authorship, factual quality,
semantic non-overlap, ciphertext sealing, source admission, suite freezing, or
authorization. Those require separate evidence and decisions. No private
prompt, key, or decrypted content belongs in Git.

## Compatibility

This package is additive and changes no model, tokenizer, dataset, mask,
checkpoint, optimizer/scheduler state, training config, evaluation result, or
exact-resume behavior.
