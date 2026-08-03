# VASU-140M Source Admission v4 Quarantine-Binding Audit

Status: implementation complete; independent implementation review pending.

## Why this change is required

The accepted prompt-matrix remediation is conditional on the 2,659-document
FineWeb quarantine remaining mandatory during source admission and release
construction. Documentation alone cannot enforce that condition. Admission v4
therefore makes the accepted matrix decision and every mandatory exclusion
first-class, hash-bound package dependencies.

## Design

Admission v4 is additive. It reuses the complete v3 validator for source
registry, legal, acquisition, lineage, quality, ten-inventory, likelihood,
deduplication, decision, and authorization rules. It adds:

- an exact accepted prompt-matrix decision binding;
- a list of required exclusion artifacts with file and canonical identities;
- mandatory non-empty exclusions for FineWeb;
- repository-path and file-identity validation; and
- v4 package identity coverage over the added bindings.

Mutation, missing files, rejected matrix decisions, optional exclusions,
duplicate exclusion paths, unsafe paths, and a FineWeb package without its
quarantine all fail closed.

## Compatibility

Existing v3 admission packages and validators remain unchanged and readable.
No model architecture, tokenizer, source bytes, dataset, mask, checkpoint,
optimizer, scheduler, evaluation output, or training configuration changes.
No source is admitted by this implementation, and training remains
unauthorized.
