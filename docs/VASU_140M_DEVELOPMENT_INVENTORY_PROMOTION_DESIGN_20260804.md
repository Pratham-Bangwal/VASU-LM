# VASU-140M Development Inventory Promotion Design

Status: design frozen; independent design review pending; construction is not
authorized.

## Problem

The independently accepted prompt matrix establishes that the five public
development inventories are structurally and scientifically suitable for
development-only use. Their current manifests remain fixture evidence with
`fixture_only=true`. Source-admission v3/v4 correctly refuses to approve a
package that binds fixture manifests. A production candidate therefore needs
new immutable paths and identities; changing the existing fixture flag in
place is forbidden.

## Selected approach

Create a one-shot, non-overwriting promotion builder that consumes the exact
five accepted fixture bundles and emits five new development candidate bundles
under:

`evaluation/candidates/vasu_140m_base_v2_development_v1/<dimension>/`

For each dimension, the builder will:

1. validate the source manifest and all bound files with the existing inventory
   validator;
2. require `split=development`, `fixture_only=true`, and the exact accepted
   source inventory identity;
3. copy payload, provenance, and contamination bytes without semantic edits;
4. bind their new candidate paths and byte hashes;
5. assign a new production-candidate suite ID and inventory ID;
6. bind the clean runtime commit;
7. set only `fixture_only=false` while keeping
   `production_suite_frozen=false`, `evaluation_run_authorized=false`, and
   `training_authorized=false`;
8. recompute and validate each inventory identity; and
9. publish all five atomically or publish none.

The builder must never read or decrypt held-out payloads. It operates only on
the public development inventories.

## Immutable source bindings

The promotion package must bind:

- the accepted prompt-matrix remediation decision;
- the source-admission v4 canonical-identity acceptance;
- all five source inventory identities;
- the tokenizer identity;
- the scorer implementation identity; and
- the runtime repository commit.

Any missing, renamed, substituted, or mutated dependency fails closed.

## Filesystem and failure policy

- Existing candidate output is a hard failure.
- Staging is a sibling directory on the destination filesystem.
- Symlink and Windows junction traversal is rejected.
- Files are written with explicit LF where textual.
- Pre-publication failure removes staging.
- Failure after final rename preserves visible incomplete evidence and does not
  create a completion receipt.
- A detached receipt is written only after all five promoted inventories pass
  repository-file validation.

## Required qualification

Tests must cover:

- exact five-dimension construction and counts;
- byte preservation for payload/provenance/contamination files;
- new manifest and inventory identities;
- source mutation and dependency mutation;
- wrong split and non-fixture source rejection;
- overwrite, partial-set, symlink, and junction rejection;
- injected write, rename, and post-rename validation failures;
- deterministic logical output across temporary roots; and
- all authorization flags remaining false.

## Compatibility

This design is additive. Existing fixture inventories remain unchanged and
continue to validate. Model architecture, tokenizer, datasets, masks,
checkpoints, optimizer/scheduler state, exact resume, and evaluation results
remain compatible.

## Non-authorization

Design acceptance would authorize only later implementation review. It does
not authorize promotion execution, production-suite freezing, source
admission, likelihood construction, evaluation execution, data release,
checkpoint access, configuration, scheduling, optimizer creation, or training.
