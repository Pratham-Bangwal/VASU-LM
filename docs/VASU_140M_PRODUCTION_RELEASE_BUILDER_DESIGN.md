# VASU-140M Production Release Builder Design

Status: independently accepted for separately reviewed implementation;
construction unauthorized.

Date: 2026-07-30

## Problem

The accepted fixture constructor proves packing and transactional publication
on at most 30 caller-supplied examples. It deliberately does not prove source
discovery, replay of the frozen 898/48/50 assignment, full-corpus compilation,
or publication of the planned 996-example release.

A production implementation must close that gap without turning a reviewed
plan into implicit permission to write protected artifacts.

## Alternatives

### Extend the fixture function

Removing the fixture cap and production-path rejection would minimize code, but
would mix test evidence with privileged behavior and make accidental
construction too easy. Rejected.

### One command that validates and publishes

A single command could revalidate sources and immediately publish. This leaves
too little separation between inspection and protected writes and weakens
independent review. Rejected.

### Two-phase qualification and authorized publication

The recommended design separates a read-only, deterministic qualification
bundle from a narrowly authorized publication step. The publication step
accepts only the exact reviewed qualification identity and exact output paths,
stages all artifacts, validates them, and atomically publishes once.

This approach adds one review boundary but provides the strongest
reproducibility, least privilege, and failure isolation.

## Frozen architecture

```text
accepted plan + accepted decisions + source files
                       |
                       v
             read-only qualification
                       |
                       v
       immutable construction qualification bundle
                       |
              independent GPT-5.5 review
                       |
          explicit release-build authorization
                       |
                       v
       isolated staging + complete validation
                       |
                       v
          atomic non-overwriting publication
```

Qualification and publication must be separate public entry points. Importing
the module, validating the plan, or producing qualification evidence must
never write token, mask, logical-manifest, schedule, or training artifacts.

## Phase 1: read-only qualification

The qualifier must:

1. validate the exact accepted plan and its canonical SHA-256;
2. validate both independent decision file identities;
3. revalidate all source, review, release-manifest, tokenizer, model-family,
   record-contract, contamination-inventory, and assignment identities;
4. compile all eligible examples in memory or disposable scratch storage;
5. reproduce exactly 898 train, 48 development, and 50 evaluation examples;
6. reject truncation, tokenizer boundary merges, split leakage, duplicate
   logical IDs, semantic collisions, and assignment drift;
7. calculate expected logical, token, mask, and manifest hashes without writing
   planned outputs;
8. emit one immutable qualification report with
   `production_release_created=false` and `training_authorized=false`.

Scratch artifacts must be outside the planned production locations and removed
after qualification. The qualifier must fail if any planned output already
exists.

## Phase 2: authorized publication

Publication requires a versioned release-build authorization record bound to:

- exact repository commit;
- plan ID and canonical SHA-256;
- plan and fixture-review decision file SHA-256 values;
- qualification report path and SHA-256;
- source, review, and source-manifest hashes;
- tokenizer, family configuration, and record-contract hashes;
- exact 898/48/50 assignment SHA-256;
- expected logical, token, mask, and release-manifest hashes;
- exact output paths;
- `overwrite_allowed=false`;
- named human approver, approval date, and one-build scope.

The publisher must reject missing, malformed, reused, expired, or mismatched
authorization. A Boolean config flag is insufficient.

Publication must:

1. verify the worktree and reviewed commit identity;
2. fail if any destination or sibling staging path exists;
3. write every artifact beneath one unique sibling staging directory;
4. flush and close every file before validation;
5. validate sizes, dtypes, record widths, masks, hashes, counts, and lineage;
6. write the logical manifest last inside staging;
7. atomically rename the complete directory into the release location;
8. atomically publish the external manifest only after the directory exists;
9. write a consumed-authorization receipt without modifying the authorization;
10. never create a schedule, training config, checkpoint, or optimizer state.

The external manifest creates a small two-object publication boundary. Recovery
must therefore be explicit: if release-directory publication succeeds but
external-manifest publication fails, the release is quarantined as incomplete
and must not be silently deleted, overwritten, or treated as usable.

## Output contract

The accepted plan defines the only allowed production locations:

- `data/processed/vasu_140m/instruction_seed_v1/`
- `data/manifests/vasu_140m/instruction_seed_v1.json`

The release directory contains only:

- `train.tokens.bin` and `train.mask.bin`;
- `development.tokens.bin` and `development.mask.bin`;
- `evaluation.tokens.bin` and `evaluation.mask.bin`;
- a logical-record lineage file;
- an internal publication manifest.

No path may be supplied through an unrestricted command-line override.

## Threat model and required failures

The implementation must fail closed for:

- changed plan, decision, source, review, tokenizer, family, or contract hash;
- different assignment membership or ordering;
- existing or partially existing production output;
- symlink, junction, or path traversal at staging or destination;
- filesystem boundary that prevents atomic rename;
- insufficient disk space before construction;
- non-`uint16` tokens, non-`uint8` masks, wrong record width, or length mismatch;
- prompt, PAD, or cross-example supervision;
- response or EOS under-supervision;
- missing, duplicated, reordered, or cross-split example IDs;
- unbound or extra files;
- artifact mutation after staging validation;
- injected failures before and after directory publication;
- authorization reuse.

## Test and evidence requirements

Implementation review requires:

- unit tests for every failure family above;
- deterministic full-source dry runs in two independent temporary roots;
- byte-identical logical, token, mask, and manifest hashes;
- exact 898/48/50 membership replay;
- decoded boundary sampling across every capability and split;
- complete mask audits, not sampling alone;
- simulated disk exhaustion and injected publication failures;
- Windows-specific atomicity, junction, and open-handle tests;
- repository-wide tests and Ruff;
- a frozen qualification report and completion audit;
- independent GPT-5.5 acceptance before any release-build authorization.

## Compatibility

The design changes no model architecture, checkpoint schema, tokenizer,
existing dataset, mask, schedule, trainer, or resume behavior. A later release
would be a new versioned dataset artifact compatible only with the accepted
VASU-140M 513-token response-masked instruction contract.

## Non-authorization

GPT-5.5 independently accepted this design on 2026-07-30 for a separately
reviewed implementation proposal. The decision authorizes neither production
release construction nor training. Production construction, scheduling,
training configuration, checkpoint selection, and training remain prohibited.
