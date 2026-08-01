# VASU-140M Evaluation-v2 Production Inventory Construction Design

Status: **design-only; no production inventory, prompt, key, or evaluation run
exists.**

## Purpose

Define the only acceptable path from the committed fixture-qualified inventory
contracts to real VASU-140M base-model evaluation inventories. This design
does not convert synthetic records into production evidence and does not permit
the repository to create or open held-out plaintext.

## Required inputs

Production construction must receive, as externally supplied immutable inputs:

1. ten independently authored prompt inventories: development and held-out for
   factuality, arithmetic, repetition, robustness, and manual review;
2. public provenance and contamination commitments for every item;
3. a recipient public-key fingerprint plus Age X25519 ciphertext for each
   held-out payload, with no private key in the repository or build runtime;
4. a source-admission decision binding all ten public inventory manifests
   before any base-data source can be approved; and
5. an accepted construction plan identity, scorer identities, tokenizer
   identity, and a clean reviewed repository commit.

Likelihood is different: its development and held-out records are derived only
after an admitted source acquisition, at whole-document boundaries, before
513-token packing, and must be bound to the eventual base-data release.

## Construction protocol

The future implementation must:

- reject fixture records, plaintext held-out payload paths, missing ciphertext,
  duplicated item/family/parent-document identities, and any record whose
  provenance or contamination commitment does not validate;
- validate development payloads through the existing inventory contract and
  store production manifests atomically with no overwrite;
- validate held-out ciphertext as opaque bytes only, retain public manifest,
  provenance, and contamination indexes, and leave `opening_authorized=false`;
- bind every manifest to the construction-plan SHA-256, source-admission
  decision SHA-256, repository commit, tokenizer, scorer, and canonical
  manifest identity;
- write a detached construction receipt, never an authorization record; and
- fail closed if a held-out key, plaintext, model, checkpoint, optimizer,
  schedule, source byte, or training configuration is present in the process.

## Acceptance criteria

Before production construction can be reviewed, tests must prove atomicity,
overwrite rejection, crash preservation, Windows link/junction rejection,
commit/plan/decision substitution rejection, ciphertext-only held-out handling,
and no plaintext recovery. A clean-checkout qualification must reproduce every
public manifest and receipt identity.

## Compatibility and non-authorization

The protocol is additive. It changes no model, tokenizer, dataset, mask,
checkpoint, optimizer, scheduler, or current evaluation output. It does not
authorize prompt authoring, key creation, source acquisition, data release,
model execution, experiment planning, authorization, or training.
