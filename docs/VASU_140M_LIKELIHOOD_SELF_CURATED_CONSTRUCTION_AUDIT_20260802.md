# VASU-140M self-curated likelihood construction audit

Date: 2026-08-02

## Outcome

The qualification-only likelihood package completed successfully at repository
commit `28b20aa9fe4f653b40471fb68d9001c64195d6a7`. It is not independent evidence
and does not freeze or authorize the production evaluation suite.

The Wikimedia source preparation completed at the 20,000-chunk ceiling with
2,878 distinct parents and 10,950,655 VASU tokens. Its output SHA-256 is
`bc2a014fadcfca02cdebce55bc47c7780838a2ad5278f0cb2a63dc5679c7b7a4` and
its manifest SHA-256 is
`e4516d2132a12e38ae9523c5cfe8a38bddf44a945c4bdfa86e1cb9a03f521286`.
The source validator passed. The source recorded zero exact duplicates and one
rejected near duplicate.

FineWeb was reused from the previously recovered immutable artifact with
SHA-256 `c90f9e21d9b73324b9165cf1fb7ffbc274fbba5ccba5ac22b7cbe48abb6d7f1e`;
no second acquisition was performed.

## Constructed inventories

Each source has 512 development and 512 held-out items. Parent-document
intersection between development and held-out is zero for both sources.

- FineWeb development inventory:
  `3964867cd2a6c11db72b57e824e503e4841e0eb5eb27a0e66d2f65818c5c24e8`
- FineWeb held-out inventory:
  `9e0c21886be39aa8b7f3d7cc67512fb04d713b7900d72885d33311eca20bfd63`
- Wikimedia development inventory:
  `ec8f6c7efcfd2ced161293ed1c87a2081f66c5a85209f2d066a7623f0b7bbd04`
- Wikimedia held-out inventory:
  `2497d6a429ed667b62a4bbf2660f47206ef45d0583960acb92b012acce59efdb`

The qualification identity is
`69f742cd0a4cf02c8de36d37cb469bc1900537a180bcf2ca4f683b557496c0b2`.
The Age-recipient fingerprint is
`3413c58d2dae91a9c253642aba8583923af3fcd5d02893ad940ca6f21b5bd2da`.
The private key was not read or copied.

## Validation

- The Wikimedia source output validator passed with 2,878 parents, 20,000
  chunks, 10,950,655 tokens, and output hash `bc2a014f...b7a4`.
- All four inventory manifests and every bound payload, provenance, contamination,
  and scorer identity passed `validate_inventory_manifest_files`.
- Both development payloads contain exactly 512 validated plaintext records.
- Both held-out payloads contain the exact Age v1 header and no plaintext
  held-out payload is stored in the repository.
- Public provenance and contamination indexes each contain exactly 512 records
  per inventory and match their manifest commitments.
- No `vasu-heldout-*` system-temporary directory remained after encryption.
- Model invocation, checkpoint access, evaluation authorization, optimizer
  updates, and training were all false.

## Compatibility and limits

This work changes no model architecture, tokenizer bytes, training dataset,
mask, checkpoint, optimizer or scheduler state, training configuration, or
exact-resume behavior. It does not satisfy the independent-curator requirement,
does not approve the two pending source-admission packages for base training,
and does not authorize model evaluation or training.
