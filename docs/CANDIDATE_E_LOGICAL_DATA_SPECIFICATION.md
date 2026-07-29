# Candidate E Logical Data Specification

Status: proposed for independent review; no Candidate E records are generated
by this document.

## Immutable identity

- Dataset family: `candidate_e_verified_arithmetic_v1`
- Stable IDs: `candidate_e_v1:{split}:{index:07d}`
- Generator seed: `20260729`
- Splits: 32,000 train, 1,000 development, 1,000 evaluation examples
- Categories and verification schema: arithmetic-v2 categories and exact
  programmatic verification metadata, preserved unchanged.

## New split isolation

Candidate E must not reuse arithmetic-v2 held-out identities, operand ranges,
or template families. Its operands are reserved as follows:

| Split | Primary operand range | Template family |
| --- | --- | --- |
| train | 10,000–19,999 | `e_train_explain_v1`, `e_train_compute_v1`, `e_train_exact_v1` |
| development | 20,000–24,999 | `e_dev_derive_v1` |
| evaluation | 25,000–29,999 | `e_eval_concise_v1` |

Tier-specific values may be smaller only when required by a categorical
definition (for example fractions or percentages), but every rendered prompt,
stable ID, and normalized-expression hash must remain split-disjoint and must
be checked against the complete arithmetic-v2 development/evaluation releases.

## Paired serialization

Both arms use the same logical records and the shared prefix:

```
Question: {prompt}
Response:
```

Control response: `Answer: {answer}`.

Treatment response: bounded, deterministic verified symbolic states followed
by the identical `Answer: {answer}`. Prompt tokens are unsupervised; response
and terminal EOS tokens are supervised. No model-generated reasoning text is
permitted.

## Acceptance requirements

Before the fixture-only release validation or any production release, an
implementation must prove deterministic regeneration, exact answer
verification, prompt/template/semantic split isolation, tokenizer-boundary
stability, 257-token capacity, target-shift mask alignment, and release hashes.
The matched-budget/evaluation protocol remains unchanged.

## Review decision

An independent reviewer must explicitly accept or reject this specification.
Acceptance permits only implementation review and fixture validation; it does
not authorize a production dataset, schedule, configuration, checkpoint, or
training run.
