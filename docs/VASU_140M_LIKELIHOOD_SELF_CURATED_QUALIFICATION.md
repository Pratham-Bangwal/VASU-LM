# VASU-140M self-curated likelihood qualification

Status: **implementation prepared; construction not yet executed.**

The accepted production inventory plan requires an independent held-out curator.
The project currently has no independent curator, so assistant-constructed
likelihood evidence must not be represented as the frozen production suite.

This additive qualification path deterministically reserves 512 development and
512 held-out whole parent documents from each admitted candidate source. It uses
SHA-256 ranking with seed 140021, creates bounded direct-likelihood context and
target records under the frozen tokenizer, and seals held-out payloads with the
owner-supplied Age X25519 public recipient. Plaintext held-out payloads exist
only in a system temporary directory during encryption and are not persisted in
the repository.

Every output remains `fixture_only=true`, `independently_curated=false`,
`production_suite_frozen=false`, `evaluation_run_authorized=false`, and
`training_authorized=false`. The package changes no model, tokenizer, training
dataset, mask, checkpoint, optimizer, schedule, or exact-resume behavior.
