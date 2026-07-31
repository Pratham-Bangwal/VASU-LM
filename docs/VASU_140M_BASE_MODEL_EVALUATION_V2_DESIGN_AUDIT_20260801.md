# VASU-140M Base-Model Evaluation v2 Design Audit — 2026-08-01

The readiness protocol requires likelihood, factuality, arithmetic,
repetition/degeneration, robustness, and manual review before training. The
existing `vasu_capability_v1` suite is useful infrastructure evidence but has
only eight instruction-style tasks and promotion gates for continued
pretraining/instruction/conversation. It is not a sufficient VASU-140M base
evaluation contract.

The v2 design preserves reusable checkpoint loading, deterministic generation,
atomic reporting, metrics, and resume concepts while separating base-model raw
continuation from assistant evaluation. It requires paired uncertainty,
development/held-out isolation, full provenance and hashes, and independent
metric families. No benchmark, result, or checkpoint was created or modified.

Validation is limited to current evaluation-framework regression tests plus
documentation integrity. Implementation and frozen prompt inventories remain
separate review-gated milestones.
