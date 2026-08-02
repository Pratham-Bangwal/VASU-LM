# VASU-140M likelihood-source preparation profile

Status: **implementation-qualified; source construction not yet executed.**

The historical Wikimedia pilot is intentionally capped at 2 million tokens and
previously retained only 402 parent documents. That is insufficient for the
frozen likelihood requirement of 512 development and 512 held-out documents.

The additive `vasu_140m_likelihood_source_v1` preparation profile preserves the
same pinned source, one-shard download ceiling, filters, contamination checks,
FineWeb cross-deduplication, atomic progress, and resume behavior while setting
separate hard ceilings of 3,000 parents, 20,000 chunks, and 15 million tokens.
The historical `pilot_v1` limits are unchanged and remain the default.

Outputs use a new isolated path and cannot overwrite the approved historical
Wikimedia pilot or quarantine release. This profile changes no model,
tokenizer, existing dataset, mask, checkpoint, optimizer state, schedule, or
exact-resume artifact. It authorizes no training.
