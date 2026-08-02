# VASU-140M assistant-authored internal evaluation suite v1

Status: **development-only fixture; not independently curated; not an
authorization gate.**

## Purpose and provenance

The project owner explicitly authorized the VASU development assistant to
create a practical internal diagnostic suite because an independent held-out
curator was unavailable. This supports engineering progress without claiming
an evidence boundary that does not exist.

The suite generates factuality (200), arithmetic (1,000), repetition (120),
robustness (120), and manual-review (60) development records. Every provenance
record has `human_authored=false`; every manifest has `fixture_only=true`; and
the suite report states that it is non-independent, has no held-out content,
and is ineligible for training data.

## Permitted use

After a separately authorized model-evaluation run, the suite may support
internal regression diagnostics, deterministic arithmetic parsing checks, and
generation-degeneration comparisons. It must not be added to a training
mixture or used as independent held-out, external benchmark, source-admission,
checkpoint-promotion, or training-authorization evidence.

## Construction

Run this one-time, non-overwriting command after the implementation is
committed:

```powershell
python scripts/build_vasu_140m_assistant_authored_internal_suite.py
```

It writes `evaluation/fixtures/vasu_140m_assistant_authored_internal_v1/`.
The generator does not import a model or access checkpoints, training datasets,
schedules, optimizers, authorization records, or private Age keys. It changes
no model, tokenizer, dataset, mask, checkpoint, optimizer/scheduler state,
training configuration, or exact-resume artifact.
