# VASU-140M Base-Model Evaluation v2 Design Review Packet

Review the design as a pre-training, base-model-specific evaluation contract.

Read `AGENTS.md`, `docs/PROJECT_STATUS.md`,
`docs/VASU_140M_BASE_PRETRAINING_READINESS.md`,
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`,
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_AUDIT_20260801.md`,
`evaluation/suites/vasu_capability_v1.json`, `evaluation/framework/`,
`evaluation/metrics.py`, and the relevant evaluation tests.

Confirm raw-continuation interface correctness, all six dimensions,
development/held-out isolation, contamination binding, direct likelihood,
reproducible generation, resumable identities, confidence intervals, separate
automatic/manual reporting, and non-authorization.

Run:

```powershell
python -m pytest tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`.
Acceptance authorizes only later implementation review. It does not freeze
prompts, open held-out sets, run a model, construct data, or authorize training.
