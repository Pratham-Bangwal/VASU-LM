# Capability Experiment Governance

This framework provides a review-stage reproducibility boundary for capability
experiments. It is deliberately separate from the runtime authorization gate:
it gathers and validates evidence but cannot authorize training.

## Workflow

1. Build immutable, non-authorizing data releases. Their manifests must state
   `training_authorized: false` and bind every artifact by SHA-256.
2. Freeze evaluation snapshots and report their paths and hashes alongside the
   release identities.
3. Use `vasu.training.experiment_governance.build_review_packet` to collect
   explicit release, schedule, configuration, and evaluation identities. The
   function rejects any supplied release/schedule/configuration that is not
   explicitly non-authorizing.
4. Conduct independent review. A review packet is evidence only; it cannot be
   converted into an authorization record.
5. Only after review may a separately scoped schedule, configuration, runtime
   validation, and authorization process be proposed. Existing
   `vasu.training.capability_cpt.require_training_authorization` remains the
   sole launch-time enforcement boundary.

## Candidate E application

`scripts/report_candidate_e_governance_preflight.py` reports Candidate E's
current state without creating a release. It hash-binds the review packet,
budget/evaluation protocol, pre-release checklist, and tokenizer audit. Its
expected state is `review_evidence_complete_release_not_approved`; this is a
healthy blocked state, not a training-ready state.

Run from the repository root:

```powershell
python scripts/report_candidate_e_governance_preflight.py
```

The report must continue to state `training_authorized: false`. A Candidate E
release, schedule, configuration, authorization, or training process requires
separate review and explicit authorization.
