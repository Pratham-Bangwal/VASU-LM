# Research Platform

VASU's research platform is a read-only reproducibility layer. It complements
the existing runtime authorization gate; it never creates data, changes an
experiment identity, approves training, or launches a process.

## Components

- `vasu.utils.lineage.build_lineage_index` scans existing manifests,
  evaluation results, and authorization records into deterministic path/size/
  SHA-256 entries. Run `python scripts/build_vasu_lineage_index.py` to inspect
  the repository without writing an index artifact.
- `vasu.utils.experiment_report.compare_evaluation_snapshots` compares only
  shared numeric leaves in two JSON evaluation snapshots and binds both source
  hashes in its report.
- `vasu.training.platform_preflight.preflight_configuration` accepts only a
  config that explicitly says `training_authorized: false`, verifies declared
  artifact identities, and reports required safety-gate states.
- `vasu.training.experiment_governance.build_review_packet` binds reviewed
  release/schedule/configuration/evaluation paths and rejects authorizing
  inputs.

## Research workflow

1. Build or inspect a non-authorizing dataset release and preserve its manifest.
2. Freeze evaluation JSON snapshots before comparison.
3. Generate read-only lineage, comparison, and preflight evidence.
4. Complete independent review.
5. Use the existing capability runtime authorization process for any separately
   approved experiment. The platform report itself is never sufficient.

## Candidate E

Candidate E is intentionally represented as review-ready but release-absent.
`python scripts/report_candidate_e_governance_preflight.py` verifies its review
documents and must report `training_authorized: false`. This protects the
boundary between preparation and a separately reviewed data/training proposal.
