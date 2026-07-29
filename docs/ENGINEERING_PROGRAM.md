# VASU Engineering Program

Status: active engineering roadmap. All implementation work remains
non-authorizing unless an experiment receives a separate reviewed authorization.

| Workstream | Current deliverable | Training/data boundary |
| --- | --- | --- |
| Candidate E specification | New-ID/range/template/split specification and review decision packet | No production data release until independent review |
| Candidate E dry validation | Fixture-only deterministic release and mask/replay validation | No Candidate E source generation |
| Arithmetic and capability evaluation | Versioned diagnostics, paired comparisons, and frozen evaluation contracts | Evaluation only; no promotion by tooling |
| Safety and configuration | Hash-bound read-only preflight, disk/thermal/resume/checkpoint contracts | Cannot turn authorization on |
| Reporting and lineage | Read-only artifact index and hash-bound JSON comparisons | Preserves historical artifacts |
| Architecture and pipeline | Profiling/audit with compatibility impact recorded before changes | No checkpoint-incompatible change without approval |
| Inference/checkpoints | Usability and reproducibility improvements with parity checks | Preserve checkpoint/tokenizer compatibility |
| Documentation and CI | Status/roadmap synchronization and regression coverage | No external publication or training side effect |

## Current completed foundations

- Candidate D control and treatment are closed; treatment was rejected under
  its pre-registered arithmetic criteria.
- Candidate E has a non-authorizing paired-target compiler, release builder,
  split/mask/budget validators, evaluation protocol, and review packet.
- The research platform has read-only lineage indexing, snapshot comparison,
  governance reports, and configuration preflight checks.

## Sequencing

1. Finish Candidate E logical-data and independent-review materials.
2. Expand evaluation/reporting and enforce regression coverage.
3. Audit architecture, pipeline, and inference improvements before any
   compatibility-affecting implementation.
4. Synchronize open-source documentation and perform an end-to-end platform
   verification.

No entry in this program authorizes Candidate E data generation, a training
configuration, a checkpoint resume, or any model-training process.
