# VASU-140M Development Inventory Promotion Execution Audit

Status: one-shot execution completed; independent result review pending.

The explicitly authorized command ran once at reviewed commit
`b6a6f55d2c1d3b99cd2ea6f4c5e56869fd03852d` and created only
`evaluation/candidates/vasu_140m_base_v2_development_v1`.

## Results

| Dimension | Records | Promoted inventory SHA-256 |
|---|---:|---|
| arithmetic | 1,000 | `fd6f75ebed75ee0fcb2bfc36fde29a1057a2e67a162eec7438d0ef727d0266bf` |
| factuality | 200 | `958257c55a0fdd4782470324cc5d1128d70ae5ef9dbf2d029c67f4e88fdb4008` |
| manual review | 60 | `ee214389a90b189432ad5bb10bf14903c62cc22226fa97b646a23a31174e3cea` |
| repetition | 120 | `a9acb73757acf8321e15dc0b4f1eb0936e5c983976b125d39cbc6cf419676a03` |
| robustness | 120 | `48d4396b1883af7fa345ce445ea2d0e726126ab625add9dc55e417a2da54460c` |

Receipt SHA-256:
`3852c57e4704efc213cc0fe34a86a58f37fa60bc43a21dae3b3c6bc9aea85818`.

All five final-path manifests passed repository-file validation. Payload,
provenance, and contamination bytes are preserved from the accepted fixtures.
Every promoted manifest has `fixture_only=false`, while
`production_suite_frozen`, `evaluation_run_authorized`, and
`training_authorized` remain false.

No retry occurred. No held-out payload or key was accessed. No source was
admitted, no likelihood inventory was constructed, no evaluation ran, no data
release or checkpoint was created or accessed, and no training action occurred.
