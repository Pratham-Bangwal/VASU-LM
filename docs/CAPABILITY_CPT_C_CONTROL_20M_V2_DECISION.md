# Candidate C Scientific Decision Record

Status: **completed; conservative control accepted as a continued-pretraining
base candidate**

Decision date: 2026-07-26  
Candidate: `capability_cpt_c_control_20m_v2`  
Parent: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`  
Candidate checkpoint: `checkpoints/vasu_60m/capability_cpt_c_control_20m_v2/final.pt`

## Scope and training record

Candidate C completed successfully with 20,004,864 processed tokens and 2,442
optimizer updates. The mixture was 91% FineWeb replay and 9% approved
Wikimedia factual data. No arithmetic data was used for training.

The selected `final.pt` and `best_wikimedia.pt` produced identical evaluation
metrics. Candidate C is a continued-pretraining base checkpoint, not an
instruction-tuned assistant checkpoint.

## Hash-bound artifacts

| Artifact | SHA-256 |
|---|---|
| Parent checkpoint | `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f` |
| Candidate `final.pt` | `f67b14c414fa7d5b38e2e9d679303c76c4e35c34d62a7de9c83cbf0fe9a8e3c2` |
| Candidate `best_wikimedia.pt` | `6005241903acb99b00d09463679395304c952e1a74a41a2efba5dc7e813093dc` |
| Factual evaluation comparison | `356d8681eacf51edac7307a8bf4297aded69eefd529f13f2b4e305e2a3e5de26` |
| Factual benchmark | `17c356f0b3a5093511402b08f307b770b8cc4443b3ca3e1cceaa0eb7044115b9` |
| Factual evaluation configuration | `9d73d7decbb4173a43a2053011518e495a156e9b8461d3c9a87778f23c9fc4c9` |
| Capability-v1 promotion report | `f1888f2123f9372a74b2864c36e238af47d27764950c99a19dde33845b49b376` |
| Arithmetic parent category summary | `89716ad9a94ebf94a42e69ebfd245e59788708cefd15e52355011904d14ddef8` |
| Arithmetic Candidate C category summary | `619b0a64e2d2c3504aaa3839060076a431ecde99dba9d6893381fac53028c972` |
| Arithmetic parent run manifest | `5ad1e4834186cc833458520f8fdb1528c012765461448133e93e3694e8f51425` |
| Arithmetic Candidate C run manifest | `3d60586ae79cf2104ed7a5d75ffbe8140da5ee2b2d366e36545a954e93b0e791` |

## Factual evaluation

| Metric | Parent | Candidate C | Change |
|---|---:|---:|---:|
| FineWeb loss | 3.356130 | 3.302959 | -0.053170522674918175 (-1.5843%) |
| Wikimedia loss | 3.393557 | 3.274323 | -0.11923486536199412 (-3.5136%) |
| Cloze reported score | 0.070 | 0.110 | +0.040 |
| Multiple choice | 0.410 | 0.420 | +0.010 |

The normalized cloze difference was estimated at +0.04 with CI95
[+0.01, +0.08], distinguishable from zero. Other benchmark confidence
intervals crossed zero. The exact comparison artifact is
`evaluation/results/capability_cpt_c_control_20m_v2/factual_cpt_v2_comparison.json`.

## Arithmetic development evaluation

The full 1,000-example development evaluation found:

| Metric | Parent | Candidate C |
|---|---:|---:|
| Exact accuracy | 0/1000 | 0/1000 |
| Malformed rate | 0.469 | 0.449 |
| Prompt leakage | 0.530 | 0.549 |
| Truncated | 0.970 | 0.973 |

There was no measurable arithmetic improvement or exact-accuracy regression.
This is a neutral result and is expected because Candidate C received no
arithmetic training.

## Capability-v1 gate

The deterministic continued-pretraining promotion report is:
`evaluation/results/capability_cpt_c_control_20m_v2_capability_v1_greedy/promotion_report.json`.

- Targeted objective gate: passed.
- General regression gate: passed.
- Human review required: false.
- Blocking reasons: none.
- Final status: passed.

Both parent and candidate scored zero on all objective categories. Therefore,
this gate establishes no measurable objective regression, not capability
improvement.

## Repetition and qualitative behavior

- Concept-explanation repeated-bigram ratio changed from
  `0.33333333333333337` to `0.4871794871794872`: worse.
- General-language repeated-bigram ratio changed from
  `0.3571428571428571` to `0.26190476190476186`: improved.

Repetition behavior is mixed and must remain a limitation in downstream
selection.

## Scientific decision

Candidate C is scientifically successful as the conservative control
continued-pretraining run. It improves FineWeb validation loss, improves
Wikimedia validation loss, produces a statistically distinguishable normalized
cloze improvement, does not improve arithmetic, shows no measurable objective
regression in capability-v1, and has mixed repetition behavior.

Candidate C is eligible to become the preferred continued-pretraining base
candidate. It does **not** automatically replace the instruction-tuned Alpaca
v3 checkpoint as the preferred interactive assistant. The Alpaca v3 checkpoint
remains the assistant-specific model until a separately authorized and
evaluated instruction-tuning decision is made.

## Follow-on authorization decisions

- Candidate A may proceed to a separate authorization review using its own
  written rationale, configuration identity, and evaluation gates.
- Candidate B remains unauthorized and conditional. It requires Candidate A
  evidence, technical integrity, arithmetic-specific review, broad/factual
  regression checks, and a written B-specific authorization decision.
- This record authorizes neither Candidate A nor Candidate B.

## Compatibility and limitations

The decision does not alter the VASU-60M architecture, tokenizer, checkpoint
schema, FineWeb/Wikimedia datasets, schedules, or existing assistant
checkpoints. Candidate C remains load-compatible with the VASU-60M model and
is intentionally separate from the VASU-31M and Alpaca-v3 assistant roles.

The experiment is not evidence of reliable arithmetic, general reasoning,
factual authority, or assistant behavior. Arithmetic exact accuracy remains
zero, objective capability scores remain uninformative, and repetition is
mixed. Further claims require new preregistered evaluation.
