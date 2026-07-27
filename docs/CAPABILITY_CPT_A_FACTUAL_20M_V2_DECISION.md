# Candidate A Scientific Decision Record

Status: **completed; controlled arithmetic treatment accepted as the preferred
continued-pretraining base checkpoint**

Decision date: 2026-07-27  
Candidate: `capability_cpt_a_factual_20m_v2`  
Parent: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`  
Selected checkpoint:
`checkpoints/vasu_60m/capability_cpt_a_factual_20m_v2/final.pt`

## Scope and training record

Candidate A completed its matched controlled treatment run with 20,004,864
processed tokens and 2,442 optimizer updates. It shares Candidate C's immutable
FineWeb step-200,000 parent and all accounting, optimizer, scheduler, warmup,
batch, sequence-length, and accumulation settings. The treatment mixture was
86% FineWeb replay, 9% approved Wikimedia factual data, and 5% verified
arithmetic v2; Candidate C was the 91% FineWeb, 9% Wikimedia, 0% arithmetic
control. The intended primary experimental variable is the 5% arithmetic
source.

## Hash-bound evidence

| Artifact | SHA-256 |
| --- | --- |
| Parent checkpoint | `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f` |
| Candidate A `final.pt` | `8a54ff13ca3ca3dac385270b2160c34e639a9ca8178603df33d30d54f0bd6103` |
| Final arithmetic development summary | `7789dfe8b478bf2d1c0b83693e1fc18213969c65beede04bb49517e4184992e3` |
| Final arithmetic held-out summary | `a9718e99bbfb0eed8bad6c945866a31bc34d002a10fae02311dd39174042f669` |
| Factual comparison | `77e9d9bf37bfcc4bae7facf41c32817dd94875cefd553b7ec79705eab31d2a91` |
| Capability-v1 promotion report | `c7d816f7ad62238eb45d3d5f7cb774f393dec189d954d6b2cadd1ef1a664862e` |

Arithmetic evidence is recorded in
`evaluation/results/candidate_a_final_arithmetic_dev/category_summary.json`
and `evaluation/results/candidate_a_final_arithmetic_eval/category_summary.json`.
Factual evidence is recorded in
`evaluation/results/capability_cpt_a_factual_20m_v2/factual_cpt_v2_comparison.json`.
The capability gate is recorded in
`evaluation/results/capability_cpt_a_factual_20m_v2_capability_v1_greedy/promotion_report.json`.

## Arithmetic evaluation and checkpoint selection

| Checkpoint | Development exact accuracy | Held-out evaluation exact accuracy |
| --- | ---: | ---: |
| FineWeb step-200,000 parent | 0 / 1000 | — |
| Candidate C control | 0 / 1000 | — |
| Candidate A `best_arithmetic.pt` | 113 / 1000 (0.113) | — |
| Candidate A `final.pt` | 117 / 1000 (0.117) | 124 / 1000 (0.124) |

Candidate A `final.pt` is selected over `best_arithmetic.pt`: it has the
higher development exact accuracy and supplies the completed held-out result.
The final development malformed rate is 0.0 and prompt-leakage rate is 0.002;
the held-out malformed rate is 0.003 and prompt-leakage rate is 0.0.

The improvement is narrow. It is concentrated in comparisons and numeric
properties; addition, subtraction, multiplication, division, percentages,
mixed expressions, sequences, and word problems remain at or near zero. This
is evidence of narrow arithmetic-format and comparison capability, not general
arithmetic reasoning.

## Factual and broad-language retention

| Metric | Parent | Candidate C | Candidate A |
| --- | ---: | ---: | ---: |
| FineWeb loss | 3.356130 | 3.302959 | 3.305468 |
| Wikimedia loss | 3.393557 | 3.274323 | 3.276147 |
| Cloze | 0.070 | 0.110 | 0.110 |
| Multiple choice | 0.410 | 0.420 | 0.450 |

Against the parent, Candidate A improves FineWeb loss by approximately 1.51%
and Wikimedia loss by approximately 3.46%. Its normalized cloze change is
`+0.04`, CI95 `[+0.01, +0.08]`, distinguishable from zero. The multiple-choice
confidence interval crosses zero. Relative to Candidate C, Candidate A's
FineWeb and Wikimedia losses are approximately 0.076% and 0.056% worse,
respectively: operationally very small likelihood costs accompanying the
measurable arithmetic result.

## Capability-v1 and repetition limitations

The continued-pretraining capability-v1 greedy gate passed its targeted and
general-regression checks, required no human review, and has no blocking
reasons. Parent and Candidate A both scored zero on every objective category.
Therefore this gate establishes no measurable regression, not capability
improvement.

Concept-explanation repeated-bigram ratio worsened from
`0.33333333333333337` to `0.4782608695652174`. General-language repetition
changed from `0.3571428571428571` to `0.3589743589743589`, approximately
unchanged. Repetition remains a limitation.

## Scientific decision

Candidate A is scientifically successful as the controlled arithmetic
treatment branch. It demonstrates measurable held-out arithmetic improvement
while retaining broad-language and factual gains at minimal likelihood cost
against Candidate C. Candidate A `final.pt` is promoted as the preferred
continued-pretraining base checkpoint.

This does not promote Candidate A as the preferred interactive assistant.
Masked Alpaca v3 remains the preferred instruction-tuned assistant. Candidate
C remains a successful control and historical comparison checkpoint. Candidate
B remains unauthorized and conditional; this decision neither authorizes nor
starts Candidate B.

## Compatibility

This decision changes only documented checkpoint selection. It does not alter
the VASU-60M architecture, checkpoint schema, tokenizer, datasets, masks,
schedules, training configurations, or hyperparameters. Candidate A and
Candidate C remain load-compatible VASU-60M checkpoints.
