# Candidate D Arithmetic Treatment Error Analysis

Status: read-only post-closeout analysis; no new training authorization

## Scope

This analysis compares the completed Candidate D treatment with Candidate A
and the matched Candidate D control on the immutable verified-arithmetic-v2
evaluation. It does not modify a checkpoint, tokenizer, dataset, mask,
schedule, evaluator, or authorization scope.

## Observed result

Candidate D treatment used a 60% operation-aware arithmetic allocation for an
additional matched 10M-token continuation from Candidate A. It nevertheless
reached 112/1000 development exact accuracy, equal to the non-arithmetic
control and below Candidate A's 117/1000. Held-out exact accuracy was 121/1000,
below Candidate A's 124/1000.

The treatment preserved the same narrow profile rather than expanding it:

| Operation family | Candidate A held-out | D treatment held-out |
| --- | ---: | ---: |
| Numeric property | 0.778 | 0.778 |
| Comparison | 0.484 | 0.429 |
| Fraction | 0.110 | 0.110 |
| Addition | 0.000 | 0.000 |
| Subtraction | 0.000 | 0.000 |
| Multiplication | 0.000 | 0.000 |
| Exact division | 0.000 | 0.000 |
| Mixed expression | 0.000 | 0.000 |
| Word problem | 0.000 | 0.000 |

The 1/91 successes in percentage and sequence are isolated observations, not
evidence of a new direct-operation capability. Six held-out fraction responses
were malformed; no truncation, unanswered output, or prompt leakage was found.

## Interpretation

The matched control removes ordinary additional continuation as an explanation,
and the high arithmetic allocation removes insufficient arithmetic exposure as
a sufficient explanation for this design. The evidence supports only a narrow
recognition/comparison behavior already present in Candidate A; it does not
support general numeric calculation. Increasing the duration or arithmetic
share of this same objective is not justified by these results.

## Future-work boundary

Candidate B remains unauthorized and must not be repurposed. Any future
arithmetic proposal requires a newly named experiment, a materially different
testable hypothesis, a matched control, immutable new planning artifacts,
held-out evaluation gates, and a separate human authorization. Candidate A
final remains the preferred continued-pretraining base.
