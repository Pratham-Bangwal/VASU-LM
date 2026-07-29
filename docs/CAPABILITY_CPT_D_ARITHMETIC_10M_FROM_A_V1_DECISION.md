# Candidate D Arithmetic Treatment Scientific Decision Record

Status: **completed; rejected for promotion**

Decision date: 2026-07-29
Candidate: `capability_cpt_d_arithmetic_10m_from_a_v1`
Parent: `checkpoints/vasu_60m/capability_cpt_a_factual_20m_v2/final.pt`
Selected checkpoint: `checkpoints/vasu_60m/capability_cpt_d_arithmetic_10m_from_a_v1/final.pt`

## Completion and integrity

The treatment completed all 1,221 optimizer updates and 10,002,432 processed
tokens. The final checkpoint has SHA-256
`c17f9a481d6ae235f4264cbe9ca2775529291da303ff1e38a2d1d3ac8258aaa2`.
It loaded strictly as a finite VASU-60M model at `global_step=1221`.

The runtime stopped safely once for its disk checkpoint safeguard and once for
the sustained GPU thermal safeguard at step 1,180. Both resumptions were
explicit and continued from validated same-candidate checkpoints; the final
accumulation group completed without replaying the parent or changing the
schedule, data identities, tokenizer, or hyperparameters.

## Frozen arithmetic evaluation

| Checkpoint | Development exact accuracy | Held-out exact accuracy |
| --- | ---: | ---: |
| Candidate A final | 117 / 1000 (0.117) | 124 / 1000 (0.124) |
| Candidate D control final | 112 / 1000 (0.112) | Not evaluated |
| Candidate D treatment final | 112 / 1000 (0.112) | 121 / 1000 (0.121) |

The treatment did not improve over Candidate A on development and was 0.003
absolute lower on held-out exact accuracy. It therefore fails the predeclared
requirement to exceed Candidate A held-out arithmetic accuracy by at least 5
percentage points. All four direct core operations (addition, subtraction,
multiplication, and exact division) scored 0% exact on held-out evaluation,
also failing the required core-operation threshold. Six held-out responses
were malformed (0.006); this does not alter the conclusion and malformed-rate
reduction alone was never sufficient for promotion.

Evidence is retained in
`evaluation/results/candidate_d_treatment_final_arithmetic_dev/` and
`evaluation/results/candidate_d_treatment_final_arithmetic_eval/`. Both runs
used the immutable verified-arithmetic-v2 splits, the final-checkpoint hash,
the unchanged tokenizer, greedy decoding, and 16 generated-token maximum.

## Broad-language and factual retention

| Metric | Candidate A | D control | D treatment final |
| --- | ---: | ---: | ---: |
| FineWeb loss | 3.305468 | 3.299042 | 3.308870 |
| Wikimedia loss | 3.276147 | 3.264998 | 3.262345 |
| Normalized cloze | 0.110 | 0.120 | 0.110 |
| Multiple choice | 0.450 | 0.440 | 0.440 |
| Greedy repetition | 0.718260 | 0.723730 | 0.724377 |
| Sampled repetition | 0.271450 | 0.295317 | 0.296281 |

Relative to the matched control, treatment FineWeb loss is 0.30% higher and
Wikimedia loss is 0.08% lower, both inside the matched-pair retention limits.
Cloze is two points below control, multiple choice is tied, and repetition is
effectively unchanged. These are retention findings, not evidence of arithmetic
promotion.

The deterministic capability suite also found no objective arithmetic,
factual, formatting, or safety/uncertainty improvement: Candidate A and the
treatment both scored zero on its objective prompts. Its passing gate is not
interpreted as a capability gain because the test had no sensitivity here.

## Decision and compatibility

Candidate D arithmetic treatment is rejected for promotion. Candidate A final
remains the preferred continued-pretraining base, and masked Alpaca v3 remains
the preferred instruction-tuned assistant. Candidate D control remains the
accepted matched-control checkpoint. Candidate B remains unauthorized.

`training_authorized` has been returned to `false` to prevent accidental
relaunch. This closeout does not alter the VASU-60M architecture, checkpoint
schema, tokenizer, source data, masks, schedules, evaluation splits, or
existing checkpoints. All treatment checkpoints and evaluation evidence are
preserved for reproducibility; no further Candidate D training is authorized
by this decision.
