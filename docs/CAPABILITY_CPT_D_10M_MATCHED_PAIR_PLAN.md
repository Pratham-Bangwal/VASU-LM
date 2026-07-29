# Candidate D matched sequential pair

Status: completed; treatment rejected for promotion. The control completed,
passed integrity and evaluation review, and was accepted as the matched
comparison checkpoint. The treatment then completed and was rejected under the
predeclared arithmetic-promotion criteria; see
`CAPABILITY_CPT_D_ARITHMETIC_10M_FROM_A_V1_DECISION.md`.

Both experiments start from Candidate A final:
`checkpoints/vasu_60m/capability_cpt_a_factual_20m_v2/final.pt`, SHA-256
`8a54ff13ca3ca3dac385270b2160c34e639a9ca8178603df33d30d54f0bd6103`.

Both use 39,072 records, 10,002,432 processed tokens, batch size 2, sequence
length 256, 19,536 microbatches, accumulation 16, 1,221 updates, cosine
scheduler length 1,221, and 25 warmup updates (the shared 2% policy). They
share optimizer, learning rate, seed, validation/checkpoint intervals, dropout
behavior, runtime safeguards, tokenizer, parent, and validation sets.

Control `capability_cpt_d_control_10m_from_a_v1` uses deterministic
largest-remainder allocation: FineWeb 35,556 records (91%) and Wikimedia
3,516 (9%). Treatment `capability_cpt_d_arithmetic_10m_from_a_v1` uses
FineWeb 12,112 (31%), Wikimedia 3,517 (9%), and the immutable operation-aware
arithmetic view 23,443 (60%). The only intended training-data difference is
that treatment replacement.

Treatment arithmetic weights are bound in
`configs/data/mixtures/vasu_60m_capability_d_arithmetic_10m_from_a_v1.operation_weights.json`.
Its staged curriculum is bound separately; arithmetic records are sampled with
replacement because 23,443 selections exceed the 3,272-record homogeneous
view. Comparison and numeric-property each receive 703 arithmetic records;
they cannot dominate a stage. Any changed source, mask, view, weight,
curriculum, schedule, parent, tokenizer, validation, or runtime identity
requires a new review.

Promotion thresholds, proposed before authorization: treatment must exceed
Candidate A held-out arithmetic exact accuracy by at least 5 percentage points,
improve at least four direct-operation families by at least 5 points each, and
raise at least one of addition/subtraction/multiplication/exact-division above
5% exact accuracy. It must stay within 1% relative FineWeb/Wikimedia loss of
the matched control, within 2 points factual accuracy, and within 2 points of
the control repetition rate. Malformed-rate reduction alone is insufficient.

Candidate B remains unauthorized. This completed plan authorizes neither a
follow-on training run nor promotion of the treatment checkpoint.
