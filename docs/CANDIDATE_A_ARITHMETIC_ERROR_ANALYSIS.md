# Candidate A verified-arithmetic-v2 error analysis

Status: **diagnostic only; no successor experiment is authorized**
Date: 2026-07-27
Checkpoint analysed: checkpoints/vasu_60m/capability_cpt_a_factual_20m_v2/final.pt
Checkpoint SHA-256: 8a54ff13ca3ca3dac385270b2160c34e639a9ca8178603df33d30d54f0bd6103

## Scope and immutable evidence

This is a read-only analysis of Candidate A's completed arithmetic treatment. It does not alter the evaluator, checkpoint, tokenizer, arithmetic release, schedule, or Candidate B. Greedy evaluation used the unchanged plain serialization Question: {prompt}\nAnswer:, a 16-token maximum, and the strict exact parser in evaluation/verified_arithmetic.py.

| Evidence | SHA-256 |
| --- | --- |
| Arithmetic-v2 manifest | 8da3a1acfb2d2481c295b0d226c4695613b56670989d001fa27e65bed366115c |
| Arithmetic tokens | 9abbab5e6a3d6d1121d47a89e9e829a3b51b072af2dc3a9daed5bae5d6fe474a |
| Arithmetic loss mask | beaf4d1f5dc325d1fc2a37cf8adc6677738e009fdf68d87fbf1bb032013098c4 |
| Final development summary | 7789dfe8b478bf2d1c0b83693e1fc18213969c65beede04bb49517e4184992e3 |
| Final held-out summary | a9718e99bbfb0eed8bad6c945866a31bc34d002a10fae02311dd39174042f669 |
| Final development run manifest | 9b06f29a0a1ea4a0d2dc667e5cabc4182c29f436769f6ee3e6141862d85c70b5 |
| Final held-out run manifest | 1cf5b2636596b9ceeea783059ab2cf81a2f02b7a1163f4bfa5a145a2170630d1 |

The completed mixture assigned 3,907 of 78,144 records (1,000,192 of 20,004,864 positions, 5.0%) to arithmetic. Its 3,261 packed arithmetic records were replayed 1.198 times; 646 appeared twice.

## Outcome pattern

Candidate A produces short answer-shaped continuations, but not reliable calculation. Development was 117/1000 (11.7%) exact, 881 incorrect, two prompt-leakage outcomes, and no malformed outcomes. Independent held-out evaluation was 124/1000 (12.4%), 873 incorrect, three malformed, and no prompt leakage. Outputs averaged 4.45 development tokens and 4.71 held-out tokens (medians 4 and 5). The three held-out malformed outputs were fraction responses; none exhausted the 16-token limit. Truncation is not the main failure.

| Held-out operation | Correct / total | Interpretation |
| --- | ---: | --- |
| Numeric property | 70 / 90 | All 90 outputs were no; success occurs only where that default is correct. |
| Comparison | 44 / 91 | The model defaults strongly to > (30/91 outputs). |
| Fraction | 10 / 91 | 66 outputs were non-fractions and three malformed. |
| Addition, subtraction, multiplication, exact division, percentage, sequence, mixed expression, word problem | 0 / 91 each | No direct-calculation capability was measured. |

By answer type, held-out booleans were 70/90, comparison symbols 27/49, integers 27/789 (3.4%), and reduced fractions 0/72. Development has the same shape (65/90, 26/48, 26/789, and 0/73). This is not a split-specific anomaly.

Across 876 held-out non-correct outputs, a conservative diagnostic classification found 544 unrelated numeric answers, 190 copied prompt operands, 79 wrong-format/non-numeric answers, 54 prompt fragments, three malformed outputs, three within 1% of the expected numeric value, and three within ten. These are diagnostics only, not alternate scoring. Near-miss arithmetic is rare. Frequent held-out answers were no (90), > (52), -1/23 (15), 649 (12), -136 (11), and -1366 (11): response priors, not a calculation strategy.

| Failure family | Prompt | Output | Expected |
| --- | --- | --- | --- |
| Sign/copied operand | 1923 + 1602 + 1967 = ? | -1923 | 5492 |
| Operand copying | sequence 1981, 1978, 1975, 1972, ? | 1981 | 1969 |
| Small numeric miss | sequence 1899, 1903, 1907, 1911, ? | 1919 | 1915 |
| Format/type failure | Simplify 4/20. | > | 1/5 |
| Comparison default | 1558 ? 1903 | > | < |

Only one development result resembles a digit transposition (1008 - 1499 gave -149, expected -491); it is not a recurring mechanism. There is no evidence that extra text, long continuations, or truncation explains the score.

## Training data, packing, and tokenizer diagnosis

The programmatically verified release is balanced by logical operation: 32,000 train examples, 2,909 each for ten operations and 2,910 additions. It contains 330 tier-1, 4,555 tier-2, 12,760 tier-3, and 14,355 tier-4 examples. Training templates are balanced: 10,692 direct_v1, 10,660 calculate_v1, and 10,648 number_only_v1. Train operands are 20--999, dev is 1000--1499, and held-out evaluation is 1500--1999; templates are held out. This deliberately measures range and wording generalization, but is harder than in-range template recall.

Packing is efficient and not a boundary-mask defect: 3,261 fixed 257-token records, 796,979 real tokens, 41,098 PAD tokens (4.90%), and 95.10% utilization. There are 764,979 supervised next-token targets (91.63% of arithmetic positions). Every real within-example target including EOS is enabled; EOS-to-next-example and PAD-involved transitions are disabled. The stored mask is converted as stored_mask[1:], so it aligns with shifted targets. No example is split, and each ends in EOS.

The source is balanced by examples but not exactly target-token mass: multiplication averages 18.94 supervised targets/example and sequence 31.94. That modest skew cannot explain zero direct-operation successes. More important, arithmetic occupied only about one million continuation positions in the 20M treatment and the dataset was seen only 1.2 times.

The unchanged 32,000-token byte-level BPE tokenizer is functional but not number-specialized. Train answers average 3.08 tokens; held-out answers 3.27. Held-out fractions average 5.57 answer tokens, integers 3.42, and comparison symbols 1.55. 100 is one token, but 99, 999, 1000, and 1499 take two; -999 takes three; 99/100 takes four. > takes two tokens whereas <, yes, and no take one. This plausibly compounds difficulty for signed, fractional, and longer answers, but does not justify a tokenizer change: the identity is shared by train and evaluation, and fragmentation is explanatory rather than a demonstrated defect.

## Training trajectory

The fixed 64-example development proxy rose from 0 at step 100 to 0.125 at step 800, then plateaued through step 2,442 (apart from 0.109375 at step 1,900). FineWeb/Wikimedia proxy losses improved monotonically from 3.308951/3.313093 to 3.272375/3.228615 while arithmetic exactness did not. Generic likelihood loss is therefore not a sufficient arithmetic selection signal.

best_arithmetic.pt was saved at step 1,600, with a 0.125 proxy score and zero proxy malformed rate. Its full development score was 113/1000, below final.pt at 117/1000; final.pt held out at 124/1000. A longer identical 5% mixture is not justified as the next test. The likely limiting factors are too little direct arithmetic exposure for the objective, the intentional range/template shift, and multi-token numeric outputs—not evaluator, mask, or tokenizer failure.

## Recommended next experiment: controlled arithmetic-exposure ablation

Do **not** repurpose Candidate B. Plan a separately named, unapproved sequential pair, for example capability_cpt_d_control_10m_v1 and capability_cpt_e_arithmetic_exposure_10m_v1. Both start from Candidate A final.pt with a fresh optimizer/scheduler. The matched control is essential: an unpaired sequential continuation cannot distinguish arithmetic exposure from ordinary extra training.

The treatment changes one scientific factor: it substitutes FineWeb replay with arithmetic while keeping Wikimedia at 9%, parent, budget, runtime, schedule construction, evaluation, and checkpoint policy matched. It uses the existing immutable arithmetic-v2 release; it creates no new dataset, template, or hidden chain-of-thought target.

| Setting | Matched control | Primary treatment |
| --- | ---: | ---: |
| FineWeb replay | 35,556 records (91.001%) | 12,113 records (31.002%) |
| Wikimedia | 3,516 records (8.999%) | 3,516 records (8.999%) |
| Verified arithmetic v2 | 0 | 23,443 records (59.999%) |
| Arithmetic effective passes | 0 | 7.189; below the hard limit of 10 |

Each branch must have 39,072 records, 19,536 two-record microbatches, 1,221 complete 16-microbatch updates, and 10,002,432 positions (39,072 * 256). Retain VASU-60M, sequence length 256, batch 2, accumulation 16, seed 42, no shuffle, standard AdamW backend, weight decay 0.1, clipping 1.0, cosine schedule of 1,221 steps, peak LR 1e-5, minimum 1e-6, and 24 warmup updates (rounded 2%). Validate every 100 updates, checkpoint every 200, use explicit exact resume only, retain the 10-GiB disk margin, and retain Candidate A thermal limits (82C warning, 87C abort after two readings). Use new isolated output directories.

The model, batch, and context match the completed run, so no VRAM growth is expected; Candidate A observed about 1.38 GiB peak allocation. Its persisted step-200 to final timestamps imply about 0.87 seconds/update after startup. Estimate 18--25 minutes compute per 1,221-update branch; run serially and budget 45--75 minutes wall time for both including validation, saves, and thermal cooldown. This is an estimate, not launch authorization.

### Pre-registered evaluation and decision rules

Use the full 1,000-example development split for checkpoint selection only. Select by highest direct-operation macro exact accuracy (addition, subtraction, multiplication, exact division, percentage, sequence, mixed expression, word problem), then overall development exactness, then lower malformed rate and lower FineWeb/Wikimedia validation loss. Do not select on held-out evaluation. Evaluate the selected treatment and control once on the frozen held-out split using the unchanged greedy evaluator, parser, serialization, and 16-token cap.

Promotion evidence must require all of the following:

- held-out overall exact accuracy at least 0.20 and a Wilson 95% lower bound above Candidate A final's 0.124 point estimate;
- direct-operation macro exact accuracy at least 0.10, with nonzero successes in at least four of eight direct families;
- treatment improvement over its matched control on held-out overall accuracy with a two-proportion 95% confidence interval excluding zero;
- FineWeb loss no worse than 3% relative to Candidate A final, Wikimedia loss no worse than 5%, no material capability-v1 regression, and no repeated-bigram-ratio increase above 0.05;
- zero integrity, exact-resume, thermal, disk, or authorization failures.

Terminate for existing hard runtime failures, two consecutive validation intervals beyond the loss guard, or any hash/identity validation failure. Five flat arithmetic-development validations after step 600 are a review trigger, not an automatic stop. Lower proxy loss or a technical-gate pass is not promotion evidence.

### One permitted fallback

If high arithmetic exposure improves only format priors or fails the direct-operation gate, the sole fallback is a separately versioned arithmetic-v3 source ablation that changes **only** training range/template coverage while retaining the same parent, 10M pair budget, 60% treatment share, and control. It needs its own generator review, immutable manifest, leakage proof, schedule, decision record, and authorization. Do not add hidden chain-of-thought by default; test concise verified answers first.

## Compatibility and authorization

This analysis changes no runtime artifact. Existing checkpoints remain load-compatible; tokenizer, datasets, splits, masks, evaluator scoring, architecture, and exact-resume behavior are unchanged. Candidate A remains the preferred continued-pretraining base; masked Alpaca v3 remains the preferred instruction-tuned assistant. Candidate B remains unauthorized.

The proposal requires new decision records, configs, deterministic manifests/schedules, validation artifacts, hash-bound authorization records, a clean reviewed tree, and explicit approval before any training.
