# VASU-60M capability CPT ablation plan

## Arithmetic v2 decision data

The v2 release covers addition, subtraction, multiplication, exact division,
comparison, sequences, fractions, percentages, short word problems,
parenthesized mixed expressions, and numeric properties. Answers are canonical
integers, reduced fractions, comparison symbols, or yes/no values and are
recomputed from structured metadata.

V2 schedule hashes:

- A: `cc9cd9b7d2a2cd55a91c805691ce1c872fac82cd6295c5626383f8aea2ca15ce`
- B: `bcc53510589a3332dc0611bc2c5faf57c7d086c62f9fb988a490808d930cff97`
- C: `92f8675a6160c18e18a0e99cbb3fedccb1c4bc4b322055d5cd5b49602f54d381`

The warning threshold is greater than five effective passes; the hard limit
is greater than ten. Overrides are source-specific, disabled by default, and
require an approved maximum plus written justification.

## Scheduled-mixture implementation status

The candidate data plane is additive: compact `schedule.bin` files map each
logical record to a hash-bound source and local record without concatenating
sources. Standard sources receive an all-ones target mask; verified arithmetic
uses its canonical target-aligned mask. Candidate C contains no arithmetic.
Validation sources remain isolated.

Schedule hashes:

- A: `6a0baafd12e0dd4c59089abeeb81342d1ac327276f3180fdacaa7c18c7699fe8`
- B: `055e35c3579cf25fa7bc5030f3fe02fdf68d0e6bc785382fbae77dcd3f91e679`
- C: `92f8675a6160c18e18a0e99cbb3fedccb1c4bc4b322055d5cd5b49602f54d381`

All launch configurations remain unauthorized.

## Purpose and evidence

Capability v1 found zero greedy arithmetic, factual, and uncertainty passes for
the FineWeb-200k base checkpoint. Alpaca masked v3 improved greedy formatting
to one of two tasks but did not improve those base capabilities. UltraChat v2
matched greedy objective results and was blocked from promotion by incomplete
structured human review.

This plan prepares a small, matched base-model continued-pretraining ablation;
it does not authorize training, alter the preferred assistant, or promote
UltraChat.

## Approved source readiness

FineWeb replay is the validated manifest-backed continuation stream beginning
at logical offset `1,638,400,000` (step 200,000). It has 107,491,730 remaining
training tokens and keeps the original validation slice isolated.

The only approved factual artifact is the quarantined Wikimedia release
`6aa10d73`: 1,915,008 train tokens and 92,944 held-out validation tokens.
It is license/provenance/hash bound and passed the frozen release policy.
It is data-eligible, but its release policy does not itself authorize a model
run.

The approved factual capacity cannot support 15--25% of a 20M-token run
without replay. All candidate plans therefore use 9% factual material
(1,800,448 tokens) and explicitly disable replacement sampling.

No separate high-quality educational-explanation source is approved. Candidate
B omits that proposed 5% rather than adding synthetic prose.

## Matched 20M nominal candidates

All plans use 78,144 fixed 256-token records: 20,004,864 aligned tokens and
2,442 complete optimizer updates at batch size 2 and accumulation 16.

| Candidate | FineWeb replay | Wikimedia | Verified arithmetic |
| --- | ---: | ---: | ---: |
| A factual | 17,204,224 (86%) | 1,800,448 (9%) | 1,000,192 (5%) |
| B balanced | 15,203,584 (76%) | 1,800,448 (9%) | 3,000,832 (15%) |
| C control | 18,204,416 (91%) | 1,800,448 (9%) | — |

The versioned planning manifests are under `configs/data/mixtures/`.

## Arithmetic data

`scripts/prepare_vasu_verified_arithmetic_v1.py` deterministically creates
addition, subtraction, multiplication, exact integer division, comparisons,
sequences, and simple complementary fractions. It uses disjoint train,
development, and evaluation ranges; emits no capability-v1 fixture wording;
and labels all records `synthetic_verified_v1`. The smoke corpus is text-only,
not a training artifact.

```powershell
python scripts/prepare_vasu_verified_arithmetic_v1.py `
  --output-dir data/interim/capability/vasu_verified_arithmetic_v1_smoke
```

## Proposed training settings (not authorized)

- Parent: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`
- Architecture/tokenizer/context: unchanged VASU-60M / 32k / 256
- Batch / accumulation: 2 / 16; AMP; exact-resume checkpoints
- Optimizer: standard AdamW by default; fused only after opt-in validation
- Learning-rate comparison: 1e-5, 2e-5, 5e-5
- Recommendation: **1e-5** for the 20M ablation. It is the lowest candidate,
  limits forgetting under a shifted mixture, and matches the prior guarded
  factual-CPT setting. 2e-5 and 5e-5 remain deferred alternatives, not defaults.
- Warmup: 2% (49 steps); weight decay: 0.1; gradient clip: 1.0
- Validation: every 100 steps; checkpoint: every 200; thermal stop: 88 C

Each candidate must use a new isolated checkpoint directory and retain best,
latest, and step-200 checkpoints. A 50M extension is allowed only after the
20M candidate passes its configured continued-pretraining gate.

## Required readiness blockers

Training commands are intentionally deferred. Before a launch, VASU needs:

1. hash-bound tokenization of the arithmetic train/development/evaluation
corpus at the exact planned capacities;
2. a generalized manifest-aware fixed-record builder for three sources (the
   existing production builder is intentionally limited to FineWeb plus
   Wikimedia);
3. a training configuration whose source schedule, mixture hash, validation
   splits, and isolated checkpoint directory pass a CPU DataLoader smoke test;
4. parent checkpoint, tokenizer, mixture, and source hashes revalidated.

Future evaluation must compare the parent and each candidate using greedy and
five-seed capability-v1 runs, held-out FineWeb/Wikimedia/arithmetic metrics,
continuation prompts, and separate degeneration/repetition statistics. A
candidate may proceed to 50M only if targeted objective accuracy improves,
FineWeb loss and repetition remain within the configured limits, no new empty
or degeneration failure appears, and the continued-pretraining gate passes.

## Arithmetic fixed-record contract (implemented, serialization deferred)

Verified arithmetic capability data uses independent fixed records of 257
tokens for VASU's 256-token context. Each tokenized logical example ends with
the tokenizer EOS token. The packer greedily adds complete examples in a
deterministic replay order; it never splits or truncates an example. The
exact-fill-only policy was rejected because arbitrary variable-length examples
cannot reliably tile a fixed record width.

After the final complete example in a record, the remaining positions contain
the existing tokenizer PAD token. No example appears after the first PAD.
Every record records its example IDs and spans, replay epoch, used-token count,
padding-token count, and utilization ratio; aggregate statistics report real
and padding tokens plus mean/minimum/maximum utilization. An optional warning
threshold makes low utilization visible without silently rejecting a corpus.

A record yields `x = tokens[:256]`, `y = tokens[1:257]`, and an authoritative
target-aligned loss mask of 256 positions. Normal within-example targets,
including final-answer-to-EOS, are supervised. EOS-to-first-token transitions
between examples are masked. EOS-to-PAD and PAD-to-PAD transitions are masked,
and any position with PAD as either input or target has mask zero. A derived
257-position storage mask preserves compatibility with the existing
`PackedInstructionDataset`, which exposes its `[1:]` target-aligned view.

The pure packing implementation is `vasu.data.arithmetic_packing`. Record
indexes are immutable: resuming at a record index returns the same stored
tokens and mask independent of earlier records. This is not yet a production
multi-source serializer or a training authorization; tokenization, artifact
metadata, and hash-bound source scheduling remain readiness blockers.

## Verified arithmetic v1 canonical release

The canonical unique arithmetic artifact is generated by
`scripts/serialize_vasu_verified_arithmetic_v1.py` under
`data/processed/capability/verified_arithmetic_v1/`. It uses the authoritative
`assets/tokenizer.json` without changing its 32,000-token vocabulary or special
IDs (`PAD=0`, `BOS=2`, `EOS=3`). Logical examples use the single versioned
format `Question: {prompt}\nAnswer: {answer}`; the serializer appends exactly
one EOS token and never inserts logical-example PAD tokens.

The release preserves development and evaluation examples as logical JSONL
with stable IDs, prompts, exact answers, operation/difficulty labels, template
and generator versions, and operand metadata. Only training examples are
packed. The canonical record-count policy is one deterministic pass over the
80 unique training examples; replay is deferred to the future scheduled
mixture layer.

The validated v1 artifact contains 9 packed records, 2,081 real tokens, and 232
PAD tokens. Mean utilization is 0.899697 (minimum 0.529183, maximum 0.992218).
Its token binary is `uint16`. Its `uint8` mask binary stores 257 token-position
values per record for `PackedInstructionDataset`; `stored_mask[1:]` is the
effective 256-position target-aligned mask received by Trainer.

`manifest.json` binds generator, serializer, tokenizer, logical splits, and
every artifact by SHA-256. It records packing/masking policy, counts,
distributions, utilization, special IDs, and validation outcomes.
`training_authorized` remains `false`: this artifact is not a multi-source
schedule and does not authorize a CPT run.

Canonical artifact SHA-256 values:

- tokens:
  `341e575dab8a001ac30155cf674c5bd4b25d687048114d36a4578673ac72d141`;
- stored masks:
  `2ddf37b626e3d6c6b92a28f6a885b99c2d995524949fbca4db1424b1cf71789d`;
- packed provenance:
  `bf16600cbb289e0c17715967512297c8c9292e124732e3564ed48dc8787728c7`;
- development/evaluation:
  `da988af5dca386fe98d4a07f87a2eff6b14ef867853b0918dcac12aa52df306c`
  and
  `f10a8bf0e668f02c0b70cabb279dfd98353cbc2d8a7cb495bea25a640d318ceb`;
- tokenizer:
  `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`;
- generator configuration:
  `415b1f82f499909d8f69cee0ae1cf52353823e6bb8f5970202080959467fd16a`.

```powershell
.venv\Scripts\python.exe scripts\serialize_vasu_verified_arithmetic_v1.py `
  --tokenizer assets\tokenizer.json `
  --output-dir data\processed\capability\verified_arithmetic_v1

.venv\Scripts\python.exe scripts\serialize_vasu_verified_arithmetic_v1.py `
  --output-dir data\processed\capability\verified_arithmetic_v1 `
  --validate-only

.venv\Scripts\python.exe scripts\validate_vasu_verified_arithmetic_v1.py `
  --manifest data\processed\capability\verified_arithmetic_v1\manifest.json
```

## Candidate C authorization readiness

The deterministic A/B/C v2 schedules supersede the early v1 pilot artifact for
the proposed 20M-token ablation. Candidate C now has an opt-in production
runtime with exact schedule resume, successful-update validation every 100
steps, checkpoints every 200 steps, separate domain-best selection, bounded
retention, disk and thermal guards, and full arithmetic-v2 evaluation.

The next action is review—not training: commit the verified implementation,
complete a Candidate C authorization record, and change only Candidate C's
`training_authorized` boolean if approved. Evaluate C against the parent before
considering a separately authorized Candidate A. Candidate B remains blocked
until A's evidence and human review satisfy its preregistered gate.
