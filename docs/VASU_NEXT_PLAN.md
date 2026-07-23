# VASU-60M capability CPT ablation plan

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
