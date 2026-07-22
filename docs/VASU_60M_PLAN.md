# VASU-60M Plan and Implementation Status

## Objective

VASU-60M increases capacity without changing the established decoder-only architecture, tokenizer, processed datasets, or checkpoint container schema. It is designed for local experimentation on an RTX 4050 Laptop GPU with 6 GB VRAM.

## Selected configuration

| Setting | Value |
| --- | ---: |
| Parameters | 58,337,792 |
| Vocabulary size | 32,000 |
| Maximum sequence length | 256 |
| Model dimension | 512 |
| Layers | 10 |
| Attention heads | 8 |
| Head dimension | 64 |
| SwiGLU hidden dimension | 2,048 |
| Dropout | 0.1 |
| RoPE theta | 10,000.0 |
| Linear bias | False |

The selected model uses the existing decoder-only Transformer design: causal PyTorch scaled-dot-product attention, RoPE, RMSNorm, SwiGLU, pre-norm residual blocks, tied token/LM-head weights, and no linear biases.

## Implementation status

### Completed

- opt-in VASU-60M configuration;
- exact parameter-count validation at 58,337,792;
- CPU construction and forward smoke test;
- CUDA forward/backward training smoke test;
- batch size 2, sequence length 256 AMP test;
- tiny real-data training and validation;
- checkpoint save and resume test;
- thermal-safe resumable block training;
- atomic checkpoint saving;
- corrupt/incomplete checkpoint filtering;
- bounded retention and low-disk protection;
- preserved base milestones and raw continuation evaluation;
- step-54,060 milestone.

Current milestone:

`checkpoints/vasu_60m/milestones/fineweb_step_54060.pt`

- Train loss: 3.616769
- Validation loss: 3.613814
- Maximum reported GPU temperature: 74°C
- Peak allocated CUDA memory: 1,376.37 MiB

### In progress

- FineWeb base pretraining from global step 54,060 toward step 100,000.

### Pending

- preserve and evaluate the step-100,000 checkpoint;
- decide whether the base model is ready for instruction tuning or needs more pretraining;
- controlled VASU-60M Alpaca tuning;
- controlled VASU-60M UltraChat tuning;
- fixed-prompt comparison against the VASU-31M manual baseline of 2.225 / 5.

VASU-60M instruction tuning has not started.

## Current training recipe

| Setting | Value |
| --- | ---: |
| Batch size | 2 |
| Gradient accumulation | 16 |
| Effective batch | 32 sequences |
| Sequence length | 256 |
| Effective tokens per optimizer step | 8,192 |
| Learning rate | 1e-4 |
| Weight decay | 0.1 |
| AMP | Enabled |
| Optimizer steps per block | 100 |
| Save interval | 10 optimizer steps |
| Thermal stop | 88°C |

The long runner permits up to 11 hours, waits 10 seconds after normal blocks and 10 minutes after thermal stops, and has a hard target at global step 100,000.

## Dataset strategy

1. Complete the current FineWeb base-pretraining decision point.
2. Evaluate the preserved step-100,000 checkpoint with raw continuations.
3. If the base model passes the decision gate, tune on Alpaca in a separate checkpoint directory.
4. Evaluate before any UltraChat stage.
5. If justified, tune on UltraChat and compare against VASU-31M using the same manual rubric.

The existing `assets/tokenizer.json` and processed FineWeb, Alpaca, and UltraChat files remain compatible. Model width and depth do not change token IDs or response-mask alignment.

## Compatibility

VASU-31M checkpoints remain valid for the original VASU-31M configuration but cannot be loaded into VASU-60M because tensor shapes differ. The tokenizer, token datasets, response masks, and checkpoint container schema remain compatible at the tooling level.

Changes to vocabulary size, dimensions, layer count, projection layout, MLP width, bias settings, parameter names, or weight tying would require a new compatibility review.

## Evaluation gate

The stable VASU-31M assistant checkpoint is `checkpoints/ultrachat_fineweb/best.pt`, with a manual baseline of 2.225 / 5.

After VASU-60M instruction tuning eventually occurs, minimum success requires exceeding 2.225 / 5 on the same prompts and criteria. A target near 2.75 / 5 is reasonable, but no VASU-60M manual instruction score exists yet.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Thermal shutdown or throttling | Short blocks, 88°C stop, cooldowns, ventilation, and manual monitoring. |
| Disk exhaustion | Minimum-free-space check, bounded operational retention, separate milestone directory. |
| Incomplete checkpoint | Temporary-file serialization, atomic replacement, validation before resume. |
| Checkpoint confusion | Dedicated VASU-60M directories and explicit matching configuration. |
| Repeated/skipped data after resume | Block slice advances from internal `global_step`; general sampler state remains an open issue. |
| Weak factuality | Continue base evaluation and use curated instruction data; do not infer reliability from loss alone. |
| Overfitting during future tuning | Separate checkpoints, short controlled runs, evaluation after every stage. |

## Decision

Continue the existing controlled FineWeb pretraining path to step 100,000. Do not begin Alpaca or UltraChat until that milestone is preserved and evaluated. Keep VASU-31M as the stable assistant fallback throughout this process.
