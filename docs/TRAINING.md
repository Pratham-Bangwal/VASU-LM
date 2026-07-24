# Training Guide

## Capability-CPT v2 authorization gates

V2 candidates require `replay_safety`, `cuda_smoke`, and
`cuda_exact_resume` to be `passed`. Hash validation covers the parent,
tokenizer, sources, source manifests, masks, and schedule. These technical
gates passed, but `training_authorized` remains false, so launch is blocked.

## Unauthorized capability-CPT candidates

Validate a candidate without training:

```powershell
python train_vasu_60m_capability_cpt.py --config configs/training/capability_cpt_a_factual_20m_v1.json --validate-only
```

Omitting `--validate-only` stops at the authorization gate with
`Training is blocked because training_authorized is false.` Candidate configs
use sequential schedule order (`shuffle: false`), batch size 2, accumulation
16, context 256, and 2,442 complete optimizer groups over 20,004,864 tokens.
Generated schedules are ignored by Git.

> Complete guide to training models with VASU.

---

# Overview

VASU supports training GPT-style decoder-only language models from scratch using PyTorch.

The training pipeline is designed to be:

* Reproducible
* Modular
* GPU friendly
* Easy to experiment with
* Compatible with future scaling efforts

---

# Training Pipeline

```text id="gf7v21"
Dataset
    ↓
Tokenizer
    ↓
Binary Dataset
    ↓
DataLoader
    ↓
Forward Pass
    ↓
Loss Computation
    ↓
Backward Pass
    ↓
Gradient Scaling (AMP)
    ↓
Gradient Clipping
    ↓
Optimizer Step
    ↓
Scheduler Step
    ↓
Checkpoint
```

---

# Training Features

## Supported Features

* Automatic Mixed Precision (AMP)
* Gradient Accumulation
* Gradient Clipping
* Validation
* TensorBoard Logging
* Sample Generation
* Checkpoint Saving
* Checkpoint Resume

---

# VASU-31M Default Training Configuration

The values in this section describe the original `ModelConfig()` defaults and general trainer example. They remain unchanged for VASU-31M compatibility; they are not the active VASU-60M FineWeb recipe.

## Model Configuration

| Parameter         | Value  |
| ----------------- | ------ |
| Vocabulary Size   | 32,000 |
| Context Length    | 256    |
| Layers            | 8      |
| Hidden Size       | 384    |
| Attention Heads   | 6      |
| Feed Forward Size | 1536   |

---

## Training Configuration

| Parameter             | Value             |
| --------------------- | ----------------- |
| Batch Size            | 4                 |
| Epochs                | 20                |
| Learning Rate         | 3e-4              |
| Weight Decay          | 0.01              |
| Gradient Accumulation | 8                 |
| Gradient Clipping     | 1.0               |
| Mixed Precision       | Enabled           |
| Optimizer             | AdamW             |
| Scheduler             | CosineAnnealingLR |

## Active VASU-60M FineWeb recipe

| Parameter | Value |
| --- | ---: |
| Parameters | 58,337,792 |
| Batch size | 2 |
| Gradient accumulation | 16 |
| Sequence length | 256 |
| Learning rate | 1e-4 |
| Weight decay | 0.1 |
| AMP | Enabled |
| Optimizer steps per block | 100 |
| Save interval | 10 optimizer steps |
| Thermal stop | 88°C |

The authoritative VASU-60M base milestone is `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`. Base pretraining and masked Alpaca v3 instruction tuning are complete. The preferred assistant checkpoint is `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.

---

# Effective Batch Size

The effective batch size is:

```text id="yvxwfa"
effective_batch_size =
batch_size × gradient_accumulation_steps

= 4 × 8
= 32
```

---

# Datasets

## Pretraining

* TinyStories
* FineWeb

## Instruction Tuning

* Alpaca
* UltraChat

---

# Dataset Preparation

## Raw Text

```text id="17pq0z"
data/
└── raw/
```

## Processed Data

```text id="p71c8r"
data/
└── processed/
```

Processed datasets are stored in binary format for faster loading.

---

# Training from Scratch

Run:

```bash id="z1ujq4"
python train.py
```

---

# Training Output

Example:

```text id="l1r5ls"
Using device: cuda
Loading tokenizer...
Loading dataset...
Building model...
Starting training...
```

---

# Mixed Precision Training (AMP)

VASU uses Automatic Mixed Precision to:

* Reduce VRAM usage
* Increase training speed
* Enable larger models on consumer GPUs

---

# Gradient Accumulation

Gradient accumulation allows VASU to simulate larger batch sizes while training on limited hardware.

```text id="n8nqfw"
batch_size = 4
gradient_accumulation_steps = 8
effective_batch_size = 32
```

---

# Gradient Clipping

Gradients are clipped to:

```text id="ktvc4y"
max_norm = 1.0
```

Benefits:

* Prevents exploding gradients
* Improves training stability

---

# Optimizer

VASU uses:

```text id="rrjkr6"
AdamW
```

Configuration:

```text id="m9p9bw"
learning_rate = 3e-4
weight_decay = 0.01
```

---

# Learning Rate Scheduler

VASU uses:

```text id="kfw58o"
CosineAnnealingLR
```

Benefits:

* Smooth learning rate decay
* Better convergence
* Improved stability

---

# Validation

Validation is performed after every epoch.

Metrics:

* Validation Loss
* Sample Generation
* Best Model Tracking

---

# TensorBoard

Launch TensorBoard:

```bash id="j8r2th"
tensorboard --logdir runs
```

Open:

```text id="v7x3j8"
http://localhost:6006
```

---

# Checkpoints

VASU saves:

```text id="34tyof"
checkpoints/
├── best.pt
├── epoch_1.pt
├── epoch_2.pt
└── vasu.pt
```

---

# Resuming Training

Training automatically resumes if:

```text id="o7xt1v"
checkpoints/vasu.pt
```

exists.

No additional command is required.

## Exact mid-epoch resume

The general `Trainer` supports deterministic shuffled DataLoader resumes. It
stores the next batch after every completed backward pass, plus accumulation,
gradient, AMP-scaler, and RNG state when a checkpoint is written. A checkpoint
taken between optimizer updates can therefore continue without replaying or
dropping samples.

Resume phase is explicit rather than inferred from a completed sampler: a
checkpoint after the final training batch but before validation completes that
validation once, while best/epoch/latest checkpoints begin at the next epoch.

`global_step` counts successful optimizer updates only. If AMP/non-finite
gradient handling skips an update, gradients are cleared as a handled failure,
the step counter does not advance, and the per-epoch scheduler is not advanced
when no successful update occurred in that epoch. The default cosine scheduler
steps once per epoch (`T_max=epochs`); resuming with a larger epoch target keeps
the checkpoint's existing scheduler horizon and emits a warning.

## AdamW execution backends

`TrainConfig.optimizer_backend` accepts `standard` (the default), `foreach`,
`fused`, and `auto`. The default preserves historical checkpoint behavior.
`fused` is an explicit CUDA-only opt-in; its optimizer state remains regular
AdamW state and loads through the normal checkpoint path. Unsupported fused
requests fail clearly instead of silently selecting a different backend.

The guarantee requires an unchanged loader contract and deterministic dataset
behavior. Existing checkpoints without `training_progress` are still valid but
use the historical next-epoch resume with an explicit warning. Specialized
VASU-60M operational runners retain their existing sequential
global-step-to-data-offset behavior and are not retroactively converted.

---

# Best Model

The model with the lowest validation loss is saved as:

```text id="c0lh06"
checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt
```

---

# Sample Generation

During training, VASU can periodically generate text samples.

Benefits:

* Monitor quality improvements
* Detect training issues
* Observe emergence of language capabilities

---

# Hardware Requirements

## Minimum

* GPU: 4 GB VRAM
* RAM: 8 GB
* Python 3.10+

## Recommended

* GPU: 6 GB+ VRAM
* RAM: 16 GB+
* CUDA-enabled GPU

Current development hardware:

* NVIDIA RTX 4050 Laptop GPU
* 6 GB VRAM

---

# Training Workflow

```text id="8wop17"
Prepare Dataset
        ↓
Train Tokenizer
        ↓
Create Binary Dataset
        ↓
Train Model
        ↓
Validate
        ↓
Save Checkpoint
        ↓
Generate Samples
        ↓
Resume or Continue Training
```

---

# Common Issues

## CUDA Out of Memory

Possible solutions:

* Reduce batch size.
* Reduce context length.
* Reduce model size.
* Increase gradient accumulation.

---

## Loss Becomes NaN

Possible causes:

* Learning rate too high
* Dataset corruption
* Numerical instability

---

## Slow Training

Possible causes:

* CPU bottleneck
* Small batch size
* Disabled AMP

## Bounded pipeline audit (RTX 4050, VASU-60M)

`scripts/profiling/profile_data_pipeline.py` profiles local memmap-backed
FineWeb, instruction, and factual-mixture data without changing a dataset or
checkpoint. In a bounded batch-size-2, sequence-length-256 run, VASU-60M was
compute-bound: FineWeb loading accounted for under 0.6% of the median
end-to-end microbatch time. Windows worker processes improved raw loader
throughput but did not produce a material end-to-end gain, so the safe
worker-zero configuration remains preferred.

Validation in the general `Trainer` uses `torch.inference_mode()` rather than
`torch.no_grad()`. This preserves validation loss while reducing bounded
validation overhead; it does not affect model, optimizer, scheduler, sampler,
or checkpoint state.

---

# Reproducibility

To improve reproducibility:

* Set random seeds.
* Save configuration files.
* Save tokenizer versions.
* Keep dataset versions documented.

---

# Future Improvements

* Distributed training
* Multi-GPU support
* LoRA fine-tuning
* Gradient checkpointing
* Better experiment tracking
* Training benchmarks
* Dataset versioning

---

# Philosophy

Training in VASU is designed around a simple principle:

> Make experiments easy to reproduce, easy to understand, and easy to extend.
