# VASU-LM Project Context

## Mission

Build and study decoder-only language models from scratch in PyTorch, with transparent architecture, local training, safe checkpoint operations, and reproducible evaluation.

## Current models

### VASU-31M

- approximately 31.17M parameters;
- completed pretraining and instruction-tuning experiment cycle;
- stable assistant checkpoint: `checkpoints/ultrachat_fineweb/best.pt`;
- manual baseline: 2.225 / 5.

### VASU-60M

- 58,337,792 parameters;
- vocabulary 32,000; context 256; dimension 512; ten layers; eight heads; hidden dimension 2,048;
- base pretraining completed through global step 200,000;
- authoritative base milestone: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`;
- masked Alpaca v3 instruction tuning completed;
- preferred assistant checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`;
- UltraChat masked v2 remains experimental and is not promoted.

## Shared design

Decoder-only causal Transformer, PyTorch scaled-dot-product attention, RoPE, RMSNorm, SwiGLU, pre-norm residuals, tied token/LM-head weights, and no linear biases.

## Hardware

- RTX 4050 Laptop GPU, 6 GB VRAM
- Intel i5-13420H
- 16 GB RAM
- Windows

## Current priority

Complete the factual-pilot review gate, validate capability-focused mixtures through short controlled ablations, and authorize further VASU-60M continuation only when evaluation shows measurable improvement without unacceptable regression.

## Constraints

Keep VASU-31M defaults and checkpoints intact. Do not mix 31M and 60M model states. Reuse the tokenizer and processed datasets. Preserve thermal, disk, atomic-save, corruption-filtering, and retention safeguards.
