# VASU-LM Project Overview

## Mission

VASU-LM is an educational and experimental project for understanding and implementing a modern decoder-only language model end to end in PyTorch. The goal is transparent engineering and measured iteration, not production deployment claims.

## Model families

### VASU-31M

- Approximately 31.17M parameters.
- Completed FineWeb pretraining and Alpaca, masked-Alpaca, and UltraChat experiments.
- Stable assistant checkpoint: `checkpoints/ultrachat_fineweb/best.pt`.
- Manual evaluation baseline: 2.225 / 5.

### VASU-60M

- 58,337,792 parameters.
- Base pretraining completed through global step 200,000.
- Authoritative base milestone: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Masked Alpaca v3 instruction tuning is complete.
- Preferred assistant checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.
- UltraChat masked v2 remains an experimental, non-promoted branch.

## Architecture summary

Both models use a decoder-only autoregressive Transformer with causal scaled-dot-product attention, RoPE, RMSNorm, SwiGLU, pre-norm residual blocks, tied embedding/output weights, and bias-free linear layers. Both use a 32,000-token tokenizer and context length 256.

VASU-31M uses dimension 384, eight layers, six heads, and a 1,536-wide MLP. VASU-60M uses dimension 512, ten layers, eight heads, and a 2,048-wide MLP.

## Training stages

```text
Tokenizer and processed datasets
    -> base pretraining
    -> base checkpoint evaluation
    -> instruction-tuning decision gate
    -> Alpaca (when authorized)
    -> evaluation
    -> UltraChat (when justified)
    -> fixed-prompt manual comparison
```

VASU-31M completed its experimental sequence. VASU-60M completed base pretraining and masked Alpaca v3 instruction tuning; further capability-focused continuation remains gated by controlled data-mixture evaluation.

## Evaluation approach

Base checkpoints use raw autoregressive continuation prompts with fixed generation settings. Instruction checkpoints use a shared prompt suite and manual scoring for relevance, factuality, instruction following, fluency, and repetition control. Validation loss is tracked as a trend but is not treated as a complete measure of assistant quality.

## Repository structure

```text
vasu/          Core model, tokenizer, training, and inference modules
scripts/       Data preparation, diagnostics, and smoke tests
evaluation/    Prompts, reports, milestone metadata, and manual scores
docs/          Architecture, status, experiments, and operational history
checkpoints/   Local resumable state and manually preserved milestones
```

## Current status

Completed: VASU-31M experiment cycle; VASU-60M base pretraining through step 200,000; masked Alpaca v3 instruction tuning; checkpoint evaluation; recovery, deduplication, and instruction-quality data pipelines.

In progress: provenance-complete factual-pilot review and controlled capability-mixture planning.

Pending: factual-pilot authorization, short mixture ablations, and any larger continued-pretraining run.

## Major risks

- factual hallucination, repetition, semantic drift, and weak reasoning;
- short 256-token context;
- limited VASU-60M safety and capability coverage despite completed instruction tuning;
- laptop thermal constraints;
- large optimizer checkpoint disk usage;
- incomplete general DataLoader sampler-state restoration.

## Contribution philosophy

- Preserve backward compatibility and known-good checkpoints.
- Make small, reviewable changes.
- Separate completed, in-progress, and planned work.
- Measure before promoting a model or training stage.
- Never treat fluent output as proof of factuality or safety.
- Keep operational safeguards enabled on consumer hardware.
