# VASU-LM

VASU-LM (Virtual AI System for Understanding) is an educational, experimental framework for building decoder-only language models from scratch in PyTorch. The repository implements the model, tokenizer integration, data preparation, pretraining, instruction tuning, checkpointing, inference, and evaluation without relying on a pretrained Transformer model.

This is a learning and research project. It is not production-ready, safety-aligned, or suitable for high-stakes use.

## Current model families

### VASU-31M

VASU-31M completed its experimental cycle: FineWeb pretraining, Alpaca tuning, masked-Alpaca experiments, UltraChat tuning, and manual evaluation. It remains the stable instruction-tuned fallback.

- Parameters: approximately 31.17M
- Context length: 256 tokens
- Best assistant checkpoint: `checkpoints/ultrachat_fineweb/best.pt`
- Manual evaluation baseline: 2.225 / 5

### VASU-60M

VASU-60M is the active model and is still in base pretraining. It is not an instruction-tuned assistant.

| Setting | Value |
| --- | ---: |
| Parameters | 58,337,792 |
| Vocabulary size | 32,000 |
| Context length | 256 |
| Model dimension | 512 |
| Layers | 10 |
| Attention heads | 8 |
| SwiGLU hidden dimension | 2,048 |
| Dropout | 0.1 |
| RoPE theta | 10,000.0 |
| Linear bias | False |

Current preserved milestone:

`checkpoints/vasu_60m/milestones/fineweb_step_54060.pt`

- Global step: 54,060
- Train loss: 3.616769
- Validation loss: 3.613814
- Next target: 100,000 optimizer steps
- Instruction tuning: not started

## Architecture

Both model families use the same decoder-only autoregressive Transformer design:

- causal multi-head self-attention through PyTorch scaled-dot-product attention;
- rotary positional embeddings (RoPE);
- RMSNorm;
- SwiGLU feed-forward layers;
- pre-norm residual blocks;
- tied token-embedding and LM-head weights;
- bias-free linear layers;
- AMP-compatible training.

The tokenizer and processed token datasets are shared. Model checkpoints are not interchangeable between 31M and 60M because their tensor shapes differ.

## Training and evaluation

VASU-60M trains on FineWeb in resumable 100-optimizer-step blocks with batch size 2, gradient accumulation 16, sequence length 256, AMP, checkpoints every 10 optimizer steps, an 88°C thermal stop, atomic saves, corrupt-checkpoint filtering, bounded retention, and low-disk protection.

Base-model milestones are evaluated using raw autoregressive continuations. VASU-31M instruction checkpoints are compared with a fixed prompt suite and manual criteria covering relevance, factuality, instruction following, fluency, and repetition control.

## Hardware used

- NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM
- Intel Core i5-13420H
- 16 GB RAM
- Windows

## Current limitations

- factual hallucinations and limited factual knowledge;
- repetition loops and incomplete generations;
- semantic and topic drift;
- weak reasoning and exact-format instruction following;
- weak long-range coherence;
- short 256-token context;
- no completed VASU-60M instruction or safety tuning;
- laptop thermal and disk constraints during long training.

## Repository layout

```text
vasu/          Model, tokenizer, training, and inference modules
scripts/       Dataset preparation, diagnostics, and smoke tests
evaluation/    Fixed prompts, generated reports, and manual scores
docs/          Architecture, status, training history, and roadmap
checkpoints/   Local training state and preserved milestones (not for Git)
```

## Development status

Completed: VASU-31M experiment cycle, VASU-60M planning and smoke tests, resumable thermal-safe block training, atomic checkpointing, retention, and base milestone evaluation.

In progress: VASU-60M FineWeb pretraining from step 54,060 toward step 100,000.

Pending: step-100,000 evaluation and the decision gate for controlled VASU-60M instruction tuning.

See [Project Status](docs/PROJECT_STATUS.md), [Architecture](docs/ARCHITECTURE.md), [Training Log](docs/TRAINING_LOG.md), and [Roadmap](docs/ROADMAP.md).

## License and use

Review the repository license before redistribution. Generated output must be treated as untrusted: neither VASU-31M nor VASU-60M should be used for medical, legal, financial, safety-critical, or production decisions.
