# VASU Model Card

## Model family

VASU-LM is a family of decoder-only autoregressive Transformers implemented from scratch in PyTorch for education and small-scale language-model research.

| Model | Parameters | Current role | Current checkpoint |
| --- | ---: | --- | --- |
| VASU-31M | approximately 31.17M | Completed instruction-tuned fallback | `checkpoints/ultrachat_fineweb/best.pt` |
| VASU-60M | 58,337,792 | Preferred experimental assistant | `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt` |

VASU-60M has completed standard Alpaca, masked-Alpaca-v2, masked-Alpaca-v3, and UltraChat masked-v2 experiments. Alpaca v3 is preferred after expanded checkpoint comparison, but it is not a reliable or safety-aligned assistant. UltraChat masked v2 remains an experimental checkpoint and is not promoted.

## Architecture

Both models use causal self-attention, PyTorch scaled-dot-product attention, RoPE, RMSNorm, SwiGLU, pre-norm residual blocks, tied input/output embeddings, and bias-free linear layers. They share a 32,000-token tokenizer and 256-token context length.

VASU-31M uses dimension 384, eight layers, six heads, and a 1,536-wide MLP. VASU-60M uses dimension 512, ten layers, eight heads, and a 2,048-wide MLP.

## Training status

VASU-31M completed FineWeb pretraining and Alpaca, masked-Alpaca, and UltraChat experiments. Its manual fixed-prompt baseline is 2.225 / 5.

VASU-60M continued FineWeb pretraining to the preserved step-200,000 base checkpoint. Masked Alpaca v3 then completed one epoch at global step 200,711. UltraChat masked v2 continued experimentally to step 201,301. The expanded greedy and sampled comparison selected Alpaca v3 because it showed lower repetition and relatively stronger behavior in several task categories. Neither checkpoint is generally reliable, manually safety-aligned, or suitable as a factual authority.

## Intended use

- learning Transformer and language-model engineering;
- reproducible local pretraining and fine-tuning experiments;
- checkpoint, dataset, inference, and evaluation research;
- low-stakes exploration of small language models.

The preferred VASU-60M checkpoint is `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`. It is intended only for experimental, educational, and locally supervised use.

## Unsupported use

The models are not supported for production deployment, high-stakes decisions, autonomous actions, or medical, legal, financial, security, and safety-critical advice. Outputs require human verification.

## Language and capability limits

Training and evaluation are primarily English-oriented. Neither checkpoint provides reliable multilingual performance.

Observed limitations include:

- factual hallucination and limited factual knowledge;
- repetition and incomplete continuations;
- semantic and topic drift;
- weak long-range coherence and reasoning;
- exact-counting and formatting failures;
- short 256-token context;
- no completed safety alignment;
- no reliability-validated or safety-aligned VASU-60M assistant checkpoint.

Arithmetic, reasoning, exact formatting, uncertainty handling, and programming correctness remain weak. Sampled decoding can reduce visible repetition relative to greedy decoding, but it does not make answers more factual or correct. The model must not be treated as a dependable medical, legal, financial, scientific, programming, or general factual authority.

Improved grammar or lower validation loss does not establish factual reliability or assistant readiness.

## Hardware constraints

Development and training use an RTX 4050 Laptop GPU with 6 GB VRAM, Intel i5-13420H, 16 GB RAM, and Windows. VASU-60M requires AMP and small micro-batches. Laptop thermals require short resumable blocks, an 88°C stop, and cooldowns. Optimizer-bearing checkpoints also create significant disk pressure.

## Checkpoint guidance

- Use `checkpoints/ultrachat_fineweb/best.pt` for the current VASU-31M assistant experiment.
- Use `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt` as the latest preserved VASU-60M base checkpoint.
- Treat `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt` as the current controlled VASU-60M instruction experiment, not a production assistant.
- Preserve `checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt` as an experimental comparison; it is not the default chat checkpoint.
- Treat `checkpoints/vasu_60m/alpaca/best.pt` and `checkpoints/vasu_60m/alpaca_masked_v2/best.pt` as unpromoted research comparisons only.
- Construct the matching model configuration before loading.
- Never load VASU-31M weights into VASU-60M or vice versa; tensor shapes differ.
- Treat operational block checkpoints as resumable training state and milestone checkpoints as preserved evaluation references.

## Evaluation

VASU-31M uses fixed prompts and manual scores across relevance, factuality, instruction following, fluency, and repetition control. VASU-60M development used the earlier fixed eight-prompt suite and a later expanded 40-prompt comparison under both greedy and sampled decoding. The expanded comparison is useful for relative checkpoint selection but is not a comprehensive benchmark and does not establish correctness or safety.

The final relative comparison found lower average repetition for Alpaca v3 in both modes and relatively stronger results in reasoning, programming, planning, uncertainty, and instruction-following categories. UltraChat showed some additional conversational variation but weaker deterministic stability and several category regressions. Neither checkpoint is generally reliable.

Compatibility: checkpoint selection and documentation do not change the VASU-60M architecture, tokenizer, checkpoint schema, FineWeb artifacts, Alpaca artifacts, UltraChat artifacts, evaluation reports, or training runners.

## Ethical considerations

The training data may contain bias, errors, or harmful content. The models can reproduce those patterns and invent plausible-sounding claims. There is no completed safety-tuning or preference-optimization stage. Do not rely on generated text without independent review.
