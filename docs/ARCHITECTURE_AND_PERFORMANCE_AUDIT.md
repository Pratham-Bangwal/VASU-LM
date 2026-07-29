# Architecture and Performance Audit

Status: read-only assessment; no model architecture or checkpoint schema was
changed.

## Current architecture

VASU uses a decoder-only, pre-norm Transformer with RMSNorm, RoPE multi-head
self-attention, PyTorch scaled-dot-product attention, residual connections,
and SwiGLU feed-forward blocks. This is a sound, maintainable baseline for the
existing VASU-60M checkpoint family.

## Findings

1. Attention validates dimension/head compatibility, cache modes, and maximum
   sequence length before execution. This protects decode correctness.
2. Dynamic KV cache preserves parity and reduces peak allocation, but current
   benchmarks show no throughput gain. Concatenating K/V tensors on every
   decode step is the likely overhead.
3. A preallocated cache implementation already exists and is the first
   optimization candidate because it can preserve model weights, tokenizer,
   prompts, checkpoint keys, and generation semantics.
4. Changes such as grouped-query attention, altered hidden dimensions, layer
   count, vocabulary, normalization, or positional encoding would invalidate
   existing checkpoints. They require a separate model-family proposal rather
   than an in-place optimization.

## Recommended sequence

1. Benchmark preallocated KV cache against uncached and dynamic-cache paths on
   fixed CPU/CUDA prompts, preserving existing logit/token parity tests.
2. Profile data-loader and packed-mask transfer time separately from model
   forward/backward time before changing pipeline code.
3. Add inference presets and checkpoint inspection tools only when they retain
   existing tokenizer/checkpoint identity.
4. Treat any parameter-shape change as a new architecture version with fresh
   smoke, resume, evaluation, and compatibility documentation.

No recommendation in this audit authorizes training or modifies existing
checkpoint compatibility.
