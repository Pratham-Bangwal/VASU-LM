# VASU-140M Family Proposal

Status: accepted as an implementation-readiness contract. The isolated model
configuration, family identity, and read-only construction preflight are
implemented. This does not modify VASU-60M, create a training configuration,
generate data, qualify runtime behavior, or authorize training.

## Decision question

Should VASU continue the existing 60M checkpoint family, or should a larger,
cleanly versioned family be prepared for a future controlled pretraining
program?

## Evidence and conclusion

The current VASU-60M architecture is a sound decoder-only baseline: pre-norm
RMSNorm, RoPE attention, SDPA, SwiGLU, tied input/output embeddings, and
explicit cache contracts are already validated. Current negative capability
results do not isolate an architectural defect, and CPU profiling did not
identify a loader change worth adopting. The appropriate next proposal is
therefore a capacity/context experiment, not an unmeasured attention or
normalization rewrite.

## Alternatives considered

| Family | Shape | Exact parameters | Assessment |
|---|---|---:|---|
| VASU-86M | dim 640, 10 layers, 10 heads, MLP 2560, context 256 | 86,029,440 | Lowest risk but too close to 60M to justify a new data/research program. |
| **VASU-140M** | **dim 768, 12 layers, 12 heads, MLP 3072, context 512** | **137,841,408** | Recommended: a meaningful capacity/context step while preserving conventional, validated components. |
| VASU-209M | dim 896, 14 layers, 14 heads, MLP 3584, context 512 | 208,528,768 | Deferred pending demonstrated VASU-140M feasibility and CUDA throughput/memory evidence. |

## Recommended family contract

```text
family_id: vasu_140m_v1
vocab_size: 32000
max_seq_len: 512
dim: 768
n_heads: 12
n_layers: 12
head_dim: 64
hidden_dim: 3072
dropout: 0.1
rope_theta: 10000.0
bias: false
weight_tying: true
attention: causal SDPA with RoPE
normalization: pre-norm RMSNorm
MLP: SwiGLU
```

The exact parameter count above is produced by the existing `VASUModel` with
this configuration. It retains 64-dimensional attention heads, avoiding a
new attention-layout variable.

## Resource envelope

Parameter tensors occupy about 525.8 MiB in FP32 or 262.9 MiB in BF16/FP16.
A conservative 16-byte-per-parameter optimizer/gradient/model planning budget
is about 2.05 GiB before activations, temporary kernels, allocator headroom,
or checkpoint serialization. A batch-one, 512-token BF16 KV cache is about
18 MiB; batch two is about 36 MiB. These are planning estimates, not a CUDA
feasibility claim. CUDA smoke measurements must establish actual peak memory,
throughput, thermal behavior, and checkpoint-I/O time before any training
proposal is considered.

## Compatibility boundary

| Artifact | Compatibility |
|---|---|
| VASU-60M checkpoints and optimizer states | Not loadable into VASU-140M; tensor shapes differ. Preserve them as a separate family. |
| Checkpoint container tooling | Reusable only after adding a family/config identity gate; never select a checkpoint by filename. |
| Tokenizer | Compatible: retain the existing 32k tokenizer unchanged and bind its SHA-256 in every new artifact. |
| Raw text sources | Potentially reusable subject to a new manifest and leakage review. |
| Existing 257-token binaries and masks | Not directly compatible with a 512-token context. Repack into isolated 513-token records and regenerate shifted masks without modifying old artifacts. |
| Inference and evaluation | Require an explicit family/config argument and new frozen baselines; do not compare base and instruction interfaces as equivalent. |
| Exact resume | Must be newly demonstrated for VASU-140M. VASU-60M resume evidence does not transfer automatically. |

## Required gates before training

1. The immutable family contract and non-executing readiness plan are complete.
   The future scientific hypothesis remains: increased capacity/context
   improves selected frozen capability evaluations at an acceptable compute
   cost.
2. The configuration is implemented as `vasu_140m_v1`; the VASU-31M default
   and opt-in VASU-60M configuration are unchanged.
3. Complete CPU and CUDA forward/backward smoke tests, cache
   parity tests, checkpoint serialization checks, and exact-resume tests.
4. Prepare new 513-token data and mask releases with deterministic rebuild,
   split isolation, token/mask alignment, and manifest hashes.
5. Establish frozen pretraining, factual, repetition, arithmetic, and
   robustness baselines before any model-quality claim.
6. Run the runtime benchmark contract on matched CUDA workloads and reject the
   family if memory, thermal, disk, or throughput gates fail.
7. Obtain a distinct scientific decision, immutable experiment plan, and
   hash-bound authorization record before any optimizer update.

See `VASU_140M_IMPLEMENTATION_READINESS.md` for the exact implemented identity
and the remaining fail-closed gates.

## Non-goals

This proposal does not introduce grouped-query attention, a vocabulary change,
new positional encoding, tokenizer retraining, checkpoint conversion, data
generation, or training. Those changes would confound the capacity/context
question and require separate proposals.
