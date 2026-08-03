# VASU Architecture

## Capability-CPT production orchestration

Candidate C uses the unchanged VASU-60M model and checkpoint schema through an
opt-in `CapabilityTrainer` subclass. The separation is deliberate: model,
tokenizer, scheduled dataset, loss, optimizer, and general Trainer semantics
remain unchanged, while capability-specific orchestration owns multi-domain
validation, domain-best selection, disk/thermal guards, checkpoint retention,
and abort reporting.

Its compact checkpoint metadata binds the experiment, parent, tokenizer,
source/schedule hashes, validation configuration, optimizer/scheduler, batch,
accumulation, and sequence settings. The saved capability state records
completed validation events, best-domain metrics, optimizer skips, retention,
thermal history, and abort reason. This additive metadata preserves the
existing tensor/container format and prevents a checkpoint from resuming under
a scientifically different validation or data configuration.

## Overview

VASU is a decoder-only autoregressive Transformer implemented from scratch in
PyTorch. VASU-31M, VASU-60M, and the implementation-readiness-only VASU-140M-v1
family share the same module design and tokenizer while using different tensor
dimensions. VASU-140M-v1 is not training-authorized.

## Model configurations

| Setting | VASU-31M | VASU-60M | VASU-140M-v1 |
| --- | ---: | ---: | ---: |
| Vocabulary size | 32,000 | 32,000 | 32,000 |
| Maximum sequence length | 256 | 256 | 512 |
| Model dimension | 384 | 512 | 768 |
| Layers | 8 | 10 | 12 |
| Attention heads | 6 | 8 | 12 |
| Head dimension | 64 | 64 | 64 |
| SwiGLU hidden dimension | 1,536 | 2,048 | 3,072 |
| Dropout | 0.1 | 0.1 | 0.1 |
| RoPE theta | 10,000.0 | 10,000.0 | 10,000.0 |
| Linear bias | False | False | False |
| Parameter count | 31,168,896 | 58,337,792 | 137,841,408 |

The VASU-31M default configuration remains unchanged. VASU-60M and VASU-140M
are selected through separate opt-in configuration factories.

The immutable `vasu.model.families` registry binds each versioned family ID to
all `ModelConfig` fields, an exact expected parameter count, a canonical
configuration SHA-256, and a family SHA-256. Callers receive a fresh config;
mutating it cannot change the registry. Declaring one family with another
family's config fails closed before model or checkpoint work.

## Shared architectural features

### Autoregressive decoder

The model predicts each next token from earlier tokens. A causal mask prevents attention to future positions.

### Token embeddings and weight tying

Token IDs map to learned embeddings. The token-embedding matrix is tied to the language-model output head, reducing parameters and keeping input and output token representations coupled.

### Pre-norm Transformer blocks

Each block applies RMSNorm before its attention and feed-forward sublayers, with residual connections around both sublayers:

```text
x = x + causal_attention(rmsnorm(x))
x = x + swiglu_mlp(rmsnorm(x))
```

### Causal self-attention

Attention uses PyTorch scaled-dot-product attention with causal behavior.
VASU-31M uses six 64-dimensional heads, VASU-60M uses eight, and VASU-140M-v1
uses twelve.

### Rotary positional embeddings

RoPE encodes relative position in query and key vectors without learned absolute-position embeddings. Both configurations use `rope_theta = 10000.0`.

### RMSNorm and SwiGLU

RMSNorm provides normalization without mean subtraction. SwiGLU supplies the gated feed-forward transformation. Linear layers do not use biases.

### AMP compatibility

The model and losses support automatic mixed-precision training on CUDA. AMP changes numeric execution, not checkpoint tensor structure.

## Inference KV cache

Uncached full-sequence generation remains the reference implementation and the default. The optional inference-only `KVCache` stores each attention layer's key/value tensors outside the model, so it has no parameters, never enters `state_dict()`, and does not alter checkpoints or training forwards.

Cached inference has two explicit stages because their masks differ:

- **Prefill:** the complete formatted prompt is processed once with normal causal attention. RoPE positions begin at zero, every layer writes prompt K/V into an empty cache, and prompt logits match the uncached full-prompt forward.
- **Decode:** exactly one newest token is processed. Its RoPE position equals the cached sequence length, its K/V are appended to the layer cache, and the one-token query may attend to every cached and current key because no future key exists.

Invalid transitions fail explicitly: prefill cannot use a populated cache, decode cannot use an empty cache or a multi-token query, cache-enabled modes require layer indices, and prompt-plus-generation length cannot exceed `max_seq_len`. Cache tensors are validated for layer, shape, batch size, head count, head dimension, device, dtype, synchronized sequence length, and the model context limit. Separate generation calls always create separate caches.

Two opt-in implementations are available. The dynamic cache concatenates each
new key/value tensor. The preallocated cache writes into fixed-capacity storage
and exposes populated views. Preallocated v2.1 uses an ordered layer-update
state machine, making synchronization validation O(1) per layer and rejecting
partially completed or out-of-order model forwards. Full prompt-plus-generated
token history remains available to repetition penalty and sampling even though
decode forwards receive only one token.

### KV-cache benchmark

Preferred checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`; CUDA; greedy decoding; three fixed prompts. Every cached generated token ID matched the uncached reference.

| Output limit | Uncached average | Cached average | Cached prefill | Cached decode | Uncached peak | Cached peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 tokens | 121.81 tok/s | 116.09 tok/s | 0.0080 s | 0.2459 s | 243.33 MiB | 234.94 MiB |
| 100 tokens | 120.94 tok/s | 119.87 tok/s | 0.0086 s | 0.7546 s | 258.86 MiB | 236.89 MiB |

Dynamic caching reduced measured peak allocation but did not improve average throughput in these short, batch-one tests. The 30-token run was about 4.9% slower and the longer run about 0.9% slower. Correctness parity is established; optimization remains future work.

A later v2.1 CPU diagnostic removed per-layer length scans and redundant old
cache views. Five interleaved synthetic VASU-60M trials at 128 tokens measured
58.1873 tok/s baseline median and 58.1999 tok/s optimized median (+0.02%); the
mean improvement was 1.24%. This is below the existing 10% promotion threshold,
so cached generation remains opt-in and uncached generation remains the default.

Cached generation also uses a fixed-capacity token-history tensor. Each sampled
or greedy token is written in place, while repetition penalty and sampling see
the exact populated prefix. This avoids reallocating and copying the full token
history on every decode step without changing generated-token semantics.

## Training structure

The base objective is causal next-token prediction. Inputs and targets are the same token sequence shifted by one position. Instruction datasets can additionally provide response masks so that loss is applied only to selected target tokens.

VASU-60M base pretraining currently uses sequence length 256, batch size 2, gradient accumulation 16, and AMP. Operational block scripts add thermal, disk, checkpoint-integrity, and retention safeguards without changing the model architecture.

## Compatibility

### Checkpoints

VASU-31M checkpoints cannot be loaded into VASU-60M because model dimension, layer count, head count, normalization vectors, attention projections, and MLP tensor shapes differ. A shared checkpoint container schema does not make model weights shape-compatible.

The checkpoint container remains compatible at the tooling level and stores the same categories of state, including model, optimizer, scheduler, epoch, loss, and global step.

KV cache is ephemeral inference state. It is not registered as a module, buffer, or parameter and therefore does not change the checkpoint container or model-state keys.

### Tokenizer and datasets

All families use the same 32,000-token tokenizer, so token IDs remain
compatible. Existing 257-token processed records and response masks remain
compatible with VASU-31M/60M but do not satisfy VASU-140M-v1's proposed
512-token training contract. Any future 140M release must use isolated
513-token records and newly aligned shifted-target masks.

The additive VASU-140M record contract stores `uint16[513]` tokens and
`uint8[513]` masks. It derives training tensors as `x=tokens[:-1]`,
`y=tokens[1:]`, and `loss_mask=stored_mask[1:]`. Complete examples are packed
independently within train, development, and evaluation splits. Each example
begins with stored mask zero, preventing supervision of synthetic
EOS-to-next-prompt transitions. The module has no production source discovery
or training entry point.

### Changes that break direct loading

Changing vocabulary size, model dimension, layer count, projection layout, MLP hidden dimension, bias settings, parameter names, or weight-tying behavior breaks or changes direct checkpoint loading. Any future architectural change requires an explicit compatibility review.

## FineWeb logical data manifest

The VASU model architecture and checkpoint schema are unchanged by the FineWeb extension. `data/processed/pretrain/fineweb_manifest.json` describes data layout only:

- original training: `fineweb_1m.bin[0:1245891252]`;
- extension training: `fineweb_extension_500m.bin[0:500000478]`;
- fixed validation: `fineweb_1m.bin[1245891252:1271317605]`.

Logical training offsets consume the remainder of the original training region before entering extension offset zero. They never enter the original validation slice. At the established 8,192 tokens per optimizer step, step 150,000 maps to original token 1,228,800,000 and step 200,000 maps to extension token 392,508,748.

Both shards use the same 32,000-token tokenizer, `uint16` IDs, and exact per-document separator text `\n[EOS]\n`.

`ManifestTokenDataset` is the additive multi-shard reader for this layout. It keeps physical file slices separate from cumulative logical offsets, opens each shard as a read-only lazy NumPy memmap, and uses binary-search lookup for reads that may cross any number of shard boundaries. A 256-token sample reads 257 logical tokens and returns the standard one-token-shifted `x` and `y` tensors. The original `TextDataset` remains unchanged for existing single-file and instruction workflows.

The specialized FineWeb block runner derives its logical start from `global_step * batch_size * gradient_accumulation_steps * sequence_length` and uses `shuffle=False`. Validation is read only from the manifest's fixed original validation slice. The manifest-aware loader changes data addressing only; it does not change model architecture, optimizer settings, tokenizer IDs, checkpoint schema, or either binary shard.

## Bounded continuation orchestration

The step-200,000 continuation retains the established 200-optimizer-step Python execution size because that is the already validated thermal/checkpoint unit; the wrapper bounds repetition instead of turning it into one unrestricted process. A shorter final execution is allowed only when required by the hard ceiling. The PowerShell orchestrator launches one child process at a time and validates every resume and result checkpoint on CPU. Validation requires the established model, optimizer, scheduler, and `global_step` state, a strict VASU-60M tensor-shape match, finite stored tensors, and a recorded SHA-256 hash.

Operational safeguards remain outside the model architecture: atomic checkpoint promotion, corrupt/incomplete checkpoint rejection, bounded retention, a 10 GiB free-disk floor, an 88°C thermal stop, exact positive progress checks, per-block logs, and a JSONL run summary. A thermal stop ends orchestration by default. Milestone files are preserved outside operational retention and are never selected merely from an unverified filename.

The non-executing readiness command is `powershell -ExecutionPolicy Bypass -File .\run_vasu_60m_to_200k.ps1 -DryRun`. The explicitly authorized execution command is the same invocation without `-DryRun`. Both commands inspect internal checkpoint state rather than trusting filenames; once step 200,000 is reached, dry-run mode reports the target and launches no trainer.

## Masked Alpaca v3 replication boundary

`train_vasu_60m_alpaca_masked_v3.py` is an isolated controlled replication of the proven masked-Alpaca-v2 path. It changes only the initialization checkpoint to the authoritative evaluated FineWeb step-200,000 milestone and redirects every operational, resumable, and best checkpoint beneath `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/`.

The v2 fixed-record data remains unchanged: each stored record has 257 tokens, the model consumes 256-token inputs and shifted targets, and `mask[1:]` aligns supervision with `y=tokens[1:]`. Prompt and padding targets remain masked, while assistant response and terminating EOS targets remain supervised. Architecture, tokenizer, optimizer, scheduler, AMP, split, seed, checkpoint payload schema, and one-epoch limit are identical to v2. The v3 runner adds preflight validation and a forward-only `--dry-run`; these safeguards do not change training semantics.

## Deterministic pretraining mixtures

`vasu.data.pretraining_mixture` adds a trainer-independent preparation layer
for small continued-pretraining experiments. It validates immutable source
paths and hashes, allocates an exact number of 256-token sequences by configured
weights, and shuffles only the source-ID schedule with an isolated local random
generator.

The prepared artifact stores independent 257-token `uint16` records. Each
record produces one 256-token input and its shifted target, so switching
sources never creates a synthetic next-token target across source boundaries.
`FixedRecordTokenDataset` memory-maps these records and resumes by record index.
The schedule and binary each have an immutable SHA-256. Existing continuous
`TextDataset`, `ManifestTokenDataset`, model architecture, tokenizer IDs, and
checkpoint schema remain unchanged.

## Instruction-quality data pipeline

`vasu.data.instruction_quality` is a trainer-independent pipeline for the
planned `vasu_instruction_quality_v1` dataset. Source examples use a strict,
versioned JSONL schema with seven capability labels, explicit provenance and
licensing, factual-verification metadata, declared output constraints, and
stable example IDs. Human decisions live in a separate JSONL and bind to the
canonical SHA-256 of the source record, so a source edit makes its decision
stale instead of silently carrying approval forward.

Exact duplicate detection compares normalized instruction, input, response,
instruction-plus-input, and full example content. Near-duplicate candidates
use deterministic word-trigram Jaccard similarity with the configuration's
0.85 threshold. Candidates are reported for review; they are never silently
deleted. Approved data uses a stable SHA-256 ordering seeded with 42 and
stratified by capability for a configured 95/5 train/validation split. Any
exact or high-confidence near-duplicate crossing the split blocks release.

Release packing reuses the masked Alpaca-v3 data path and exact prompt:
`User: {instruction}\n{optional input}\nAssistant:` followed by one leading
space before the response. Prompt/header targets are masked with zero;
assistant response and terminating EOS targets are supervised with one; and
padding targets are masked with zero. Train and validation are packed
separately, and any truncated approved response blocks release. This data
pipeline changes no model, tokenizer, checkpoint, or trainer behavior.

## Deterministic mid-epoch resume

The general `Trainer` now uses `ResumableBatchSampler` for shuffled training.
Each epoch permutation is regenerated from a fixed seed and epoch number; the
checkpoint stores only the loader contract, seed, epoch, and next unconsumed
batch index rather than a full sample-order list. The trainer advances that
position after backward processing, so DataLoader prefetching cannot cause a
saved position to skip untrained samples.

New `training_progress` checkpoint metadata also preserves partial gradients,
gradient-accumulation position, AMP scaler state, and Python/NumPy/PyTorch/CUDA
RNG states. It is additive: model tensor keys, tokenizer IDs, processed
datasets, and existing checkpoint fields remain unchanged. Exact parameter
reproducibility requires deterministic dataset/model execution; worker-side
random transforms must be stateless or deterministic and
`persistent_workers=True` is rejected.

DataLoader iterator construction consumes a base seed even with zero workers.
Training and validation loaders therefore use dedicated deterministically
seeded generators, isolating loader bookkeeping from the global PyTorch RNG
used by dropout and other stochastic model operations. This prevents a newly
constructed resume iterator from shifting model randomness. Worker-side
random transforms must still be stateless or deterministic.
