# VASU-140M Implementation Readiness

Status: configuration, identity, bounded CPU execution, and model-only
checkpoint contracts complete; CUDA, exact resume, data, experiment, and
training gates remain closed.

## Why this layer exists

The VASU-140M proposal changes model width, depth, and context length. Treating
it as another informal `ModelConfig` literal would allow config drift,
ambiguous checkpoint selection, and accidental reuse of incompatible 60M
artifacts. The additive family layer gives the proposed model a strict,
versioned identity without changing any existing persistent interface.

## Implemented contract

`vasu_140m_v1` is bound to:

| Field | Value |
|---|---:|
| Vocabulary | 32,000 |
| Context | 512 |
| Width | 768 |
| Heads | 12 |
| Layers | 12 |
| SwiGLU hidden width | 3,072 |
| Dropout | 0.1 |
| RoPE theta | 10,000.0 |
| Linear bias | false |
| Exact parameters | 137,841,408 |
| Config SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Family SHA-256 | `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b` |

The config digest covers every `ModelConfig` field in canonical JSON. The
family digest additionally covers identity schema
`vasu.model-family-identity.v1` and the exact family ID. Config validation
rejects non-positive dimensions, non-divisible attention shapes, invalid
dropout/RoPE values, and non-boolean bias.

The registry also records the unchanged `vasu_31m_v1` and `vasu_60m_v1`
contracts. It exposes immutable family entries and returns fresh mutable
configs, preventing caller mutation from changing a registered identity.

## Read-only preflight

Run:

```powershell
python scripts/preflight_vasu_140m.py
```

The command creates the model on PyTorch's meta device, so it allocates no
parameter storage and performs no forward, backward, optimizer, dataset,
checkpoint, or training operation. It verifies:

- exact config and family identity;
- exact parameter count;
- 64-dimensional attention heads;
- tied token-embedding/output weights;
- complete meta-device construction.

Its JSON always reports `training_authorized: false` and lists every remaining
gate.

## Compatibility

- Existing VASU-31M and VASU-60M factories and defaults are unchanged.
- Existing checkpoint state keys and checkpoint containers are unchanged.
- Existing checkpoints remain loadable by their matching configurations.
- No VASU-31M/60M checkpoint or optimizer state is loadable into VASU-140M.
- The existing 32k tokenizer and raw token IDs remain compatible.
- Existing 257-token records and masks remain valid for current workflows but
  are not a 512-context VASU-140M release.
- CPU FP32 exact-resume equivalence is qualified for the frozen synthetic
  VASU-140M workload; CUDA/AMP and real-data execution remain unqualified.

## Remaining fail-closed gates

The bounded FP32 CPU forward/backward and cache-parity gate passed on
2026-07-30. See `VASU_140M_CPU_QUALIFICATION_20260730.md`.
The additive model-only checkpoint round trip and explicit wrong-family
rejection gate also passed. See
`VASU_140M_CHECKPOINT_QUALIFICATION_20260730.md`.
The corrected synthetic exact-resume v2 gate passed after isolating DataLoader
iterator RNG from model RNG. See
`VASU_140M_EXACT_RESUME_QUALIFICATION_20260730.md`.

1. Matched CUDA memory, throughput, thermal, and checkpoint-I/O evidence.
2. Independently reviewed, deterministic 513-token data and shifted-mask
   releases with hashes and split isolation.
3. Frozen pretraining, factual, repetition, arithmetic, and robustness
   evaluation baselines.
4. A written scientific plan, control, promotion/rejection criteria, clean
   reviewed commit, successful full preflight, and exact hash-bound human
   authorization.

None of this work authorizes data generation, training configuration creation,
checkpoint conversion, or an optimizer update.
