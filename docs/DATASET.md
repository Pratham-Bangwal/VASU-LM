# Dataset Guide

## Current Datasets

### Pretraining

* TinyStories
* FineWeb

### Instruction Tuning

* Alpaca
* UltraChat

---

## Dataset Structure

```text
data/
├── raw/
└── processed/
```

---

## Pipeline

```text
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Binary Dataset
    ↓
Training
```

---

## Binary Datasets

Processed datasets are stored in binary format for:

* Faster loading
* Lower memory overhead
* Efficient random access

---

## Response Masking

Instruction tuning datasets support response masking.

Only assistant responses contribute to loss computation.

---

## Future Improvements

* Dataset versioning
* Dataset metadata
* Streaming datasets
* Larger pretraining corpora

---

## VASU-140M specification-only instruction plan

The VASU-140M instruction seed plan is metadata and read-only qualification,
not base-pretraining data. It references two existing reviewed CC0 instruction
batches and excludes four examples that exactly match frozen evaluation
questions. The subsequently authorized one-time production publication created
the immutable 996-example 513-token release with deterministic 898/48/50
train/development/evaluation membership; its consumed receipt and publication
audit remain separate from training authorization.

This plan is compatible only with the response-masked instruction contract. It
must not be presented as VASU-140M base-pretraining data.

The associated fixture-only transactional constructor remains a bounded test
layer. The separately reviewed production builder performed the one authorized
publication with deterministic token/mask validation and failure recovery.

The production builder remains fail closed: a future publication would require
a new exact, expiring one-build authorization and a clean reviewed commit. The
existing authorization was consumed and cannot authorize another publication
or any training.
