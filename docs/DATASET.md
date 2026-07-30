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
not a dataset release. It references two existing reviewed CC0 instruction
batches and excludes four examples that exactly match frozen evaluation
questions. The remaining 996 examples have deterministic planned
train/development/evaluation membership, but no 513-token binary, stored mask,
logical manifest, or training authorization exists.

This plan is compatible only with the response-masked instruction contract. It
must not be presented as VASU-140M base-pretraining data.
