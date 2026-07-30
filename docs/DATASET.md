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

The associated transactional constructor is currently fixture-only. It rejects
the planned production locations and more than 30 examples, so it cannot build
the 996-example release. Its purpose is to prove deterministic token/mask
publication, integrity validation, and failure recovery before production
construction is separately designed and reviewed.

The production-builder implementation proposal remains review-gated. Its
qualifier compiles the accepted sources only in memory and freezes expected
hashes and counts. Publication is unreachable without a separate exact,
expiring one-build authorization and a clean reviewed commit. No such
authorization or production artifact currently exists.
