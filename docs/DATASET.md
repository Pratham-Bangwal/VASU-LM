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
