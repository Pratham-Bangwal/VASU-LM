# Tokenizer Guide

> Documentation for the VASU Byte-Level BPE tokenizer.

---

# Overview

VASU uses a **Byte-Level Byte Pair Encoding (BPE)** tokenizer implemented using the Hugging Face Tokenizers library.

The tokenizer is responsible for:

* Training a vocabulary from text data
* Converting text into token IDs
* Converting token IDs back into text
* Saving and loading tokenizer files
* Providing consistent tokenization across training and inference

---

# Why Tokenization Matters

Large Language Models do not understand raw text directly.

Instead, text must be converted into numerical representations.

Example:

```text id="j23mqp"
Input:
Hello world!

Tokens:
["Hello", "Ġworld", "!"]

Token IDs:
[1543, 893, 12]
```

The model learns relationships between these token IDs.

---

# Why Byte-Level BPE?

VASU uses Byte-Level BPE because it provides several advantages:

## Handles Any Text

Since tokenization starts from bytes, every Unicode character can be represented.

---

## Smaller Vocabulary

Common subwords are merged into larger tokens.

Example:

```text id="p8q7ha"
play
playing
player
played
```

can share common pieces.

---

## Better Generalization

The tokenizer can handle:

* Rare words
* Misspellings
* New words
* Different languages

without requiring an enormous vocabulary.

---

## Industry Standard

Byte-Level BPE is used by several modern language models.

---

# Tokenizer Architecture

```text id="m1lfzt"
Raw Text
    ↓
Byte-Level Pre-Tokenizer
    ↓
BPE Encoding
    ↓
Token IDs
```

---

# Implementation

VASU uses:

```text id="t6mqwr"
Hugging Face Tokenizers
```

Core components:

* Tokenizer
* BPE model
* ByteLevel pre-tokenizer
* ByteLevel decoder
* BPE trainer

---

# Vocabulary Size

Current default:

```text id="rfwh3n"
32,000 tokens
```

The vocabulary size may increase in future versions as the model scales.

---

# Special Tokens

VASU currently defines four special tokens:

| Token   | Purpose               |
| ------- | --------------------- |
| `[PAD]` | Padding sequences     |
| `[UNK]` | Unknown tokens        |
| `[BOS]` | Beginning of sequence |
| `[EOS]` | End of sequence       |

---

# Tokenizer Training

## Training Data

The tokenizer is trained using text files.

Example structure:

```text id="tm8xrd"
data/
└── raw/
    ├── file1.txt
    ├── file2.txt
    └── file3.txt
```

---

## Training Process

```text id="u4m1fp"
Text Files
      ↓
Byte Extraction
      ↓
Initial Vocabulary
      ↓
BPE Merges
      ↓
Final Vocabulary
      ↓
tokenizer.json
```

---

# Training the Tokenizer

Example:

```python id="j8xqmt"
tokenizer = VASUTokenizer()

tokenizer.train(
    data_dir="data/raw",
    vocab_size=32000,
)

tokenizer.save(
    "assets/tokenizer.json"
)
```

---

# Saving

Tokenizer files are saved as:

```text id="zt89cq"
assets/tokenizer.json
```

This file contains:

* Vocabulary
* Merge rules
* Special tokens
* Configuration

---

# Loading

```python id="e8v0aq"
tokenizer = VASUTokenizer()
tokenizer.load(
    "assets/tokenizer.json"
)
```

---

# Encoding

Convert text into token IDs:

```python id="n2xg6r"
ids = tokenizer.encode(
    "Hello world!"
)
```

Example:

```text id="2w4tkp"
[1543, 893, 12]
```

---

# Decoding

Convert token IDs back into text:

```python id="zw5q2x"
text = tokenizer.decode(ids)
```

Output:

```text id="f0q8ph"
Hello world!
```

---

# Tokenizer Workflow

```text id="n1qud8"
Raw Text
     ↓
Tokenizer Training
     ↓
tokenizer.json
     ↓
Encode Text
     ↓
Token IDs
     ↓
Model
     ↓
Generated Token IDs
     ↓
Decode
     ↓
Generated Text
```

---

# Compatibility

The tokenizer must remain compatible with:

* Datasets
* Checkpoints
* Inference pipeline

Changing the vocabulary can invalidate existing checkpoints.

---

# Versioning Recommendation

Future versions of VASU should introduce tokenizer versioning:

```text id="a4xjzc"
tokenizer-v1.json
tokenizer-v2.json
tokenizer-v3.json
```

This avoids accidentally breaking older checkpoints.

---

# Best Practices

## Do Not Change Vocabulary Mid-Training

Changing the tokenizer after training begins will make checkpoints incompatible.

---

## Save Tokenizer with Checkpoints

Recommended structure:

```text id="m7r9sl"
checkpoints/
├── best.pt
├── epoch_1.pt
└── tokenizer.json
```

---

## Document Dataset Changes

Different datasets can significantly affect tokenizer quality.

---

# Future Improvements

* Larger vocabularies
* Tokenizer versioning
* Dataset-specific tokenizers
* Multilingual support
* Special token management
* Tokenizer benchmarks

---

# Design Philosophy

The tokenizer is the first stage of understanding language.

VASU's tokenizer is intentionally simple, transparent, and easy to study so that developers can understand exactly how text becomes tokens and how tokens become meaning inside the model.
