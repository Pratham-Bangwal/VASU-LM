# Inference Guide

> Documentation for text generation and inference in VASU.

---

# Overview

Inference is the process of generating text using a trained VASU model.

Unlike training, inference does not update model parameters. The model simply predicts the next token repeatedly until generation stops.

---

# Inference Pipeline

```text id="j1r4vn"
Prompt
   ↓
Tokenizer
   ↓
Token IDs
   ↓
VASU Model
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
   ↓
Append Token
   ↓
Repeat
   ↓
Generated Text
```

---

# Text Generation Process

Given a prompt:

```text id="e4n1su"
Once upon a time,
```

The model predicts:

```text id="h8r0mw"
there
```

Then:

```text id="z0x9ab"
Once upon a time, there
```

The model predicts:

```text id="t3g7uy"
was
```

This process repeats until:

* Maximum length is reached
* End-of-sequence token is generated
* User stops generation

---

# Autoregressive Generation

VASU generates text one token at a time.

```text id="f9m6lw"
Token₁ → Token₂ → Token₃ → Token₄ → ...
```

Each prediction depends on all previously generated tokens.

---

# Input Processing

## Step 1

User prompt:

```text id="d5r2bn"
Hello, how are you?
```

## Step 2

Tokenizer converts text into token IDs.

```text id="r4j7yc"
[125, 781, 492, 13]
```

## Step 3

The model produces logits:

```text id="m7q9ta"
(batch_size,
 sequence_length,
 vocab_size)
```

For VASU-31M:

```text id="n1k6wp"
(batch_size,
 sequence_length,
 32000)
```

---

# Logits

Logits are raw scores for every token in the vocabulary.

Example:

```text id="w3h1qb"
Token A → 5.2
Token B → 1.7
Token C → -2.4
```

These values are converted into probabilities during sampling.

---

# Sampling

Sampling decides which token should be generated next.

---

# Greedy Decoding

Available by setting `top_k=1`.

```text id="z8v3mx"
Choose the token with the highest probability.
```

Advantages:

* Fast
* Deterministic

Disadvantages:

* Can become repetitive
* Less creative

---

# Temperature Sampling

Implemented in `sample_next_token`.

Formula:

```text id="g2q4ur"
logits = logits / temperature
```

Lower temperature:

* More deterministic.

Higher temperature:

* More creative.

---

# Top-k Sampling

Implemented in `sample_next_token`. `top_k` is clamped to the vocabulary size.

Example:

```text id="p6j8nv"
Top-k = 50
```

Only the top 50 most likely tokens are considered.

Benefits:

* Better diversity
* Less randomness

---

# Top-p (Nucleus) Sampling

Implemented in `sample_next_token`.

Example:

```text id="m5r7fx"
Top-p = 0.9
```

The smallest set of tokens whose cumulative probability exceeds 90% is considered.

Benefits:

* More natural text generation
* Widely used by modern LLMs

---

# Repetition Penalty

Implemented in `sample_next_token`.

Purpose:

Reduce repetitive outputs.

Example:

```text id="s9n3ht"
hello hello hello hello
```

A repetition penalty discourages repeatedly generating the same token.

---

# Maximum Generation Length

Generation usually stops when:

```text id="t1m5jy"
generated_tokens >= max_new_tokens
```

---

# End of Sequence Token

Generation may also stop when:

```text id="u6p2ke"
[EOS]
```

is generated.

---

# KV Cache

VASU includes support for Key-Value caching.

---

## Without KV Cache

```text id="k7w3la"
Prompt
   ↓
Recompute entire sequence
   ↓
Generate token
```

Complexity increases rapidly.

---

## With KV Cache

```text id="x4p1eg"
Prompt
   ↓
Reuse previous attention states
   ↓
Generate token
```

Benefits:

* Faster generation
* Lower latency
* Better chat performance

---

# Inference Workflow

```text id="v2f6zc"
Load Tokenizer
      ↓
Load Checkpoint
      ↓
Encode Prompt
      ↓
Generate Tokens
      ↓
Decode Tokens
      ↓
Return Text
```

---

# Example Workflow

```python id="e3k7fn"
tokenizer.load("assets/tokenizer.json")

checkpoint = torch.load(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model"],
    strict=True,
)
model.eval()

prompt = "Hello"

tokens = tokenizer.encode(prompt)

output = generate(tokens)

print(output)
```

---

# Interactive Chat

Current workflow:

```text id="h0y8dw"
User Message
      ↓
Conversation History
      ↓
Prompt Construction
      ↓
Generation
      ↓
Assistant Response
```

---

# Future Features

## Sampling

* Temperature Sampling
* Top-k Sampling
* Top-p Sampling
* Repetition Penalty
* Beam Search

---

## Performance

* Better KV Cache
* Quantization
* Batch Inference
* Speculative Decoding

---

## User Experience

* Streaming Generation
* Interactive CLI
* Web Interface
* Voice Interface

---

# Performance Considerations

Inference speed depends on:

* Model size
* Context length
* GPU memory
* Sampling strategy
* KV cache efficiency

---

# Common Issues

## Slow Generation

Possible causes:

* Long context lengths
* KV cache disabled
* CPU inference

---

## Repetitive Responses

Possible causes:

* Greedy decoding
* No repetition penalty
* Small model size

---

## Incoherent Responses

Possible causes:

* Insufficient training
* Poor prompt formatting
* Limited model capacity

---

# Future Direction

The long-term goal of VASU inference is to provide:

* Fast generation
* Interactive conversations
* Tool usage
* Memory systems
* Personal AI assistant capabilities

---

# Design Philosophy

Inference should be:

* Fast
* Simple
* Easy to understand
* Easy to extend
* Suitable for experimentation

Every generation step in VASU is intentionally transparent so developers can understand exactly how the model turns a prompt into meaningful text.
