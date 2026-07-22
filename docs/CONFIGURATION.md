# Configuration Guide

> Documentation for VASU's configuration system.

---

# Overview

VASU uses a configuration-driven design to make experiments reproducible and easy to modify.

Instead of hardcoding hyperparameters throughout the codebase, all important settings are stored in configuration classes.

Benefits:

* Reproducibility
* Cleaner code
* Easier experimentation
* Simpler scaling
* Better maintainability

---

# Configuration System

VASU currently uses two configuration classes:

```text id="k7m2ph"
ModelConfig
TrainConfig
```

---

# ModelConfig

`ModelConfig` defines the architecture of the language model.

Example:

```python id="x3q8ut"
ModelConfig(
    vocab_size=32000,
    max_seq_len=256,
    dim=384,
    n_heads=6,
    n_layers=8,
    hidden_dim=1536,
    dropout=0.1,
    rope_theta=10000.0,
    bias=False,
)
```

---

# Parameters

## vocab_size

```python id="n4w9ra"
vocab_size = 32000
```

Description:

Number of tokens in the tokenizer vocabulary.

Impact:

* Larger vocabulary → more parameters.
* Smaller vocabulary → more subword splitting.

Recommendation:

```text id="f8t6zl"
32000–50000
```

---

## max_seq_len

```python id="a5m2uk"
max_seq_len = 256
```

Description:

Maximum number of tokens the model can process.

Impact:

* Larger context improves understanding.
* Increases memory usage.

Future goal:

```text id="r9y3xs"
1024+
```

---

## dim

```python id="b1j8cv"
dim = 384
```

Description:

Hidden dimension of the model.

This determines:

* Embedding size
* Attention dimension
* Transformer capacity

Larger values increase:

* Model quality
* Memory usage
* Training time

---

## n_heads

```python id="w6r1pd"
n_heads = 6
```

Description:

Number of attention heads.

Head dimension:

```text id="s2n4vg"
head_dim = dim / n_heads
         = 384 / 6
         = 64
```

---

## n_layers

```python id="u8q5mh"
n_layers = 8
```

Description:

Number of Transformer blocks.

Increasing layers generally improves:

* Reasoning
* Language understanding
* Model capacity

---

## hidden_dim

```python id="z7k3fn"
hidden_dim = 1536
```

Description:

Intermediate dimension of the SwiGLU feed-forward network.

Expansion ratio:

```text id="p4y9ct"
1536 / 384 = 4×
```

---

## dropout

```python id="h3x7vu"
dropout = 0.1
```

Description:

Dropout probability used during training.

Purpose:

* Reduce overfitting.
* Improve generalization.

---

## rope_theta

```python id="j1m6op"
rope_theta = 10000.0
```

Description:

Base frequency used by Rotary Positional Embeddings.

Larger values may improve long-context extrapolation.

---

## bias

```python id="e5q2zw"
bias = False
```

Description:

Controls whether linear layers use bias terms.

Modern LLMs commonly disable bias parameters.

Benefits:

* Fewer parameters
* Slightly lower memory usage

---

# TrainConfig

`TrainConfig` controls the training process.

Example:

```python id="m7k9xa"
TrainConfig(
    batch_size=4,
    epochs=20,
    learning_rate=3e-4,
    weight_decay=0.01,
    seed=42,
    gradient_accumulation_steps=8,
    grad_clip=1.0,
    use_amp=True,
)
```

---

# Training Parameters

## batch_size

```python id="y4f1nc"
batch_size = 4
```

Description:

Number of samples processed simultaneously.

Increasing batch size:

* Improves GPU utilization.
* Requires more VRAM.

---

## epochs

```python id="x6z8qp"
epochs = 20
```

Description:

Number of passes through the dataset.

---

## learning_rate

```python id="d9w7lu"
learning_rate = 3e-4
```

Description:

Controls how quickly the model updates its parameters.

Too large:

* Unstable training.

Too small:

* Slow convergence.

---

## weight_decay

```python id="q8t4kr"
weight_decay = 0.01
```

Description:

L2 regularization used by AdamW.

Benefits:

* Reduces overfitting.
* Improves generalization.

---

## seed

```python id="p1r6jb"
seed = 42
```

Description:

Random seed used for reproducibility.

---

## gradient_accumulation_steps

```python id="s5x8km"
gradient_accumulation_steps = 8
```

Description:

Allows larger effective batch sizes.

Effective batch size:

```text id="g7v3zu"
4 × 8 = 32
```

---

## grad_clip

```python id="n2c7fy"
grad_clip = 1.0
```

Description:

Maximum gradient norm.

Purpose:

* Prevent exploding gradients.
* Improve stability.

---

## use_amp

```python id="r4m8jh"
use_amp = True
```

Description:

Enables Automatic Mixed Precision training.

Benefits:

* Lower VRAM usage
* Faster training
* Larger models on consumer GPUs

---

## checkpoint_path

```python id="k8p1xd"
checkpoint_path = "checkpoints/vasu.pt"
```

Description:

Path to the latest training checkpoint.

---

## checkpoint_dir

```python id="a2y6vf"
checkpoint_dir = "checkpoints"
```

Description:

Directory used to store checkpoints.

---

## save_every_steps

```python id="z5q3ne"
save_every_steps = 1000
```

Description:

Frequency of saving checkpoints.

---

# Effective Batch Size

Formula:

```text id="m9t2kb"
effective_batch_size =
batch_size × gradient_accumulation_steps
```

Current value:

```text id="v6x4rp"
32
```

---

# Configuration Workflow

```text id="c3n8mu"
ModelConfig
      ↓
Build Model
      ↓
TrainConfig
      ↓
Build Trainer
      ↓
Start Training
```

---

# Experiment Management

Recommended practice:

Create separate configurations for different experiments.

Example:

```text id="f7k1sy"
configs/
├── vasu_31m.py
├── vasu_60m.py
├── vasu_long_context.py
└── vasu_chat.py
```

---

# Backward Compatibility

Configuration changes can break:

* Checkpoints
* Tokenizers
* Training scripts

Major configuration changes should be documented.

---

# Future Configuration Features

* YAML configuration files
* Command-line overrides
* Configuration validation
* Experiment tracking
* Automatic hyperparameter logging

---

# Example Configurations

## Small Model

```python id="u3v8mp"
dim = 256
layers = 6
heads = 4
```

---

## Medium Model

```python id="r7n2wa"
dim = 512
layers = 12
heads = 8
```

---

## Large Model

```python id="q6x1zd"
dim = 768
layers = 16
heads = 12
```

---

# Design Philosophy

Configuration should make experimentation simple.

A new experiment should require changing a few configuration values, not rewriting the entire codebase.

This principle allows VASU to scale from a small educational project to a larger research and production framework while remaining easy to understand and maintain.
