# VASU

Virtual AI System for Understanding

A GPT-style decoder-only transformer language model built completely from scratch in PyTorch.

## Features

- Decoder-only Transformer
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Weight Tying
- Mixed Precision (AMP)
- Gradient Accumulation
- Memory Mapped Dataset
- TensorBoard
- Checkpoint Resume
- TinyStories Training Pipeline

## Current Model

Parameters: ~31M

Dataset:
TinyStories

Vocabulary:
32000 ByteLevel BPE

GPU:
RTX 4050 Laptop GPU

Framework:
PyTorch

Status:
Under Active Development

## Folder Structure

...

## Quick Start

python scripts/train_tokenizer.py

python scripts/prepare_dataset.py

python train.py