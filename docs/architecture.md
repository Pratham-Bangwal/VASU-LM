# VASU-1 Architecture

## Overview

VASU (Virtual AI System for Understanding) is a lightweight decoder-only Transformer language model built from scratch using PyTorch.

## Goals

- Train from scratch
- Run on a single RTX 4050 GPU
- Support conversational text generation
- Modular and extensible architecture
- Modern transformer design

## Core Components

- Byte-level BPE tokenizer
- Token embeddings
- Rotary Positional Embeddings (RoPE)
- Multi-Head Self-Attention
- RMSNorm
- SwiGLU Feed Forward Network
- Weight tying
- Autoregressive text generation

## Future Features

- Streaming generation
- Quantization
- LoRA fine-tuning
- Function calling
- Vision module