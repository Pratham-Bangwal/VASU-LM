# Contributing to VASU

Thank you for your interest in contributing to **VASU (Virtual AI System for Understanding)**.

VASU aims to be an educational, research-oriented, and eventually production-ready open-source language model framework. Every contribution, whether it is code, documentation, testing, or ideas, is appreciated.

---

# Philosophy

VASU follows a simple principle:

> Build every layer. Understand every layer.

Contributions should prioritize:

* Correctness
* Maintainability
* Scalability
* Reproducibility
* Clean software engineering

---

# Ways to Contribute

You can contribute by:

* Reporting bugs
* Suggesting features
* Improving documentation
* Adding tests
* Implementing new features
* Optimizing performance
* Conducting experiments
* Improving training and inference pipelines

---

# Before You Start

Please:

1. Read the documentation.
2. Check existing issues.
3. Search for duplicate feature requests.
4. Open a discussion before implementing major architectural changes.

---

# Development Setup

## Clone the Repository

```bash
git clone <repository-url>
cd VASU
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Coding Standards

## Python Style

* Follow PEP 8.
* Use meaningful variable names.
* Keep functions small and focused.
* Prefer composition over large monolithic classes.

---

## Type Hints

Use type hints whenever possible.

Example:

```python
def train_epoch(model: nn.Module) -> float:
    ...
```

---

## Documentation

Public functions and classes should include docstrings.

Example:

```python
def encode(text: str) -> list[int]:
    """
    Encodes text into token IDs.
    """
```

---

# Project Structure

```text
vasu/
├── model/
├── training/
├── inference/
├── tokenizer/
├── datasets/
├── callbacks/
└── utils/
```

Keep responsibilities separated.

Avoid placing unrelated functionality into existing modules.

---

# Engineering Principles

## Prefer Clean Solutions

Avoid hacks or temporary fixes when a maintainable solution exists.

---

## Preserve Compatibility

When possible:

* Keep checkpoints compatible.
* Keep tokenizer formats compatible.
* Avoid unnecessary breaking changes.

---

## Modular Design

Every module should have a single responsibility.

Examples:

* Model → neural network architecture
* Training → optimization and training loops
* Inference → text generation
* Tokenizer → tokenization logic
* Datasets → data processing

---

# Adding New Features

Before implementing a new feature, answer:

1. Why is this change needed?
2. Which files will change?
3. How does it affect the rest of the project?
4. Does it break compatibility?
5. Is documentation required?

---

# Pull Request Guidelines

## Small Changes

Examples:

* Bug fixes
* Documentation improvements
* Tests

Can be submitted directly.

---

## Large Changes

Examples:

* New attention mechanisms
* Training architecture changes
* New inference systems

Please open a discussion first.

---

# Commit Message Style

Use Conventional Commits.

Examples:

```text
feat: add top-p sampling
fix: resolve checkpoint loading bug
docs: update architecture documentation
refactor: simplify attention module
test: add tokenizer tests
```

---

# Testing

Run tests before submitting changes.

```bash
pytest
```

New features should include:

* Unit tests
* Integration tests when applicable
* Documentation updates

---

# Documentation Requirements

Major changes must update the appropriate documents.

| Change               | Documents to Update    |
| -------------------- | ---------------------- |
| New feature          | README, PROJECT_STATUS |
| Architecture changes | ARCHITECTURE           |
| New milestone        | ROADMAP                |
| Release              | CHANGELOG              |
| Model changes        | MODEL_CARD             |

---

# Reporting Bugs

Please include:

* Operating system
* Python version
* PyTorch version
* GPU information
* Error message
* Steps to reproduce

---

# Feature Requests

Feature requests should include:

* Motivation
* Proposed solution
* Alternatives considered
* Potential drawbacks

---

# Code Review Checklist

Before submitting:

* [ ] Code follows PEP 8.
* [ ] Type hints added where appropriate.
* [ ] Documentation updated.
* [ ] Tests pass.
* [ ] No unnecessary dependencies added.
* [ ] Compatibility considered.

---

# Areas Where Help Is Needed

## Architecture

* Flash Attention
* Grouped Query Attention
* Long-context support

## Training

* Multi-GPU training
* LoRA fine-tuning
* Evaluation benchmarks

## Inference

* Streaming generation
* Quantization
* Efficient serving

## Tooling

* Web UI
* API server
* Voice support

---

# Code of Conduct

Please be respectful and constructive.

The goal of VASU is to learn, experiment, and build together.

---

# Thank You

Thank you for contributing to VASU.

Every contribution helps move the project closer to becoming a modern, production-quality open-source language model framework.
