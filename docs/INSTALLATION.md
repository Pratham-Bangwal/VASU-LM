# Installation Guide

## Requirements

* Python 3.10+
* PyTorch 2.x
* CUDA (optional)

---

## Clone Repository

```bash
git clone <repository-url>
cd VASU
```

---

## Create Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python train.py
```

Expected:

```text
Using device: cuda
Loading tokenizer...
Loading dataset...
Building model...
```
