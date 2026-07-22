# Quick Start

## Train a Model

```bash
python train.py
```

---

## Resume Training

Simply run:

```bash
python train.py
```

Checkpoint loading is automatic.

---

## Load Tokenizer

```python
tokenizer = VASUTokenizer()
tokenizer.load("assets/tokenizer.json")
```

---

## Load Model

```python
model = VASUModel(ModelConfig())
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
```

---

## Generate Text

```python
prompt = "Hello"

tokens = tokenizer.encode(prompt)
output = generate(tokens)
```

---

## Next Steps

* Read `ARCHITECTURE.md`
* Read `TRAINING.md`
* Read `INFERENCE.md`
