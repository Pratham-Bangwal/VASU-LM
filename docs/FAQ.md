# Frequently Asked Questions

## Why build VASU?

To understand every component of modern LLMs and eventually build a production-ready AI framework.

---

## Is VASU built from scratch?

Yes.

Every major component is implemented independently while using PyTorch as the tensor and autograd backend.

---

## Is VASU production ready?

Not yet.

VASU is currently an educational and research project.

---

## Why use Byte-Level BPE?

* Handles any text.
* Good generalization.
* Industry standard.

---

## Why use RoPE?

* Better positional understanding.
* Better long-context extrapolation.

---

## Why use RMSNorm?

* Faster than LayerNorm.
* Lower computational cost.

---

## Why use SwiGLU?

* Better parameter efficiency.
* Used by modern LLMs.

---

## Can I train on a consumer GPU?

Yes.

VASU is designed to work on consumer hardware.

---

## Which GPU was used?

NVIDIA RTX 4050 Laptop GPU (6 GB).

---

## Can I contribute?

Absolutely.

See `CONTRIBUTING.md`.
