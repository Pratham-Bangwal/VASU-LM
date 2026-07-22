# Troubleshooting

## CUDA Out of Memory

Solutions:

* Reduce batch size.
* Reduce sequence length.
* Increase gradient accumulation.

---

## NaN Loss

Possible causes:

* Learning rate too high.
* Dataset corruption.
* Numerical instability.

---

## Slow Training

Possible causes:

* CPU bottleneck
* AMP disabled
* Small batch size

---

## Checkpoint Won't Load

Possible causes:

* Model configuration changed.
* Vocabulary changed.
* Checkpoint corrupted.

---

## Shape Mismatch

Possible causes:

* Different vocab size.
* Different model dimensions.
* Different layer count.

---

## Poor Generation Quality

Possible causes:

* Insufficient training.
* Poor dataset quality.
* Small model size.
