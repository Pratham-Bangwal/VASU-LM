# VASU-140M CUDA/AMP Operational Qualification Design

Status: **design-only; non-authorizing.**

## Objective

Specify a bounded, reproducible, no-optimizer CUDA/AMP qualification for the
frozen `vasu_140m_v1` family. The qualification answers only whether the
intended hardware can execute and serialize the model safely under a defined
synthetic workload. It does not qualify a dataset, model quality, a training
schedule, an optimizer update, or a training run.

## Why this is required

CPU evidence, a model-only CPU checkpoint round trip, and synthetic CPU exact
resume do not establish CUDA kernel behavior, AMP numerical behavior, peak
GPU memory, thermals, power behavior, or checkpoint-I/O time. Those properties
must be measured before a VASU-140M base-pretraining plan can choose a runtime
envelope.

## Frozen identity and workload boundary

| Field | Required value |
|---|---|
| Family | `vasu_140m_v1` |
| Config SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Family SHA-256 | `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b` |
| Tokenizer/data inputs | none; deterministic synthetic token IDs only |
| Device | one explicitly selected CUDA device |
| Precision | CUDA autocast BF16 when supported, otherwise FP16; recorded exactly |
| Batch size | 1 |
| Sequence length | 512 |
| Warmup / measured iterations | 3 / 5 forward-backward iterations |
| Gradient handling | inspect finite gradients, then clear; no optimizer or update |
| Checkpoint I/O | one model-only checkpoint in a newly created temporary directory; reload strictly, validate, then delete the directory |
| Result destination | versioned ignored evaluation result; no overwrite permitted |

The runner must reject CPU execution, unavailable CUDA, a non-140M family,
identity drift, a non-empty result path, and a pre-existing qualification
temporary directory. It must set deterministic seeds and synchronize CUDA
before timing or reading memory statistics.

## Required observations

The immutable result must record:

- GPU name, compute capability, total memory, driver/runtime, PyTorch, Python,
  operating system, and selected CUDA device;
- AMP dtype and any fallback reason;
- model construction time, per-iteration forward/backward timing, synchronized
  token throughput, and peak allocated/reserved CUDA memory;
- finite logits, loss, and gradients for every measured iteration, plus the
  maximum observed gradient norm;
- baseline and post-run free-disk space, checkpoint bytes, write time, strict
  CPU reload time, file SHA-256, and checkpoint-family validation result;
- thermal/power samples before, during, and after the measured workload when a
  supported local telemetry provider is available; otherwise an explicit
  `telemetry_unavailable` state and provider error;
- model-state key equality before/after qualification, cleared-gradient proof,
  and explicit `optimizer_created=false`, `optimizer_updates=0`, and
  `training_authorized=false` fields.

## Safety policy

The implementation must reserve no production output location. It may create
only a process-unique temporary directory below the system temporary root and
must remove it only after successful strict reload and validation. On failure,
it must preserve the temporary directory path for manual inspection and never
promote it into `checkpoints/`.

The qualification must fail before the checkpoint-I/O phase if free disk space
is below the documented conservative threshold. It must stop before a further
measured iteration if telemetry reports a temperature at or above the existing
88°C laptop safety threshold. Missing telemetry is recorded rather than
silently treated as a safe thermal reading; an authorization-stage runtime
must still meet its own thermal-monitor requirement.

## Acceptance and rejection criteria

The qualification passes only if all required identity, CUDA, finite-value,
memory, strict-reload, state-invariance, checkpoint-cleanup, and result-write
checks pass. It does not define a training batch size, learning rate, or token
budget. The result must report measured values; no fixed throughput or memory
number is assumed in advance.

It rejects non-finite values, identity drift, unexpected model-state mutation,
checkpoint reload mismatch, failed cleanup after a successful run, result
overwrite, insufficient disk, or a thermal safety threshold event. Telemetry
unavailability is a visible result state that blocks promotion to an
authorization-stage runtime but does not falsify the narrow CUDA numerical
measurement.

## Alternatives considered

| Alternative | Decision |
|---|---|
| Reuse VASU-60M CUDA results | Rejected: model size/context and memory behavior differ. |
| Use actual base-pretraining data | Rejected: no reviewed 140M base-data release exists. |
| Create an optimizer to mimic training | Rejected: adds optimizer-state and update risks without answering this gate. |
| Use a tiny sequence only | Rejected: it does not exercise the 512-token memory boundary. |
| Bounded synthetic 512-token qualification | Recommended: measures the relevant operational envelope without training. |

## Required implementation and review sequence

1. Implement an isolated `scripts/qualify_vasu_140m_cuda_amp.py` and focused
   tests using mocked CUDA/telemetry boundaries where hardware is unavailable.
2. Produce a frozen synthetic qualification fixture from a controlled test
   seam; this is not a claim about real hardware.
3. Obtain independent implementation review.
4. Commit accepted code and reproduce a clean-commit qualification identity.
5. Run the real bounded CUDA qualification only after its reviewed execution
   package is accepted. Its result advances only readiness gate 1.

No step above authorizes source discovery, a base-data release, an experiment
configuration, optimizer creation outside the isolated test fixture, a
checkpoint under `checkpoints/`, or training.
