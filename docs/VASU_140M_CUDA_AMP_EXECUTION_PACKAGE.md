# VASU-140M CUDA/AMP One-Shot Execution Package

Status: **pending independent review and explicit human execution approval.**

## Authorized scope if later approved

At most one invocation of:

```powershell
python scripts\qualify_vasu_140m_cuda_amp.py --device 0 --output evaluation\results\vasu_140m_cuda_amp_qualification_20260731.json
```

This is a synthetic, no-optimizer CUDA qualification only. It may create one
system-temporary model-only checkpoint and one immutable ignored JSON result.
It must not create an artifact under `checkpoints/`, read a dataset, construct
an optimizer, create a schedule/configuration, or train.

## Preconditions immediately before execution

1. `git status --short` is empty.
2. `python scripts\smoke_vasu_140m_cuda_amp_postcommit.py` succeeds and emits
   implementation SHA `8cbf761948770acf7d26cde639b803884a3dc0cf0ed2d878723c0e7368846494`,
   test SHA `80a3958dca60d6809fd0961bdb67e9b66c987fc2d636e0042069f806a4834654`,
   `cuda_invoked=false`, and `training_authorized=false`.
3. The output path does not exist.
4. CUDA device 0 is present, and temporary free disk is at least 2 GiB.
5. Pre-run telemetry is below 88°C when available. If telemetry is unavailable,
   the result remains visible but cannot satisfy a later authorization-stage
   thermal requirement.
6. A human explicitly authorizes this exact one-shot command after independent
   review. Earlier publication, planning, or training permissions do not
   substitute for that approval.

## Stop and failure behavior

The tool stops on unavailable CUDA, identity failure, insufficient disk,
non-finite values, checkpoint reload failure, output overwrite, or 88°C
thermal event. It preserves any failing temporary directory for inspection and
does not retry automatically. A failed run is evidence, not a reason to rerun
without a new documented decision.

## Post-run handling

Validate the emitted result identity, confirm no temporary directory remains
after a successful run, record hardware observations, and obtain independent
review. A pass advances only the CUDA/AMP readiness sub-gate; it does not
authorize a base-data release, real-data resume test, training plan, or
training.
