# VASU-140M Real-Data Exact-Resume Contract Implementation Audit

Date: 2026-08-01

Status: **author-side implementation evidence; non-authorizing.**

## Scope

This package implements the fail-closed identity and evidence boundary required
before a real-data exact-resume execution package can be constructed. It does
not implement or invoke the bounded CUDA workload itself.

The implementation provides:

- an exact execution-specification schema bound to VASU-140M, a future
  immutable two-source base release, no-replacement schedule, evaluation
  development evidence, prospective optimizer/scheduler implementations, CUDA
  AMP mode, two optimizer updates, and the two required interruption points;
- byte-count and SHA-256 verification with an open handle plus a second
  immediately-before-load mutation check;
- deterministic digests for model, optimizer, scheduler, scaler, partial
  gradients, sampler, validation, and Python/NumPy/PyTorch CPU/CUDA RNG state;
- exact control/resume comparisons with no numerical tolerance;
- a strict result schema requiring the complete adversarial matrix, atomic and
  strict checkpoint evidence, cleanup, and non-authorization; and
- a frozen contract smoke that creates no optimizer, checkpoint, data release,
  CUDA workload, or training path.

## Identities

- Source-admission hardening anchor: `07f476b991f276e9c3fa378ddce76988769e90b9`
- Implementation SHA-256: `fdc2e231fb88ba89b7239dced2966252256c36acfd7757dad0f640f50df1e782`
- Test SHA-256: `9e14709d6a86e2594089c9d3308e474b5fa07902557ad04fe1d7317d2ca1ad08`
- Smoke SHA-256: `6c7b86deecfba3b7c4efdd8adce42732fec4ffb85f830638697c8ec08a710f82`
- Frozen fixture SHA-256: `00663a5ec3fae5776fbb16d9d95b1eac878f8a53a055020fa57f331dee238194`
- Fixture specification SHA-256: `b766ac2e1fca964af376fca84382ec2f562b22a867d474545c1bdc77d1070dc3`
- Fixture result SHA-256: `e75cb4e30b0bf05d0f0d209a385e6b7a50e72b525b7e75e72ae1c60a4faaad41`

## Validation

`python -m pytest tests\test_vasu_140m_real_data_resume.py -q`

- 21 passed.

`python -m pytest tests\test_vasu_140m_real_data_resume.py tests\test_vasu_140m_exact_resume_qualification.py tests\test_resumable_training.py tests\test_checkpoint_io.py tests\test_model_family_checkpoint_identity.py -q`

- 59 passed;
- 12 existing CPU pin-memory warnings from resumable-training tests.

`python -m ruff check vasu\training\vasu_140m_real_data_resume.py tests\test_vasu_140m_real_data_resume.py scripts\smoke_vasu_140m_real_data_resume_contract.py`

- passed.

The smoke output reproduced
`evaluation/fixtures/vasu_140m_real_data_resume_contract_qualification_v1.json`
exactly.

Protected path checks remained absent:

- `data/processed/vasu_140m/base_pretraining`;
- `data/manifests/vasu_140m/base_pretraining`;
- `checkpoints/vasu_140m/base_pretraining`; and
- `evaluation/results/vasu_140m/real_data_exact_resume`.

## Compatibility

The package is additive. It changes no model architecture, tokenizer, dataset,
mask, trainer, optimizer, scheduler, persistent checkpoint schema, or existing
resume path. Existing checkpoints and datasets remain compatible.

## Remaining Work

After clean commit, a separate reviewed workload runner must integrate these
contracts with the future accepted production release. Real source
selection/acquisition, production data publication, execution-specification
construction, CUDA execution, and training remain separately gated.

## Non-Authorization

No CUDA workload was invoked, optimizer was created, optimizer update was
performed, checkpoint was created, production input was opened, training was
started, or authorization was changed. This evidence does not authorize any of
those actions.
