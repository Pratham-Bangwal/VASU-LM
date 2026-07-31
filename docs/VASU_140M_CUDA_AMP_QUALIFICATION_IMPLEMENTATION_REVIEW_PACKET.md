# VASU-140M CUDA/AMP Qualification Implementation Review Packet

Review the implementation against the accepted design. Required files:

1. `AGENTS.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/VASU_140M_CUDA_AMP_QUALIFICATION_DESIGN.md`
4. `docs/VASU_140M_CUDA_AMP_QUALIFICATION_DESIGN_INDEPENDENT_REVIEW_DECISION_20260731.md`
5. `docs/VASU_140M_CUDA_AMP_QUALIFICATION_IMPLEMENTATION_AUDIT_20260731.md`
6. `scripts/qualify_vasu_140m_cuda_amp.py`
7. `tests/test_vasu_140m_cuda_amp_qualification.py`
8. `scripts/qualify_vasu_140m_cpu.py`
9. `scripts/qualify_vasu_140m_checkpoint.py`

Confirm the implementation has no hidden data, optimizer, schedule, training,
or production-checkpoint path. Check CUDA rejection, BF16/FP16 selection,
synthetic 512-token inputs, temporary checkpoint isolation, result overwrite
protection, cleanup/failure preservation, thermal handling, and report fields.

Run the focused tests, Ruff, `python scripts/preflight_vasu_140m.py`,
`git diff --check`, and `git status --short`.

Acceptance does not authorize a real CUDA execution, commit, push, data
release, configuration, schedule, optimizer creation, checkpoint creation
under `checkpoints/`, or training.
