# Codex Context for VASU-LM

## Verified project state

- VASU is a from-scratch PyTorch decoder-only language-model project.
- VASU-31M remains a historical evaluated baseline.
- VASU-31M assistant checkpoint: `checkpoints/ultrachat_fineweb/best.pt`.
- VASU-31M manual baseline: 2.225 / 5.
- VASU-60M has 58,337,792 parameters.
- Base pretraining completed through global step 200,000.
- Authoritative base checkpoint: `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`.
- Preferred assistant checkpoint: `checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt`.
- UltraChat masked v2 is experimental and not promoted.

## Compatibility rules

- Keep `ModelConfig()` VASU-31M defaults unchanged.
- Use the opt-in VASU-60M configuration for 60M checkpoints.
- Never load VASU-31M weights into VASU-60M or vice versa.
- The shared 32,000-token tokenizer and processed datasets remain reusable.
- Do not change checkpoint schema without explicit approval.

## Operational rules

- Preserve thermal stopping, atomic saves, corrupt-checkpoint filtering, retention, and disk checks.
- Keep milestone checkpoints outside operational retention.
- Distinguish completed training, experimental branches, and planned work using `docs/PROJECT_STATUS.md` as the authoritative source.
- Prefer diagnostics and small reviewable changes over broad rewrites.
