# Checkpoints Guide

## Arithmetic v2 schedule identity

Arithmetic v2 resume identity includes the arithmetic manifest SHA-256 in
addition to token/mask and schedule hashes. A changed schedule or regenerated
source manifest blocks exact continuation rather than silently changing data.

## Capability-CPT checkpoint roles

The production Candidate C runtime writes `latest.pt` for recovery,
`final.pt` after schedule completion, and separate `best_fineweb.pt`,
`best_wikimedia.pt`, and `best_arithmetic.pt` files. It does not use an opaque
mixed best score. The step-200 milestone and two newest other periodic
checkpoints are retained; protected final/latest/domain-best files are never
removed by periodic retention.

Each save is serialized to a sibling temporary file, flushed, reloaded on CPU,
checked for its required keys and internal step, then atomically promoted.
A SHA-256/size sidecar is written after promotion. Explicit resume rejects
temporary, truncated, corrupt, wrong-architecture, and identity-mismatched
files before model allocation. Capability validation events and best-domain
state are checkpointed, so an interval event is not repeated after resume.

## Scheduled-mixture identity

For scheduled-mixture training, additive `training_progress.dataset_identity`
stores the schedule SHA-256 and ordered source token/mask hashes. Exact resume
is rejected if the active schedule or source identity differs. Existing
checkpoints without this field retain their existing compatibility behavior;
no checkpoint tensor or container key was changed.

## Checkpoint roles

VASU checkpoints preserve model weights and, for resumable training checkpoints, optimizer, scheduler, epoch, loss, and `global_step` state.

### VASU-31M assistant checkpoint

`checkpoints/ultrachat_fineweb/best.pt`

This is the stable instruction-tuned fallback and the source of the 2.225 / 5 manual baseline.

### VASU-60M preserved milestone

`checkpoints/vasu_60m/milestones/fineweb_step_54060.pt`

This is a base-pretraining milestone at global step 54,060. It is not an assistant checkpoint. The next planned milestone is step 100,000 and is not complete.

### VASU-60M operational checkpoints

`checkpoints/vasu_60m/fineweb_blocks/`

The block trainer resumes from the highest valid internal `global_step`. Operational retention keeps the latest three periodic step checkpoints, latest two block-final checkpoints, and latest thermal-stop checkpoint. Preserved milestones are in a separate directory and are not subject to this retention.

## Integrity safeguards

- Save to a temporary file before atomic replacement.
- Ignore `.tmp`, corrupt, or incomplete candidates.
- Require the existing checkpoint keys before resume.
- Refuse new saves when free disk falls below the configured threshold.
- Refuse accidental overwrite of an existing checkpoint path.

## Compatibility

Always construct the matching model configuration. VASU-31M and VASU-60M use the same checkpoint container convention but incompatible model tensor shapes. The shared tokenizer does not make model weights interchangeable.

## Exact mid-epoch resume

New checkpoints written through the general `Trainer` add a
`training_progress` metadata field. It records compact sampler state (seed,
epoch, next batch), partial accumulated gradients, AMP scaler state, and
process RNG states. This continues from the exact next training batch without
serializing the shuffled permutation.

The field also records an explicit phase: `train`,
`post_train_pre_validation`, or `next_epoch`. A final-batch step checkpoint
uses `post_train_pre_validation`, so a restart performs the pending validation
and scheduler step exactly once. Best, epoch, and latest checkpoints are saved
with `next_epoch` only after validation, scheduler stepping, and sampler
advancement are complete.

Older checkpoints remain loadable. They emit a `RuntimeWarning` and use legacy
next-epoch behavior because they do not contain enough state to prove an exact
mid-epoch position. Exact resume also requires an identical dataset size,
batch size, seed, shuffle setting, and `drop_last` setting; a mismatch fails
clearly rather than silently approximating the position.

Generic checkpoint writes are atomic: VASU serializes to a sibling `.tmp` file
and promotes it with `os.replace` only after serialization succeeds. A failed
write leaves the previous checkpoint intact. Non-finite gradients explicitly
skip their optimizer update, so neither `global_step` nor the epoch scheduler
claims progress that did not occur.

Bounded VASU-60M checkpoint profiling on the development RTX 4050 measured
approximately 700 MiB for an optimizer-boundary checkpoint and 934 MiB for a
mid-accumulation checkpoint. The difference is the required saved gradient
state for exact resume. Boundary checkpoints store an empty gradient mapping.
The profiler writes only to `tmp/profiling/checkpoints/` and removes artifacts
unless explicitly asked to retain them.

## Best practices

- Keep milestone checkpoints outside operational retention.
- Record model configuration, dataset stage, losses, and global step.
- Verify a checkpoint before deleting older recovery points.
- Never commit large model checkpoints to Git.
- Do not infer validity from a filename alone; inspect the payload and internal `global_step`.
