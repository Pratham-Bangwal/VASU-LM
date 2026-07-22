# Checkpoints Guide

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

## Best practices

- Keep milestone checkpoints outside operational retention.
- Record model configuration, dataset stage, losses, and global step.
- Verify a checkpoint before deleting older recovery points.
- Never commit large model checkpoints to Git.
- Do not infer validity from a filename alone; inspect the payload and internal `global_step`.
