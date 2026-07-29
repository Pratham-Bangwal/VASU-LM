# Candidate E Matched Budget and Evaluation Protocol

Status: proposed for independent review; non-authorizing.

## Frozen matching rule

Candidate E control and treatment must use the same parent checkpoint,
tokenizer, optimizer, scheduler, seed, microbatch shape, gradient
accumulation, total optimizer updates, and **processed model-token budget**.
The processed-token budget is computed as:

```
scheduled_records × sequence_length
```

where every scheduled record has 256 next-token positions. Both arms must
therefore schedule the identical record count and complete the identical update
count. They must not be matched by logical-example count or supervised-target
count: treatment deliberately exposes verified intermediate target tokens,
which is the hypothesis under test.

The release manifests must report, before any schedule is accepted:

- packed-record count and mean utilization for each arm;
- total stored response/EOS supervised tokens and supervision fraction;
- deterministic replay policy and projected per-example reuse;
- the exact equal processed-token and optimizer-update accounting.

Any unmatched parent, record count, sequence length, update count, source
mixture, tokenizer, mask, or runtime identity invalidates the comparison.

## Frozen evaluation protocol

1. Use the new Candidate E development split only for checkpoint selection.
   Select the highest macro exact accuracy over addition, subtraction,
   multiplication, exact division, percentage, sequence, mixed expression,
   and word problem; break ties by overall exact accuracy, lower malformed
   rate, then lower retention loss.
2. Do not inspect the new held-out evaluation split until one checkpoint is
   selected per arm.
3. Run each selected checkpoint once on the held-out split with identical
   greedy decoding, prompt serialization, parser, maximum generation length,
   evaluator revision, and device/runtime configuration.
4. Report overall and per-operation exact accuracy, malformed rate,
   unanswered rate, prompt-leakage rate, output-length distribution, and
   bootstrap confidence intervals for the treatment-minus-control difference.
5. Run the existing factual retention and fixed-prompt repetition evaluations
   unchanged for both selected checkpoints. Report all raw results, including
   failures; no result may be omitted because a primary gate fails.

## Pre-registered decision gates

The treatment can be considered a positive signal only if all conditions hold:

- held-out macro direct-operation exact accuracy exceeds matched control by at
  least 5 percentage points and the paired bootstrap 95% interval excludes
  zero;
- at least four direct-operation families improve by 5 points or more;
- malformed plus prompt-leakage rate is no worse than control by more than 1
  percentage point;
- FineWeb and Wikimedia validation loss are each within 1% relative of control;
- factual accuracy is within 2 points of control; and
- fixed-prompt repetition is within 2 points of control, with manual review of
  any material qualitative regression.

Failure of any condition rejects promotion. It does not invalidate the stored
experiment result and does not authorize a retry, extension, or follow-on run.

## Safety and integrity gates before execution

Before a separate training authorization, the approved release and schedule
must pass deterministic regeneration/hash checks, target-shift mask checks,
CUDA smoke, exact-resume, disk-space, thermal, checkpoint-validation, and
clean-tree gates. The parent checkpoint SHA-256, release manifests, schedule,
configuration, evaluator revision, and runtime identity must be recorded in
the authorization record. No new training is authorized by this protocol.
