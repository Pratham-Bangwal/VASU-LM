# Candidate D Training-Signal Audit and Next Hypothesis

Status: evidence complete; non-authorizing design brief

## Evidence

The immutable arithmetic-v2 audit (`vasu_arithmetic_training_signal_audit_v1`)
processed all 32,000 logical training examples. No answer contains serialized
intermediate steps. Final answers average 3.075 tokenizer tokens (range 1--8).
Only 130,403 of 764,979 supervised targets (17.05%) cover final answer/EOS
positions; 634,576 targets supervise non-final context. The release and
tokenizer identities remain respectively
`8da3a1acfb2d2481c295b0d226c4695613b56670989d001fa27e65bed366115c` and
`04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`.

## Proposed hypothesis

A future arithmetic treatment should test whether **verified, answer-only
supervision of deterministic intermediate computations** improves direct
operation generalization relative to an otherwise matched final-answer-only
control. This is a proposal, not an authorization or a claim that scratchpads
will solve the observed failure.

## Required design gates

Before any implementation or training: define a new experiment identity, build
an immutable step-verification generator, prove train/dev/eval separation and
template isolation, compare answer-only final targets with answer-only verified
intermediate targets under a matched control, pre-register direct-operation and
retention gates, validate masks at shifted target positions, and obtain separate
human authorization. Candidate B remains blocked and no existing authorization
may be reused.

## Compatibility

This brief changes no model architecture, checkpoint schema, tokenizer, source
dataset, schedule, mask, or training configuration. Candidate A remains the
preferred continued-pretraining base.
