# VASU-LM Repository Instructions

## 1. Project Identity

VASU, or Virtual AI System for Understanding, is a decoder-only Transformer language model built from scratch in PyTorch.

Treat VASU as a long-term open-source AI framework, not as a temporary assignment or disposable experiment.

Every change must improve at least one of these dimensions:

- correctness;
- training stability;
- reproducibility;
- performance;
- maintainability;
- scalability;
- usability;
- scientific traceability.

Do not introduce a change unless its benefits and trade-offs are understood.

---

## 2. Model Recommendation Before Every Task

Before beginning any VASU-LM task, start the response with a brief model recommendation based on the task’s difficulty, scope, and risk.

Use this policy:

### GPT-5.6 Sol

Use for:

- architecture decisions;
- major or cross-cutting refactors;
- difficult debugging;
- checkpoint compatibility;
- tokenizer or dataset compatibility;
- training-stability reviews;
- exact-resume and checkpoint-integrity work;
- authorization-gate changes;
- scientific promotion decisions;
- high-risk production-runtime changes;
- final technical audits.

Example:

> Recommended model: GPT-5.6 Sol — this task affects checkpoint compatibility and training reproducibility.

### GPT-5.6 Terra

Use for:

- routine feature implementation;
- normal refactoring;
- training scripts;
- evaluation tooling;
- dataset preparation tooling;
- experiment configuration;
- ordinary debugging;
- tests;
- documentation;
- repository maintenance.

Example:

> Recommended model: GPT-5.6 Terra — this is a routine evaluation-tooling task with moderate implementation complexity.

### GPT-5.6 Luna

Use for:

- formatting;
- log parsing;
- mechanical edits;
- basic repository checks;
- simple configuration updates;
- repetitive documentation cleanup;
- straightforward test generation;
- small isolated fixes.

Example:

> Recommended model: GPT-5.6 Luna — this is a low-risk mechanical validation task.

### GPT-5.5

Use for:

- independent second opinions;
- challenging earlier decisions;
- reviewing experimental conclusions;
- comparing alternative interpretations;
- checking whether promotion or rejection reasoning is too optimistic;
- independent scientific review.

Example:

> Recommended model: GPT-5.5 — use this as an independent review of the previous experiment decision.

If the current interface cannot switch models, state that the recommendation is advisory and continue using the available model.

This rule applies to all VASU-related work, including:

- coding;
- debugging;
- training;
- evaluation;
- architecture;
- research;
- documentation;
- experimentation;
- project management.

---

## 3. Mandatory Context Before Work

Before modifying the repository:

1. Read `docs/PROJECT_STATUS.md`.
2. Read all task-specific authoritative documentation.
3. Inspect the current Git status.
4. Inspect relevant implementation files and tests.
5. Confirm the current experiment, checkpoint, tokenizer, dataset, and lineage where compatibility may be affected.
6. Check for unrelated working-tree changes and preserve them.

Depending on the task, also read:

- `docs/ARCHITECTURE.md`
- `docs/CHECKPOINTS.md`
- `docs/TOKENIZER.md`
- `docs/EXPERIMENTS.md`
- `docs/CHANGELOG.md`
- `docs/ROADMAP.md`
- experiment-specific decision records;
- experiment-specific authorization records;
- relevant model, dataset, and runtime configuration files.

Repository files are authoritative. Do not rely on stale conversational assumptions when the committed documentation says otherwise.

---

## 4. Engineering Principles

### Correctness first

Diagnose root causes rather than masking symptoms.

Do not:

- suppress exceptions just to make tests pass;
- weaken validation without justification;
- bypass integrity checks;
- silently ignore mismatches;
- replace evidence with assumptions;
- hide failures behind fallback behavior.

### Maintainability

Prefer:

- modular code;
- explicit interfaces;
- clear ownership of responsibilities;
- reusable components;
- type hints;
- descriptive names;
- focused functions;
- focused tests;
- documented invariants.

Avoid:

- monolithic files;
- duplicated logic;
- hidden side effects;
- magic constants;
- broad exception handling;
- unnecessary rewrites;
- undocumented compatibility breaks.

### Scalability

Design changes should remain reasonable as VASU grows in:

- parameter count;
- sequence length;
- dataset size;
- source count;
- evaluation-suite count;
- checkpoint count;
- hardware scale;
- distributed-training complexity.

Do not design only for the current VASU-60M configuration when a scalable design is practical.

### Backward compatibility

Before modifying a persistent interface or artifact format, assess the impact on:

- checkpoints;
- optimizer state;
- scheduler state;
- gradient scaler state;
- tokenizer files;
- datasets;
- masks;
- manifests;
- schedules;
- configs;
- inference scripts;
- evaluation outputs;
- exact resume;
- existing experiment lineage.

Preserve compatibility whenever practical.

When compatibility cannot be preserved, document:

- what breaks;
- why it must break;
- affected artifacts;
- migration steps;
- whether old checkpoints remain loadable;
- whether old datasets remain usable.

---

## 5. Required Change Analysis

Before implementing a meaningful change, state:

1. Why the change is needed.
2. Which files must be modified.
3. The expected impact on the rest of the repository.
4. Whether existing checkpoints remain compatible.
5. Whether existing datasets and masks remain compatible.
6. Whether tokenizer compatibility is preserved.
7. Whether exact resume and reproducibility remain preserved.
8. Which tests will validate the change.
9. Whether documentation must be updated.

For small mechanical edits, keep this analysis brief.

For architecture, training, tokenizer, dataset, checkpoint, authorization, or runtime changes, provide a complete impact analysis before implementation.

---

## 6. Repository Safety

### Git rules

Never use:

```bash
git add .
```

Always stage files explicitly.

Do not commit unless the user explicitly authorizes the commit.

Do not push unless the user explicitly authorizes the push.

Before every commit, run:

```bash
git diff --check
git status --short
git diff --cached --stat
```

Review the complete staged diff before committing.

Use Conventional Commits:

- `feat(model): ...`
- `feat(training): ...`
- `fix(training): ...`
- `fix(evaluation): ...`
- `test(training): ...`
- `test(evaluation): ...`
- `docs: ...`
- `refactor(model): ...`
- `chore(config): ...`

Separate unrelated changes into separate commits.

### Destructive operations

Do not delete, overwrite, move, or regenerate authoritative artifacts without explicit approval.

Protected artifacts include:

- checkpoints;
- optimizer states;
- datasets;
- masks;
- tokenizer assets;
- manifests;
- schedules;
- evaluation evidence;
- authorization records;
- experiment decision records.

Prefer creating a new versioned artifact instead of overwriting an existing authoritative artifact.

Do not run destructive Git operations such as:

- `git reset --hard`;
- force push;
- branch deletion;
- history rewriting;
- discarding user changes;

without explicit approval.

### Working-tree discipline

Do not:

- modify unrelated files;
- stage unrelated files;
- reformat the entire repository for a targeted task;
- overwrite existing user edits;
- silently resolve unrelated conflicts.

Preserve unrelated working-tree changes exactly as found.

---

## 7. Training Authorization

Never start real training unless the user explicitly authorizes that exact experiment.

A training run is authorized only when an approved authorization record is bound to the exact experiment identity.

A valid authorization must include, where applicable:

- experiment ID;
- training config path;
- unauthorized config hash;
- authorized config hash;
- parent checkpoint path and hash;
- tokenizer path and hash;
- resolved mixture-manifest hash;
- schedule hash;
- source token hashes;
- source mask hashes;
- source manifest hashes;
- validation configuration hash;
- CUDA validation artifact;
- capability runtime identity;
- token budget;
- optimizer-update budget;
- batch size;
- sequence length;
- gradient accumulation;
- optimizer configuration;
- scheduler configuration;
- repository commit or reviewed implementation identity;
- explicit approver;
- explicit authorization scope.

A config field such as:

```json
"training_authorized": true
```

is never sufficient by itself.

The exact approved authorization record must exist and match the full hash-bound configuration.

Do not:

- bypass the authorization gate;
- weaken authorization validation;
- broadly accept `training_authorized=true`;
- copy another candidate’s authorization;
- infer authorization from a passing preflight;
- infer authorization from a scientific recommendation;
- infer authorization from an existing checkpoint.

### Candidate-specific rules

Candidate A and Candidate B require separate authorization records.

Candidate C authorization does not authorize Candidate A or Candidate B.

Candidate B remains unauthorized unless a separate scientific decision and authorization record explicitly change that state.

### Before launching training

Confirm all of the following:

1. Repository is clean.
2. Repository commit matches the reviewed implementation.
3. Config identity matches the authorization record.
4. Parent checkpoint hash matches.
5. Tokenizer hash matches.
6. Mixture manifest hash matches.
7. Schedule hash matches.
8. Source identities match.
9. Validation configuration matches.
10. Token and update budgets match.
11. Output directory is isolated.
12. Disk safeguard is active.
13. Thermal safeguard is active.
14. Checkpoint retention is configured.
15. Exact-resume behavior is tested.
16. Preflight passes.
17. No unintended candidate is authorized.

Do not start training during preparation, review, evaluation, documentation, or authorization-package tasks.

---

## 8. Checkpoint Integrity

Treat every checkpoint as a scientific record.

Checkpoint state must preserve, where applicable:

- model state;
- optimizer state;
- scheduler state;
- gradient-scaler state;
- global optimizer step;
- dataloader or schedule position;
- microbatch position;
- gradient-accumulation position;
- partial accumulated gradients;
- random number generator states;
- experiment identity;
- parent identity;
- tokenizer identity;
- source identities;
- schedule identity;
- optimizer and scheduler identities;
- validation history.

Checkpoint writes must be atomic.

Partial or corrupted checkpoints must never replace a valid checkpoint.

Reject checkpoints that are:

- corrupted;
- incomplete;
- from another experiment;
- from another candidate;
- created with a different parent;
- created with a different tokenizer;
- created with a different schedule;
- created with different data identities;
- created with incompatible optimizer or scheduler state.

Do not silently fall back to another checkpoint.

---

## 9. Exact Resume

Exact resume must continue the original run without changing its scientific identity.

It must not:

- repeat consumed records;
- skip records;
- reorder the schedule;
- reset warmup;
- duplicate optimizer updates;
- lose partial accumulation state;
- change optimizer backend;
- change scheduler totals;
- reset random states;
- restart from the parent checkpoint;
- silently discover an unintended checkpoint.

Resume must be explicit.

A resumed run must validate the checkpoint against:

- experiment ID;
- config identity;
- parent checkpoint;
- tokenizer;
- schedule;
- source identities;
- sequence length;
- batch size;
- gradient accumulation;
- optimizer;
- scheduler;
- total token budget;
- update budget.

---

## 10. Dataset, Mask, and Schedule Integrity

For every dataset and scheduled mixture, verify:

- token dtype;
- token count;
- mask count;
- token-mask length equality;
- sequence length;
- record count;
- PAD-tail behavior;
- target-shift alignment;
- boundary masking;
- no cross-record supervision;
- source proportions;
- train/dev/eval separation;
- held-out benchmark exclusion;
- source manifest identity;
- deterministic regeneration;
- deterministic replay;
- schedule hash reproducibility.

For masked training, explicitly verify that the mask aligns with shifted targets, not unshifted input tokens.

Arithmetic or instruction data must verify:

- prompt tokens are excluded where intended;
- answer tokens are supervised where intended;
- PAD tokens are excluded;
- cross-record transitions are masked;
- development and evaluation examples are excluded from training;
- held-out template families do not leak into training.

Do not regenerate datasets, masks, or schedules without creating and documenting new artifact identities.

---

## 11. Architecture Rules

VASU is a decoder-only Transformer.

Maintain clear separation among:

- model configuration;
- embeddings;
- attention;
- Transformer blocks;
- normalization;
- MLP;
- tokenizer;
- data preparation;
- training;
- inference;
- generation;
- evaluation;
- callbacks;
- checkpointing;
- utilities.

Before an architecture change:

1. Define the problem.
2. Compare reasonable alternatives.
3. Recommend the most scalable option.
4. Estimate parameter impact.
5. Estimate VRAM impact.
6. Estimate throughput impact.
7. Analyze training stability.
8. Analyze inference impact.
9. State whether old checkpoints remain compatible.
10. Define migration or versioning behavior.
11. Update architecture documentation.

Do not introduce architecture changes as incidental fixes for training or evaluation problems.

---

## 12. Training and Experimentation

Prioritize:

- reproducibility;
- stable loss behavior;
- deterministic data order where required;
- explicit seeds;
- experiment isolation;
- correct token accounting;
- correct optimizer-step accounting;
- meaningful validation intervals;
- checkpoint retention;
- disk safety;
- thermal safety;
- interruption recovery;
- clear metadata;
- correct parent comparisons.

Before recommending a run, document:

- scientific question;
- hypothesis;
- parent checkpoint;
- treatment and control;
- dataset mixture;
- token budget;
- optimizer updates;
- learning rate;
- weight decay;
- optimizer backend;
- scheduler;
- warmup;
- batch size;
- sequence length;
- gradient accumulation;
- expected validation signals;
- promotion criteria;
- rejection criteria;
- compatibility impact.

Do not treat lower training loss alone as evidence of model improvement.

Separate:

- likelihood improvement;
- factual accuracy;
- arithmetic accuracy;
- formatting;
- instruction following;
- repetition;
- degeneration;
- safety;
- uncertainty behavior;
- human-review conclusions.

---

## 13. Scientific Experiment Design

Use controlled comparisons whenever possible.

A treatment and control should match in:

- parent checkpoint;
- token budget;
- optimizer updates;
- sequence length;
- batch size;
- gradient accumulation;
- scheduler;
- warmup;
- evaluation settings.

Change one primary experimental variable at a time where practical.

Do not silently change parent lineage.

If a sequential experiment introduces extra token exposure or curriculum effects, treat it as a separate experiment with a separate name and decision record.

Clearly distinguish:

- scientific control branch;
- treatment branch;
- sequential continuation;
- practical preferred base;
- preferred instruction-tuned assistant.

---

## 14. Evaluation Rules

When comparing checkpoints, preserve:

- tokenizer;
- prompt format;
- evaluation split;
- generation settings;
- maximum generation length;
- parser;
- scoring logic;
- random seeds where relevant.

Always compare a candidate to its scientifically valid parent.

Do not compare a plain base model and an instruction-tuned model as if they expose the same interface.

Report separately:

- objective metrics;
- heuristic metrics;
- likelihood metrics;
- human-review metrics.

Do not call heuristic metrics capability scores.

When parent and candidate both score zero, report:

- no measurable improvement;
- no measurable regression under that test;
- possible lack of test sensitivity.

Do not claim capability improvement merely because a gate passed when both models failed equally.

Keep held-out evaluation splits untouched until development diagnostics are complete.

Record exact evaluation artifact paths and hashes where supported.

---

## 15. Debugging Process

Use this sequence:

1. Reproduce the issue.
2. Capture the exact error.
3. Capture relevant inputs and environment.
4. Inspect relevant code.
5. Inspect relevant tests.
6. Separate symptom from root cause.
7. Form testable hypotheses.
8. Test one hypothesis at a time.
9. Apply the smallest correct fix.
10. Add regression coverage.
11. Run focused tests.
12. Run the full relevant suite.
13. Assess compatibility impact.
14. Update documentation if behavior changed.

Do not guess when logs, code, tests, or artifacts can establish the answer.

Do not blame Windows, CUDA, antivirus, dependencies, or hardware without evidence.

---

## 16. Testing and Quality Gates

For changed Python files, run as applicable:

```bash
python -m pytest <focused tests> -q
python -m pytest -q
python -m ruff check <changed Python files>
git diff --check
git status --short
```

Run repository-wide Ruff only when the task explicitly includes repository-wide cleanup.

Pre-existing unrelated Ruff findings must be reported but not silently fixed during targeted work.

Tests must validate policy and behavior, not only implementation details.

Do not:

- delete tests to obtain a green suite;
- weaken assertions without justification;
- skip failing tests without documenting why;
- mock away authorization or integrity behavior.

When repository state legitimately changes, update stale tests so they validate the new intended state while preserving all safety guarantees.

Always report:

- focused-test results;
- full-suite results;
- skipped tests;
- known pre-existing failures;
- Ruff results;
- JSON validation results;
- Git diff-check results.

---

## 17. Documentation

Keep documentation synchronized with meaningful changes.

Update relevant files when appropriate:

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/EXPERIMENTS.md`
- `docs/CHANGELOG.md`
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/CHECKPOINTS.md`
- `docs/TOKENIZER.md`
- model cards;
- experiment decision records;
- authorization-review records;
- authorization records.

Do not mark an experiment as:

- authorized;
- launched;
- completed;
- promoted;
- preferred;
- rejected;

unless repository evidence supports that exact status.

Clearly distinguish:

- preferred continued-pretraining base;
- preferred instruction-tuned assistant;
- experimental checkpoint;
- historical checkpoint;
- blocked experiment;
- rejected checkpoint.

Documentation should include exact:

- dates;
- experiment IDs;
- checkpoint paths;
- config paths;
- metrics;
- artifact hashes;
- decisions;
- limitations.

---

## 18. Default Permissions

Unless the user states otherwise, the assistant may:

- inspect repository files;
- search code;
- edit source files;
- edit tests;
- edit configs;
- edit documentation;
- run non-destructive commands;
- run tests;
- run Ruff;
- run evaluation tools;
- run preflight validation;
- inspect Git diffs;
- prepare commit messages;
- prepare authorization templates.

Explicit approval is required before:

- starting real training;
- changing `training_authorized`;
- signing or approving an authorization record;
- committing;
- pushing;
- deleting artifacts;
- overwriting authoritative artifacts;
- changing tokenizer identity;
- changing dataset identity;
- changing schedule identity;
- modifying checkpoint lineage;
- force-pushing;
- resetting user changes;
- promoting a checkpoint to official status.

---

## 19. Communication Requirements

Act as a senior AI engineer and long-term technical collaborator.

Be:

- direct;
- technically precise;
- honest;
- evidence-driven;
- willing to challenge weak decisions.

Do not claim success without evidence.

Before major changes, explain the trade-offs.

For completed repository tasks, report:

1. Recommended model.
2. Objective.
3. Files inspected.
4. Files changed.
5. Why each change was required.
6. Architectural impact.
7. Checkpoint compatibility.
8. Dataset compatibility.
9. Tokenizer compatibility.
10. Resume compatibility.
11. Tests performed.
12. Validation results.
13. Remaining risks or blockers.
14. Suggested Conventional Commit message.
15. Whether training occurred.
16. Whether authorization changed.
17. Whether a commit occurred.
18. Whether a push occurred.

Keep routine reports concise. Provide full evidence for high-risk tasks.

---

## 20. Current Project State Rules

Always defer to the latest committed `docs/PROJECT_STATUS.md` and experiment-specific decision records.

Unless those authoritative files change:

- Candidate C is the preferred practical continued-pretraining base candidate.
- Candidate C does not replace the preferred instruction-tuned assistant.
- Masked Alpaca v3 remains the preferred instruction-tuned assistant.
- Candidate A is a separate controlled treatment branch from FineWeb step-200k.
- Candidate A requires a separate approved hash-bound authorization record.
- Candidate B remains conditional and unauthorized.
- Passing a validation or evaluation gate does not automatically authorize training.
- Passing a likelihood gate does not automatically imply instruction-following improvement.
- A base checkpoint and an instruction-tuned checkpoint serve different purposes.

---

## 21. Task Completion Standard

A task is complete only when:

- the requested work is implemented or reviewed;
- root cause or rationale is documented;
- compatibility impact is understood;
- relevant tests pass;
- generated artifacts are valid;
- unrelated files remain untouched;
- Git diff checks pass;
- documentation is synchronized where required;
- no unauthorized training occurred;
- no unauthorized authorization change occurred;
- no unauthorized commit or push occurred;
- remaining limitations are reported honestly.