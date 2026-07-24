# VASU-60M Capability-CPT Scientific Authorization Packet

Status: **technically ready for explicit Candidate C review; training is not
authorized**

Prepared against repository commit
`79e3e2b30b28145aea598d1980a7375c07577809`. The working tree was not clean
at preflight: `docs/EVALUATION.md` and `vasu/training/resume_state.py` were
modified, and `tests/test_verified_arithmetic_serializer.py` was untracked.
Those pre-existing changes were preserved. The capability-runtime work is also
uncommitted, so an authorized launch is intentionally blocked until this exact
change set is reviewed and committed.

## A. Repository and artifact preflight

| Item | Verified value | Result |
|---|---|---|
| Parent checkpoint | `checkpoints/vasu_60m/milestones/fineweb_step_200000.pt` | present |
| Parent SHA-256 | `88688ee85fc880967dafb322277565d4754e049a22c0d8e86060991438436a2f` | match |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` | match |
| Candidate C manifest SHA-256 | `e1279efbf30996c4742471484b7636633cc67fbddf715ee7d113b396349391e7` | match |
| Candidate C schedule SHA-256 | `92f8675a6160c18e18a0e99cbb3fedccb1c4bc4b322055d5cd5b49602f54d381` | match |
| Candidate A manifest SHA-256 | `bda898d742d44123f647cd318aedc2bf28dedb24986da688716aee2778cb0ba9` | match |
| Candidate A schedule SHA-256 | `cc9cd9b7d2a2cd55a91c805691ce1c872fac82cd6295c5626383f8aea2ca15ce` | match |
| CUDA validation SHA-256 | `3555983cc9bcd58e6df70592f21c6b6c0ff064208b342bbc6f4cdc88eab05268` | match |
| Free space on `D:` | 49.99 GiB at final preflight | sufficient for the bounded-retention policy |
| Candidate C/A/B output directories | absent | no conflict |

All source identities in the C and A manifests were recomputed:

| Source | Token SHA-256 | Mask SHA-256 | Manifest SHA-256 |
|---|---|---|---|
| FineWeb extension replay | `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e` | none | `d6f1d112fbd0135563c173feaf700162982718a6ced6abd537df0e749c267a00` |
| Wikimedia factual release | `fee0b88832fc569ce73393bc232e29a80cb981e31779ce4c42129d15a0be876f` | none | `cec126ae4e804a4f42699f96625752199c73b0caafbc4bf8b6d3bc3f47560a81` |
| Verified arithmetic v2 | `9abbab5e6a3d6d1121d47a89e9e829a3b51b072af2dc3a9daed5bae5d6fe474a` | `beaf4d1f5dc325d1fc2a37cf8adc6677738e009fdf68d87fbf1bb032013098c4` | `8da3a1acfb2d2481c295b0d226c4695613b56670989d001fa27e65bed366115c` |

Replay safety passes without overrides:

- C: FineWeb 0.1694 effective passes, Wikimedia 0.9402; maximum reuse 1.
- A: FineWeb 0.1601 effective passes, Wikimedia 0.9402, arithmetic 1.1981;
  arithmetic maximum reuse 2.
- The configured warning and hard limits remain 5 and 10 passes.

The CUDA report records an RTX 4050 Laptop GPU, PyTorch 2.11.0+cu128, AMP,
batch size 2, 16 accumulated microbatches, finite losses and gradients, one
successful standard-AdamW update per candidate, 1,162.21 MiB peak allocation,
and exact-resume parity at optimizer, mid-accumulation, source-transition, and
arithmetic-replay boundaries.

## B. Exact token and step accounting

C and A have identical outer accounting:

| Quantity | Exact value |
|---|---:|
| Schedule records | 78,144 |
| DataLoader batches / microbatches | 39,072 |
| Records per microbatch | 2 |
| Tokens per record | 256 target tokens |
| Tokens per microbatch | 512 |
| Target tokens | 20,004,864 |
| Gradient accumulation | 16 microbatches |
| Effective batch | 32 records / 8,192 tokens |
| Optimizer updates | 2,442 |
| Warmup updates | 49 |
| Scheduler updates | 2,442 |
| Final partial accumulation | none |

`39,072 / 16 = 2,442` exactly. The last optimizer update therefore contains
all 16 microbatches and all 8,192 tokens.

With `checkpoint_interval=200`, checkpoint events occur at steps 200 through
2,400. The production policy retains the milestone at step 200 and the two
newest non-milestone periodic checkpoints. It separately protects
`latest.pt`, `final.pt`, and the three domain-best checkpoints. A representative
optimizer-boundary VASU-60M checkpoint is 667.74 MiB (the configuration
conservatively budgets 700,175,393 bytes). Including the three retained
periodics, five protected boundary checkpoints, one possible
mid-accumulation temporary checkpoint, and a 10 GiB safety margin, preflight
requires about 16.1 GiB. The check is repeated before every atomic save.

Historical FineWeb block logs used 18.75 MiB for 241 files and 47,520 logged
steps. A single-process 2,442-step run should be far smaller; reserve 25 MiB
per candidate for text/JSON logs until the production logger is measured.

## C. Runtime estimate

The closest measured evidence is the step-152,400-to-200,000 VASU-60M run on
the same RTX 4050, batch size 2, context 256, accumulation 16, AMP, and
standard AdamW. Across 237 complete 200-step blocks:

- minimum: 0.7836 seconds/update;
- median: 0.8490 seconds/update;
- maximum: 1.1372 seconds/update.

These measurements included checkpointing every 10 steps and periodic
FineWeb validation, so they are conservative for a 200-step checkpoint
cadence. Applied to 2,442 updates:

| Estimate | One candidate | C plus A |
|---|---:|---:|
| Optimistic (observed minimum) | 31.9 min | 63.8 min |
| Realistic (observed median) | 34.5 min | 69.1 min |
| Conservative (observed maximum) | 46.3 min | 92.6 min |

Blocks with a FineWeb validation result had a 1.70-second higher median than
blocks without one. Twenty-five events (steps 100 through 2,400 plus final
step 2,442) would therefore add about 42.5 seconds **for the measured
FineWeb-sized validation only**. Wikimedia validation and arithmetic
generation overhead have not been measured and must not be invented.
Checkpoint cost is already embedded in the observed runtime envelope but is
not isolated. Plan on roughly 35–50 minutes of core training per candidate
plus the separately measured multi-domain interval and post-run evaluation.

## D. Hyperparameter review

- **Learning rate 1e-5:** appropriate. It is one tenth of the historical
  1e-4 base-pretraining rate and limits destructive drift during a short
  continuation from step 200,000.
- **Batch size 2, accumulation 16:** appropriate and CUDA-validated. It gives
  8,192 target tokens per optimizer update without exceeding laptop VRAM.
- **Sequence length 256:** required for checkpoint/model compatibility and
  matches the packed schedule contract.
- **Optimizer backend `standard`:** retain. It is the default and the backend
  used by the validation artifact. Fused AdamW remains opt-in.
- **Weight decay 0.1:** retain the established base-pretraining regularization.
- **Warmup 49 updates:** appropriate as `ceil(2442 * 0.02)`, despite loading
  trained weights, because optimizer state intentionally starts fresh.
- **Cosine horizon 2,442; minimum LR 1e-6:** internally consistent and avoids
  a stale scheduler horizon.
- **Gradient clipping 1.0:** appropriate and hard-coded by the launcher.
- **AMP:** appropriate and hard-coded on; the exact CUDA smoke passed.
- **Validation every 100:** implemented after successful optimizer updates,
  plus the final update. Events contain separate FineWeb loss, Wikimedia loss,
  and a deterministic 64-example arithmetic-development proxy.
- **Checkpoint every 200:** implemented at optimizer boundaries with verified
  atomic writes, SHA-256 sidecars, bounded retention, and explicit recovery.

## E. Candidate C evaluation gates

Candidate C is a new base-model control, not an assistant candidate. Compare
the parent, `final.pt`, `latest.pt`, all three domain-best checkpoints, early
`step_200.pt`, and the newest retained periodic checkpoint.

Required measurements:

1. FineWeb and held-out Wikimedia validation loss with identical subsets.
2. Verified arithmetic v2 development and evaluation exact accuracy.
3. Capability-v1 greedy objective categories and separate heuristics.
4. Repeated-bigram ratio, repeated-trigram ratio, empty-output rate, response
   length, and fixed general continuation outputs.
5. Checkpoint strict load, finite tensors, optimizer/scheduler/scaler state,
   internal step, and SHA-256.
6. Wall time, peak CUDA memory, maximum temperature, optimizer skips, and
   tokens/source records consumed.

The configured continued-pretraining gate is inclusive:

- each targeted category (`arithmetic`, `factual`) must have candidate minus
  parent objective accuracy `>= 0`;
- each general category (`formatting`, `safety_uncertainty`) must have parent
  minus candidate accuracy `<= 0.02`;
- missing categories block promotion.

Additional scientific requirements:

- either Wikimedia validation loss improves from the preserved parent baseline
  of 3.393544 or factual objective accuracy improves;
- FineWeb loss must not regress by more than the established 3% guardrail:
  from 3.356130 to at most 3.456814;
- no empty/degenerate output;
- repetition metrics must be reported independently and manually reviewed
  because the current suite has no configured numeric repetition gate;
- all configured promotion gates pass.

Candidate C need not replace the preferred instruction-tuned assistant.

## F. Candidate A evaluation gates

Primary comparison is A versus C; the step-200,000 parent remains a secondary
reference. Evaluate the same checkpoint positions and all C metrics, plus
verified arithmetic v2 dev/eval exact accuracy.

A passes only when:

- dev and eval arithmetic exact accuracy both exceed C;
- the arithmetic improvement is not explained only by capability-v1's two
  arithmetic fixtures;
- FineWeb loss remains within the same 3% parent guardrail;
- factual accuracy and Wikimedia loss do not materially regress from C;
- the configured continued-pretraining gate passes;
- repetition does not materially worsen on fixed prompts;
- manual review finds no template echo, answer-only degeneration on general
  prompts, operand copying, or synthetic-format leakage.

No single training-loss value can satisfy these gates. A numeric
“material arithmetic improvement” threshold must be preregistered after the
exact-answer evaluator reports uncertainty; it is not silently invented here.

## G. Candidate B conditional gate

Candidate B remains unauthorized and its boolean stays false. It can receive a
separate authorization review only after all of the following:

1. A completes with intact hashes, exact accounting, finite state, and a valid
   resumable checkpoint.
2. A passes technical integrity and the C-versus-A evaluation.
3. A shows a real arithmetic improvement, but the result is explicitly judged
   insufficient for the experiment objective.
4. Broad/factual regression remains within preregistered limits.
5. No arithmetic replay, memorization, or template-degeneration warning is
   open.
6. Human review covers representative arithmetic and general continuations.
7. A written B-specific rationale and approver/date record is created.

## H. Checkpoint-selection policy

- Periodic milestones: every 200 successful optimizer updates.
- Early diagnostic: `step_200.pt`.
- Recovery checkpoint: explicit, validated `latest.pt` or a named retained
  periodic checkpoint; implicit discovery is forbidden.
- Final checkpoint: `final.pt` after the epoch transition; it is not resumable.
- Domain-best checkpoints: `best_fineweb.pt`, `best_wikimedia.pt`, and
  `best_arithmetic.pt`. Arithmetic ties prefer lower malformed rate, then
  shorter evaluation duration.
- Evaluation set: `final.pt`, `latest.pt`, all three domain-best checkpoints,
  `step_200.pt`, and the newest retained periodic checkpoint.

No opaque mixed score or arithmetic training loss selects a checkpoint.

## I. Abort criteria

Abort and preserve the last validated checkpoint on:

- non-finite training/validation loss or non-finite gradients;
- any repeated optimizer skip (one isolated AMP overflow is recorded and
  reviewed; two consecutive or three total skips abort);
- parent, tokenizer, manifest, source, schedule, or resume-identity mismatch;
- FineWeb validation loss above 1.03 times the parent baseline at any scheduled
  validation event;
- Wikimedia validation loss above 1.10 times its parent baseline;
- arithmetic dev improves while general/factual outputs show obvious template
  leakage or answer-format degeneration;
- any CUDA OOM (the run manifest records the abort; no automatic retry);
- two consecutive readings at or above 87 C, or one critical reading at
  90 C; warnings begin at 82 C;
- failed atomic save, corrupt/missing recovery checkpoint, or low-disk guard;
- resumed sample identity, sampler offset, accumulation state, optimizer,
  scheduler, GradScaler, RNG, or global step diverging from the checkpoint.

Do not abort for ordinary batch-loss noise.

## J. Authorization recommendation

Candidate C is **technically and scientifically ready for explicit
authorization after review and a clean commit**. The runtime now provides
successful-update interval validation, durable validation-event resume,
separate domain-best checkpoints, explicit exact resume, atomic verified
checkpoint writes, corrupt-checkpoint rejection, bounded retention, disk
preflight/checkpoint guards, and the 82/87/90 C thermal policy. The full
arithmetic-v2 evaluator is deterministic, hash-bound, and persists every
completed example so interrupted evaluation resumes without duplication.

Training remains blocked now because the working tree is intentionally dirty
and `training_authorized` remains `false`. An approver must review the exact
commit, complete the authorization record, then make only the preregistered
boolean edit. Candidate A still requires a separate authorization after C is
evaluated. Candidate B remains blocked pending the evidence listed in section
G.
