# VASU-140M Base-Pretraining Readiness Protocol

Status: **planning-only; all execution gates closed.**

## Purpose

The VASU-140M instruction seed v1 release is published, but no compatible
`vasu_140m_v1` base checkpoint exists. This protocol defines the evidence
required before proposing a first VASU-140M base-pretraining experiment. It
does not select a source, create a data release, set a schedule or training
configuration, create a checkpoint, or authorize training.

The architectural question remains narrow: can the frozen 140M
capacity/context family be trained and evaluated reproducibly at an acceptable
runtime cost, while preserving strict lineage and held-out evaluation
isolation? It is not a claim that the family is already better than VASU-60M.

## Immutable boundaries

| Boundary | Requirement |
|---|---|
| Family | `vasu_140m_v1`; 137,841,408 parameters |
| Model configuration SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Family SHA-256 | `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b` |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Context/data format | isolated 513-token records; no reuse of 257-token binaries as a 140M release |
| Parent checkpoint | none; the first base-pretraining run would initialize a new 140M lineage |
| Instruction seed v1 | published but excluded from base-pretraining source selection |
| Training authority | false until a future hash-bound authorization record is accepted |

VASU-31M and VASU-60M weights, optimizer state, schedules, and instruction
branches are not valid initialization substitutes. Candidate A remains a
VASU-60M scientific reference only.

## Readiness gates

Every gate below must pass before a base-pretraining experiment can be
proposed. Passing any individual gate does not authorize another gate or
training.

### 1. CUDA and operational qualification

Run a bounded, no-update VASU-140M CUDA/AMP qualification on the intended
hardware. It must record GPU/software identity, precision mode, peak allocated
and reserved memory, throughput, finite forward/backward behavior, gradient
norm behavior, temperature, power limits where available, and checkpoint I/O
time. The evidence must define conservative memory, thermal, disk-space, and
throughput acceptance limits before a schedule is selected.

The existing CPU, model-only checkpoint, and synthetic CPU exact-resume
qualifications are necessary but insufficient; they do not establish CUDA/AMP
or real-data behavior.

### 2. Base-pretraining source and record release

Prepare a separate, licensed, provenance-bound VASU-140M base-pretraining
source plan. It must identify each source revision, license, source hash,
normalization, filtering, contamination inventory, cross-source deduplication,
train/development/evaluation isolation, and deterministic 513-token packing.

Existing FineWeb 257-token binaries may inform a future source review but are
not automatically compatible. The published instruction seed is explicitly
out of scope for this base-pretraining release and must remain unchanged.

### 3. Frozen base-model evaluation contract

Before any optimizer update, freeze a VASU-140M base-model evaluation package
that separates likelihood, factuality, arithmetic, repetition/degeneration,
robustness, and manual review. It must pin prompt inventories, splits,
generation settings, scorers, seeds, parser versions, and evaluation artifact
schemas. Development diagnostics and held-out evaluation must be distinct.

The evaluation must compare only scientifically valid interfaces: base model to
base model under matching raw-continuation prompts. It must not present a
VASU-60M instruction checkpoint as a direct quality control for a 140M base
model.

### 4. Real-data exact-resume and checkpoint integrity

After a base-data release exists, demonstrate an interrupted, mid-accumulation
real-data run that exactly preserves model, optimizer, scheduler, scaler,
sampler position, partial gradients, RNG state, source identities, validation
state, and checkpoint SHA-256/size sidecars. Verify that an invalid family,
tokenizer, source, schedule, or configuration identity fails before loading or
resuming.

### 5. Scientific experiment plan and independent review

Only after gates 1–4 pass may a versioned experiment plan be written. It must
state one hypothesis, an appropriate control/comparison, source mixture,
token/update budget, learning rate, optimizer, scheduler, warmup, batch size,
accumulation, validation cadence, stop conditions, promotion/rejection
criteria, expected runtime envelope, and isolated output paths. It must bind
all source, record, tokenizer, evaluation, runtime, and repository identities.

GPT-5.5 must independently review that immutable plan and its evidence. A
review acceptance remains non-authorizing.

### 6. Final preflight and explicit authorization

The prospective runtime must then be clean and match the reviewed commit.
Preflight must verify the exact configuration, schedule, source release,
evaluation package, output isolation, disk/thermal safeguards, checkpoint
retention, and real-data resume evidence. Only a distinct human-approved,
self-consistent authorization record may permit one named training run.

## Alternatives considered

| Alternative | Decision |
|---|---|
| Start instruction tuning from Candidate A | Rejected: wrong family and invalid parent lineage. |
| Start a 140M run using legacy 257-token data | Rejected: incompatible context/record contract and unreviewed source lineage. |
| Select an external checkpoint | Deferred: would require a separate provenance, license, architecture, tokenizer, and checkpoint-identity review. |
| Build readiness evidence first | Recommended: resolves feasibility and reproducibility without an optimizer update. |

## Acceptance criteria for this protocol

This protocol is ready for independent review if it accurately preserves the
frozen family/tokenizer boundary, rejects incompatible parents and data, and
does not imply authorization. Acceptance only permits the listed readiness
work to be designed and reviewed separately; it permits neither a base-data
release nor any training action.
