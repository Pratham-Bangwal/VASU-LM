# VASU-140M Instruction Seed Release Plan

Status: frozen specification; independently accepted for future release
construction review; non-authorizing.

Date: 2026-07-30

## Scope

`vasu_140m_instruction_seed_v1` is a specification-only plan for a future
response-masked instruction release. It is not a base-pretraining corpus. The
accepted 513-token contract represents prompt/response examples with
assistant-only loss, so FineWeb, Wikimedia, or other full-loss text must not be
silently routed through it.

The plan requires a compatible trained VASU-140M base checkpoint before the
release could be used in an experiment. No such checkpoint is selected or
authorized. The plan creates no logical release manifest, token binary, mask
binary, schedule, training configuration, checkpoint, or optimizer update.

## Frozen identities

| Field | Identity |
| --- | --- |
| Plan ID | `vasu_140m_instruction_seed_v1` |
| Plan schema | `vasu.model-family-release-plan.v1` |
| Canonical plan SHA-256 | `8da8cc92ef913bb567c96ec47986f849b84eb85ac2d11f3ad245fb1639361e16` |
| Formatted plan file SHA-256 | `f61faf9793bc9e4733708bb65246ba9b19374b8e50fe80f3a702ff94139d3868` |
| Qualification report schema | `vasu.model-family-release-plan-report.v1` |
| Canonical report SHA-256 | `8147215c2f58044aa60374f39e772e13fce3913090ea5c2d8acc915aeeca6fcb` |
| Formatted report file SHA-256 | `21e98f69ead396b480b46156da308c50153205342b6a9c711ff657999c2e32aa` |
| Record specification SHA-256 | `1bbbeafe836104dc4326a3a00e0dfcef21a965bbc02f56408be4305ae16ce7e5` |
| Record width | 513 |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |

The machine-readable plan is
`configs/data/releases/vasu_140m_instruction_seed_v1.plan.json`. Any
identity-bearing change creates a new plan version and review decision.

## Source selection and license

The plan references two existing, human-approved, purpose-written sources:

| Source | Examples | License | Manifest SHA-256 |
| --- | ---: | --- | --- |
| `vasu_instruction_quality_v1_batch_001` | 500 | CC0-1.0 | `5a9060222f07c8977d6ac0e741a22aaedf08af76bfe1b4b1e33b764b0fce8a86` |
| `vasu_instruction_quality_v1_batch_002` | 500 | CC0-1.0 | `c38f208e4bb1bd6e608fa19e285ed2357b7160eed67354422560314911f117f0` |

Every source record has `source_type=human_written`, English language, the
expected purpose-written provenance, and CC0-1.0 metadata. Every one of the
1,000 source IDs has a hash-bound human approval. Source, review, manifest,
tokenizer, and record-contract identities are verified before qualification.

The rejected Batch 002 model-promotion result does not invalidate its reviewed
data. This plan selects data artifacts, not the rejected checkpoint.

## Combined quality and deduplication

The combined sources have:

- zero schema or review-binding findings;
- zero exact duplicate groups across instruction, input, response, and
  combined normalized views;
- zero normalized-trigram Jaccard candidates at threshold 0.85;
- zero semantic-hash collisions among eligible examples.

Deduplication is global across both sources and later across all three planned
splits. A per-source pass alone is insufficient.

## Evaluation contamination quarantine

The plan pins 2,618 prompts across ten authoritative inventories:

- 8 from `vasu_capability_v1`;
- 300 from `factual_cpt_v2`;
- 216 from `ultrachat_promotion_v1`;
- 40 from the current fixed checkpoint-comparison prompt set;
- 8 from its preserved original subset;
- 6 from the factual-CPT v1 prompt set;
- 40 from verified-arithmetic v1 development/evaluation; and
- 2,000 from verified-arithmetic v2 development/evaluation.

The first engineering review rejected the narrower three-file inventory because
it omitted other authoritative held-out prompts. The expanded read-only
comparison found four exact prompt matches:

| Source example | Frozen benchmark item | Disposition |
| --- | --- | --- |
| `viq1_b001_000006` | `factual_cpt_v2:mc_046` | Quarantine |
| `viq1_b001_000090` | `factual_cpt_v2:mc_054` | Quarantine |
| `viq1_b002_000008` | `factual_cpt_v2:mc_066` | Quarantine |
| `viq1_b002_000132` | `prompts.json:loop_simple` | Quarantine |

The matched questions concern Brazil's continent, the expansion of CPU, and
what a URL identifies, plus an explanation of programming loops. Their content
is not defective; they are excluded solely to preserve evaluation
independence. After quarantine, 996 examples remain with zero full-prompt and
zero eight-word-fragment matches.

Changing any benchmark inventory requires a new inventory hash, contamination
scan, assignment identity, plan identity, and review.

## Deterministic split policy

The plan does not reuse source validation examples for training:

1. Preserve all 48 eligible source-native validation examples as development.
2. Group the remaining source-native training examples by source and
   capability.
3. Allocate exactly 50 evaluation examples by largest remainder across those
   strata.
4. Within each stratum, rank by
   `SHA256("140513:<source_id>:<capability>:<example_id>")`.
5. Assign the held-out prefix to evaluation and the remainder to training.

Frozen counts are 898 train, 48 development, and 50 evaluation. The assignment
SHA-256 is
`59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394`.
No assignment file or production logical manifest has been created.

## Future release gates

Before construction, a separate reviewer must accept this exact plan. A later
builder must then:

1. bind the exact plan, source, review, benchmark, tokenizer, and record
   identities;
2. serialize the existing Alpaca-compatible `User:` / `Assistant:` prompt and
   leading-space response without truncation;
3. prove tokenizer prefix boundaries for every eligible example;
4. rebuild the frozen ID assignment exactly;
5. re-run combined deduplication and contamination checks;
6. produce isolated `uint16[513]` token and `uint8[513]` stored-mask artifacts;
7. validate shifted-target, prompt, EOS, cross-example, PAD, and split
   invariants;
8. publish atomically to the planned versioned paths without overwrite; and
9. obtain a separate immutable-release decision.

Even a future approved data release would not authorize training. Training
would require a compatible base checkpoint, written hypothesis, evaluation
plan, reviewed configuration, clean repository, successful preflight, and
exact human authorization.
