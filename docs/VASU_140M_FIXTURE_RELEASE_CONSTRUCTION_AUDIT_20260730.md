# VASU-140M Fixture Release Construction Audit

Status: engineering requirements proven; independent decision accepted.

Audit date: 2026-07-30

## Requirement matrix

| Requirement | Evidence | Result |
| --- | --- | --- |
| Fixture-only scope | Hard 30-example cap and caller-supplied inputs | Proven |
| Production-path isolation | Planned release and manifest destinations rejected | Proven |
| Immutable lineage | Exact plan, decision, family, model, tokenizer, and record identities | Proven |
| Deterministic replay | Two independent builds yield identical manifests and bytes | Proven |
| Record layout | `uint16[513]` tokens and `uint8[513]` masks | Proven |
| Mask correctness | Prompt/PAD excluded; response/EOS supervised | Proven |
| Split isolation | Existing strict record-contract validation reused | Proven |
| Non-overwrite | Existing destination rejected | Proven |
| Atomic publication | Completed staging directory renamed as one unit | Proven |
| Crash cleanup | Injected failure removes staging and publishes nothing | Proven |
| Artifact integrity | Size, SHA-256, layout, and unbound-file checks | Proven |
| No training semantics | No config, checkpoint, optimizer, schedule, or entry point | Proven |
| Independent construction review | Separate packet exists | Proven |
| Independent decision | GPT-5.5 accepted the exact fixture layer on 2026-07-30 | Proven |

## Engineering disposition

The fixture builder was independently accepted as fixture-only construction
evidence. It remains deliberately not a production release builder: the
planned VASU-140M paths are blocked, real source discovery is absent, and the
fixture cap prevents processing the 996-example plan.

Even an accepted review would authorize only retaining this test harness and
using its evidence to design a separately gated production builder. Production
release construction and all training remain unauthorized.
