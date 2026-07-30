# VASU-140M 513-Token Independent Review Packet

Status: independently reviewed and accepted; non-authorizing.

Date: 2026-07-30

## Decision requested

Accept or reject the frozen 513-token record and shifted-mask engineering
contract for a future, separately specified VASU-140M production-data release.
Acceptance permits only later source, logical-data, and immutable release
review. It does not permit production data construction or training.

## Requirement-to-evidence audit

| Requirement | Evidence | Result |
| --- | --- | --- |
| Family and context identity | Frozen family/config identities and 512/513 contract | Pass |
| Tokenizer identity and special IDs | Real tokenizer audit in smoke and focused test | Pass |
| Stable prompt/response boundary | Full encoding must begin with independently encoded prompt | Pass |
| Correct shifted-target alignment | `loss_mask=stored_mask[1:]`; transition assertion | Pass |
| Prompt and PAD exclusion | Logical and packed validators plus tests | Pass |
| Response and EOS supervision | Logical and packed validators plus tests | Pass |
| Cross-example exclusion | Every example begins with stored mask zero | Pass |
| Fixed shape and dtypes | `uint16[513]` tokens and `uint8[513]` masks | Pass |
| No truncation or splitting | Overlength rejection and complete-example packer | Pass |
| Split isolation | Global ID and semantic-hash uniqueness | Pass |
| Deterministic rebuild | Two builds match the frozen fixture report | Pass |
| Frozen report integrity | Schema/lineage checks plus exact canonical-identity pinning and rehashed-mutation rejection | Pass |
| No production path | Caller-owned examples; smoke only prints fixture evidence | Pass |
| No authorization semantics | Specification and report state false | Pass |

The immutable fixture report SHA-256 is
`7329698aefc3ab4593b1bfba246487a83ed0210b4e3fbe96f72c8bb7b8c8e7e3`.

## Reviewer questions

1. Is complete-example, input-order packing the correct trade-off for
   deterministic lineage and masking integrity?
2. Are semantic-hash uniqueness and stable IDs sufficient generic split
   invariants, with source-specific leakage rules added later?
3. Are stored masks versus shifted trainer masks unambiguous and adequately
   tested?
4. Does the contract fail closed on every silent-target corruption path?
5. Are the non-authorizations and future release gates explicit enough?

## Explicit exclusions

This packet contains no source approval, production count, sampling seed,
license decision, mixture, schedule, model-quality hypothesis, compute budget,
training configuration, or authorization.
