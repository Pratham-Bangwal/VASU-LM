# Candidate E Reference Tokenizer Audit

Status: evidence only; non-authorizing; no Candidate E release was generated.

## Scope

The Candidate E masked-target compiler was exercised against the authoritative
`assets/tokenizer.json` over the full existing arithmetic-v2 generation
envelope: 32,000 train, 1,000 development, and 1,000 evaluation records. This
is a serialization and capacity check, not approval to reuse arithmetic-v2
splits for Candidate E.

## Result

Every one of the 34,000 records compiled successfully for both variants. In
particular, the common `Question ... Response` prefix remained an exact token
prefix of the full serialization in every case. The compiler would have failed
closed if a tokenizer merge had crossed that target boundary.

| Reference split | Examples | Control length (min / mean / max) | Treatment length (min / mean / max) |
| --- | ---: | ---: | ---: |
| train | 32,000 | 18 / 29.91 / 45 | 24 / 39.83 / 59 |
| development | 1,000 | 26 / 32.83 / 42 | 33 / 42.96 / 56 |
| evaluation | 1,000 | 26 / 33.12 / 42 | 33 / 43.48 / 56 |

All observed examples fit within the fixed 257-token packed-record width.

## Supervised-token evidence

| Reference split | Control supervised response/EOS tokens | Treatment supervised response/EOS tokens |
| --- | ---: | ---: |
| train | 194,403 | 512,043 |
| development | 6,171 | 16,297 |
| evaluation | 6,267 | 16,633 |

The treatment's longer supervised target confirms that matching example count
would not match optimization-token exposure. A future approved protocol must
select and freeze an equal-token/update accounting rule before release or
training.

## Boundary

Candidate E requires new stable IDs, operand ranges, template families, and
split identities. This reference audit does not establish those choices, does
not assess split isolation against future releases, and does not authorize data
generation, training configuration, or training.
