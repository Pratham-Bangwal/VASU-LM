# VASU-140M short-answer contamination policy remediation

Status: **v2 implementation qualified; real-source rerun pending.**

The first real Wikimedia scan proved that automatically rejecting a source
document for an isolated short answer is not scientifically valid. The v1 rule
rejected 1,849 of 1,854 non-reserved Wikimedia parents because ordinary prose
contained common one-word answers or arithmetic values. An answer without its
prompt does not reveal the evaluation task.

The v2 streaming scanner therefore preserves every short-answer match as
hash-bound `audit_only` evidence. Exact prompt matches and eight-word fragments
remain blocking and quarantine the parent document. A document with both kinds
is rejected. This does not weaken prompt or fragment isolation and prevents
common vocabulary from making an otherwise usable corpus ineligible.

The original v1 report is preserved as negative evidence. Source admission,
release construction, model execution, checkpoint creation, and training remain
unauthorized.
