"""Two-phase, authorization-gated VASU-140M production release tooling.

Qualification is read-only. Publication requires an exact one-build
authorization and never creates schedules, training configs, or checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from vasu.data.instruction_quality import load_jsonl
from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.data.vasu_140m_records import (
    EXPECTED_SPLITS,
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    TOKENIZER_SHA256,
    LogicalExample,
    PackedRecord,
    compile_text_example,
    pack_split,
    sha256_json,
    validate_split_isolation,
)
from vasu.data.vasu_140m_release_plan import (
    EXPECTED_SPLIT_COUNTS,
    FROZEN_PLAN_SHA256,
    PLAN_ID,
    _derive_assignments,
    load_release_plan,
    sha256_file,
    validate_release_plan,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


QUALIFICATION_SCHEMA_ID = "vasu.production-release-qualification.v1"
INTERNAL_MANIFEST_SCHEMA_ID = "vasu.production-release-internal-manifest.v1"
EXTERNAL_MANIFEST_SCHEMA_ID = "vasu.production-release-manifest.v1"
EXTERNAL_MANIFEST_TEMPLATE_SCHEMA_ID = (
    "vasu.production-release-manifest-template.v1"
)
AUTHORIZATION_SCHEMA_ID = "vasu.production-release-authorization.v1"
RECEIPT_SCHEMA_ID = "vasu.production-release-authorization-receipt.v1"
PLAN_DECISION_SHA256 = (
    "4522d1c36a73700da9f3eec7cdd87561fded1a8f8661fa0ef88ab18a89058890"
)
PLAN_DECISION_PATH = (
    "docs/VASU_140M_INSTRUCTION_SEED_RELEASE_INDEPENDENT_REVIEW_DECISION.md"
)
FIXTURE_DECISION_PATH = (
    "docs/VASU_140M_FIXTURE_RELEASE_CONSTRUCTION_"
    "INDEPENDENT_REVIEW_DECISION_20260730.md"
)
DESIGN_DECISION_PATH = (
    "docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_DESIGN_"
    "INDEPENDENT_REVIEW_DECISION_20260730.md"
)
FIXTURE_DECISION_SHA256 = (
    "96f517d54e6808fb380454506353db21199fd2b5655a344e74da00e240e520fb"
)
DESIGN_DECISION_SHA256 = (
    "c067cb691564b2108c18ebac763b10c1ee0c5044fa89abff43cdc47c01304013"
)
PRODUCTION_RELEASE_PATH = "data/processed/vasu_140m/instruction_seed_v1"
PRODUCTION_MANIFEST_PATH = "data/manifests/vasu_140m/instruction_seed_v1.json"
RECEIPT_DIRECTORY = "data/manifests/vasu_140m/authorization_receipts"
EXPECTED_ASSIGNMENT_SHA256 = (
    "59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394"
)
_ARTIFACT_NAMES = {
    f"{split}.{suffix}.bin"
    for split in EXPECTED_SPLITS
    for suffix in ("tokens", "mask")
} | {"logical_records.jsonl", "internal_manifest.json"}
_AUTHORIZATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class PreparedRelease:
    """Deterministic in-memory release bytes plus qualification evidence."""

    artifacts: Mapping[str, bytes]
    internal_manifest: Mapping[str, object]
    external_manifest: Mapping[str, object]
    qualification: Mapping[str, object]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_evidence(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _decision_identities(repository_root: Path) -> dict[str, str]:
    identities = {
        "plan_decision_sha256": sha256_file(repository_root / PLAN_DECISION_PATH),
        "fixture_decision_sha256": sha256_file(repository_root / FIXTURE_DECISION_PATH),
        "design_decision_sha256": sha256_file(repository_root / DESIGN_DECISION_PATH),
    }
    expected = {
        "plan_decision_sha256": PLAN_DECISION_SHA256,
        "fixture_decision_sha256": FIXTURE_DECISION_SHA256,
        "design_decision_sha256": DESIGN_DECISION_SHA256,
    }
    if identities != expected:
        raise ValueError("independent decision identity mismatch")
    return identities


def _logical_lines(
    splits: Mapping[str, Sequence[LogicalExample]],
    lineage: Mapping[str, Mapping[str, str]],
    packed: Mapping[str, Sequence[PackedRecord]],
) -> bytes:
    rows: list[dict[str, object]] = []
    for split in EXPECTED_SPLITS:
        span_locations = {
            span.example_id: {
                "packed_record_index": record_index,
                "start": span.start,
                "end": span.end,
                "target_start": span.target_start,
            }
            for record_index, record in enumerate(packed[split])
            for span in record.spans
        }
        for example in splits[split]:
            source = lineage[example.example_id]
            rows.append(
                {
                    "example_id": example.example_id,
                    "split": split,
                    "source_id": source["source_id"],
                    "capability": source["capability"],
                    "semantic_sha256": example.semantic_sha256,
                    **span_locations[example.example_id],
                }
            )
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def qualify_compiled_release(
    *,
    splits: Mapping[str, Sequence[LogicalExample]],
    lineage: Mapping[str, Mapping[str, str]],
    decision_identities: Mapping[str, str],
    repository_commit: str,
    implementation_sha256: str,
    enforce_production_counts: bool = True,
    source_audit: Mapping[str, object] | None = None,
) -> PreparedRelease:
    """Create deterministic qualification evidence without filesystem writes."""

    counts = validate_split_isolation(splits)
    if enforce_production_counts and counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError("compiled split counts do not match the production plan")
    example_ids = {
        example.example_id for split in EXPECTED_SPLITS for example in splits[split]
    }
    if set(lineage) != example_ids:
        raise ValueError("lineage must cover every compiled example exactly")
    if len(repository_commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase Git identity")
    expected_decisions = {
        "plan_decision_sha256",
        "fixture_decision_sha256",
        "design_decision_sha256",
    }
    if set(decision_identities) != expected_decisions:
        raise ValueError("decision identities are incomplete")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in decision_identities.values()
    ):
        raise ValueError("decision identities must be lowercase SHA-256 values")
    assignment_rows = [
        {
            "example_id": example.example_id,
            "source_id": lineage[example.example_id]["source_id"],
            "split": split,
        }
        for split in EXPECTED_SPLITS
        for example in splits[split]
    ]
    assignment_sha256 = sha256_json(
        sorted(assignment_rows, key=lambda row: str(row["example_id"]))
    )
    if enforce_production_counts and assignment_sha256 != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("compiled assignment identity does not match the production plan")
    if len(implementation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in implementation_sha256
    ):
        raise ValueError("implementation_sha256 must be a lowercase SHA-256")

    packed = {
        split: pack_split(splits[split], split=split) for split in EXPECTED_SPLITS
    }
    artifacts: dict[str, bytes] = {}
    split_reports: dict[str, object] = {}
    for split in EXPECTED_SPLITS:
        token_bytes = b"".join(record.tokens.tobytes() for record in packed[split])
        mask_bytes = b"".join(record.stored_mask.tobytes() for record in packed[split])
        token_name = f"{split}.tokens.bin"
        mask_name = f"{split}.mask.bin"
        artifacts[token_name] = token_bytes
        artifacts[mask_name] = mask_bytes
        split_reports[split] = {
            "logical_example_count": len(splits[split]),
            "packed_record_count": len(packed[split]),
            "used_token_count": sum(record.used_token_count for record in packed[split]),
            "supervised_token_count": sum(
                int(record.stored_mask.sum()) for record in packed[split]
            ),
            "tokens": {"path": token_name, **_artifact_evidence(token_bytes)},
            "stored_mask": {"path": mask_name, **_artifact_evidence(mask_bytes)},
        }
    logical_bytes = _logical_lines(splits, lineage, packed)
    artifacts["logical_records.jsonl"] = logical_bytes
    internal_body: dict[str, object] = {
        "schema_id": INTERNAL_MANIFEST_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_width": RECORD_WIDTH,
        "repository_commit": repository_commit,
        "implementation_sha256": implementation_sha256,
        "decision_identities": dict(decision_identities),
        "split_counts": counts,
        "assignment_sha256": assignment_sha256,
        "source_audit": dict(
            source_audit
            or {
                "mode": "caller_supplied_fixture",
                "decoded_round_trip_count": 0,
                "capability_split_counts": {},
            }
        ),
        "splits": split_reports,
        "logical_records": {
            "path": "logical_records.jsonl",
            **_artifact_evidence(logical_bytes),
        },
        "requires_external_manifest": True,
        "training_authorized": False,
    }
    internal_body["manifest_sha256"] = sha256_json(internal_body)
    internal_bytes = _json_bytes(internal_body)
    artifacts["internal_manifest.json"] = internal_bytes
    external_body: dict[str, object] = {
        "schema_id": EXTERNAL_MANIFEST_TEMPLATE_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "release_directory": PRODUCTION_RELEASE_PATH,
        "internal_manifest": {
            "path": f"{PRODUCTION_RELEASE_PATH}/internal_manifest.json",
            **_artifact_evidence(internal_bytes),
        },
        "repository_commit": repository_commit,
        "implementation_sha256": implementation_sha256,
        "training_authorized": False,
    }
    external_body["template_sha256"] = sha256_json(external_body)
    qualification_body: dict[str, object] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "repository_commit": repository_commit,
        "implementation_sha256": implementation_sha256,
        "decision_identities": dict(decision_identities),
        "split_counts": counts,
        "assignment_sha256": assignment_sha256,
        "source_audit": internal_body["source_audit"],
        "artifact_evidence": {
            name: _artifact_evidence(payload)
            for name, payload in sorted(artifacts.items())
        },
        "external_manifest_template_sha256": external_body["template_sha256"],
        "production_release_created": False,
        "training_authorized": False,
        "release_build_permitted": False,
    }
    qualification_body["qualification_sha256"] = sha256_json(qualification_body)
    prepared = PreparedRelease(
        artifacts=artifacts,
        internal_manifest=internal_body,
        external_manifest=external_body,
        qualification=qualification_body,
    )
    validate_prepared_release(prepared)
    return prepared


def validate_prepared_release(prepared: PreparedRelease) -> None:
    """Perform a complete serialized record, boundary, and mask audit."""

    if set(prepared.artifacts) != _ARTIFACT_NAMES:
        raise ValueError("prepared release artifact set is invalid")
    if prepared.qualification.get("training_authorized") is not False:
        raise ValueError("prepared release authorizes training")
    if prepared.external_manifest.get("training_authorized") is not False:
        raise ValueError("external manifest authorizes training")
    internal_body = dict(prepared.internal_manifest)
    internal_hash = internal_body.pop("manifest_sha256", None)
    if sha256_json(internal_body) != internal_hash:
        raise ValueError("internal manifest identity mismatch")
    external_body = dict(prepared.external_manifest)
    external_hash = external_body.pop("template_sha256", None)
    if sha256_json(external_body) != external_hash:
        raise ValueError("external manifest template identity mismatch")
    qualification_body = dict(prepared.qualification)
    qualification_hash = qualification_body.pop("qualification_sha256", None)
    if sha256_json(qualification_body) != qualification_hash:
        raise ValueError("qualification identity mismatch")
    evidence = prepared.qualification.get("artifact_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("qualification artifact evidence is invalid")
    for name, payload in prepared.artifacts.items():
        if evidence.get(name) != _artifact_evidence(payload):
            raise ValueError(f"qualification does not bind artifact: {name}")

    logical_rows = [
        json.loads(line)
        for line in prepared.artifacts["logical_records.jsonl"].splitlines()
    ]
    split_reports = prepared.internal_manifest.get("splits")
    if not isinstance(split_reports, Mapping):
        raise ValueError("internal split reports are invalid")
    for split in EXPECTED_SPLITS:
        tokens = np.frombuffer(
            prepared.artifacts[f"{split}.tokens.bin"], dtype=np.uint16
        )
        masks = np.frombuffer(
            prepared.artifacts[f"{split}.mask.bin"], dtype=np.uint8
        )
        if tokens.size != masks.size or tokens.size % RECORD_WIDTH:
            raise ValueError(f"{split} token/mask record layout mismatch")
        token_rows = tokens.reshape(-1, RECORD_WIDTH)
        mask_rows = masks.reshape(-1, RECORD_WIDTH)
        if not np.isin(mask_rows, (0, 1)).all():
            raise ValueError(f"{split} contains a non-binary mask")
        rows = [row for row in logical_rows if row["split"] == split]
        if len(rows) != split_reports[split]["logical_example_count"]:
            raise ValueError(f"{split} logical lineage count mismatch")
        by_record: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_record.setdefault(int(row["packed_record_index"]), []).append(row)
        if set(by_record) != set(range(len(token_rows))):
            raise ValueError(f"{split} packed-record lineage is incomplete")
        for record_index, spans in by_record.items():
            spans.sort(key=lambda row: int(row["start"]))
            expected_start = 0
            for span in spans:
                start = int(span["start"])
                end = int(span["end"])
                target_start = int(span["target_start"])
                if start != expected_start or not start < target_start < end:
                    raise ValueError(f"{split} contains invalid logical boundaries")
                if int(mask_rows[record_index, start]) != 0:
                    raise ValueError(f"{split} supervises a cross-example target")
                if np.any(mask_rows[record_index, start:target_start]):
                    raise ValueError(f"{split} supervises prompt tokens")
                if not np.all(mask_rows[record_index, target_start:end]):
                    raise ValueError(f"{split} leaves response or EOS unsupervised")
                if int(token_rows[record_index, end - 1]) != 3:
                    raise ValueError(f"{split} logical example lacks terminal EOS")
                expected_start = end
            if np.any(token_rows[record_index, expected_start:] != 0):
                raise ValueError(f"{split} PAD tail contains content")
            if np.any(mask_rows[record_index, expected_start:] != 0):
                raise ValueError(f"{split} PAD tail is supervised")


def authorized_external_manifest(
    prepared: PreparedRelease,
    authorization_id: str,
) -> dict[str, object]:
    """Bind the qualified external-manifest template to one authorization ID."""

    if not authorization_id:
        raise ValueError("authorization_id must be non-empty")
    template = prepared.external_manifest
    body: dict[str, object] = {
        "schema_id": EXTERNAL_MANIFEST_SCHEMA_ID,
        "plan_id": template["plan_id"],
        "plan_sha256": template["plan_sha256"],
        "release_directory": template["release_directory"],
        "internal_manifest": template["internal_manifest"],
        "repository_commit": template["repository_commit"],
        "implementation_sha256": template["implementation_sha256"],
        "qualification_sha256": prepared.qualification["qualification_sha256"],
        "authorization_id": authorization_id,
        "training_authorized": False,
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


def qualify_production_release(
    *,
    repository_root: Path,
    repository_commit: str,
    plan_path: Path | None = None,
    tokenizer_path: Path | None = None,
) -> PreparedRelease:
    """Read and compile the accepted sources without writing production outputs."""

    root = repository_root.resolve()
    plan = load_release_plan(
        plan_path or root / "configs/data/releases/vasu_140m_instruction_seed_v1.plan.json"
    )
    validate_release_plan(plan, root, require_outputs_absent=True)
    tokenizer_file = tokenizer_path or root / "assets/tokenizer.json"
    if sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise ValueError("tokenizer file identity mismatch")
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_file))

    records: list[dict[str, Any]] = []
    source_ids: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for source in plan["sources"]:
        source_id = str(source["source_id"])
        manifest = json.loads((root / str(source["manifest_path"])).read_text())
        for record in load_jsonl(root / str(source["source_path"])):
            example_id = str(record["example_id"])
            records.append(record)
            source_ids[example_id] = source_id
            source_splits[example_id] = manifest["split_assignments"][example_id]
    quarantine = set(plan["release_scope"]["quarantined_example_ids"])
    eligible = [record for record in records if str(record["example_id"]) not in quarantine]
    eligible_ids = {str(record["example_id"]) for record in eligible}
    assignments = _derive_assignments(
        eligible,
        {key: value for key, value in source_ids.items() if key in eligible_ids},
        {key: value for key, value in source_splits.items() if key in eligible_ids},
        seed=int(plan["split_policy"]["seed"]),
        evaluation_count=int(plan["split_policy"]["evaluation_count"]),
    )
    splits: dict[str, list[LogicalExample]] = {split: [] for split in EXPECTED_SPLITS}
    lineage: dict[str, dict[str, str]] = {}
    capability_split_counts: dict[str, int] = {}
    decoded_round_trip_count = 0
    for record in eligible:
        example_id = str(record["example_id"])
        split = assignments[example_id]
        prompt = format_alpaca_prompt(str(record["instruction"]), str(record["input"]))
        compiled = compile_text_example(
            tokenizer=tokenizer,
            example_id=example_id,
            split=split,
            prompt=prompt,
            response=f" {record['response']}",
        )
        decoded = tokenizer.decode(list(compiled.token_ids[:-1]))
        expected_text = f"{prompt} {record['response']}"
        if decoded.lstrip(" ") != expected_text:
            raise ValueError(f"tokenizer decoded round-trip mismatch: {example_id}")
        decoded_round_trip_count += 1
        capability_split_key = f"{record['capability']}:{split}"
        capability_split_counts[capability_split_key] = (
            capability_split_counts.get(capability_split_key, 0) + 1
        )
        splits[split].append(compiled)
        lineage[example_id] = {
            "source_id": source_ids[example_id],
            "capability": str(record["capability"]),
        }
    return qualify_compiled_release(
        splits=splits,
        lineage=lineage,
        decision_identities=_decision_identities(root),
        repository_commit=repository_commit,
        implementation_sha256=sha256_file(Path(__file__)),
        enforce_production_counts=True,
        source_audit={
            "mode": "complete_production_source_audit",
            "decoded_round_trip_count": decoded_round_trip_count,
            "capability_split_counts": dict(sorted(capability_split_counts.items())),
        },
    )


def validate_qualification_report(report: Mapping[str, object]) -> None:
    """Validate a serialized read-only qualification report."""

    required = {
        "schema_id",
        "plan_id",
        "plan_sha256",
        "repository_commit",
        "implementation_sha256",
        "decision_identities",
        "split_counts",
        "assignment_sha256",
        "source_audit",
        "artifact_evidence",
        "external_manifest_template_sha256",
        "production_release_created",
        "training_authorized",
        "release_build_permitted",
        "qualification_sha256",
    }
    if set(report) != required:
        raise ValueError("qualification report fields are invalid")
    expected = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "production_release_created": False,
        "training_authorized": False,
        "release_build_permitted": False,
    }
    for field, value in expected.items():
        if report[field] != value:
            raise ValueError(f"qualification report {field} is invalid")
    source_audit = report["source_audit"]
    if (
        not isinstance(source_audit, Mapping)
        or source_audit.get("mode") != "complete_production_source_audit"
        or source_audit.get("decoded_round_trip_count") != 996
    ):
        raise ValueError("qualification source audit is incomplete")
    evidence = report["artifact_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != _ARTIFACT_NAMES:
        raise ValueError("qualification artifact evidence is incomplete")
    body = dict(report)
    reported_hash = body.pop("qualification_sha256")
    if sha256_json(body) != reported_hash:
        raise ValueError("qualification report identity mismatch")


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _validate_protected_path(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    candidate = path.absolute()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("protected path escapes repository root") from error
    current = candidate
    while current != resolved_root:
        if current.exists() and _is_link_or_junction(current):
            raise ValueError(f"protected path traverses a link or junction: {current}")
        current = current.parent


def validate_authorization(
    *,
    authorization: Mapping[str, object],
    prepared: PreparedRelease,
    repository_commit: str,
    repository_root: Path,
    current_date: date | None = None,
) -> Path:
    """Validate an exact, unused, one-build authorization record."""

    required = {
        "schema_id",
        "authorization_id",
        "scope",
        "approved_by",
        "approval_date",
        "expires_date",
        "repository_commit",
        "implementation_sha256",
        "plan_id",
        "plan_sha256",
        "qualification_sha256",
        "expected_artifact_evidence",
        "external_manifest_sha256",
        "release_directory",
        "external_manifest_path",
        "receipt_path",
        "overwrite_allowed",
        "training_authorized",
        "authorization_sha256",
    }
    if set(authorization) != required:
        raise ValueError("authorization fields are invalid")
    expected = {
        "schema_id": AUTHORIZATION_SCHEMA_ID,
        "scope": "one_production_release_build",
        "repository_commit": repository_commit,
        "implementation_sha256": prepared.qualification["implementation_sha256"],
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "qualification_sha256": prepared.qualification["qualification_sha256"],
        "expected_artifact_evidence": prepared.qualification["artifact_evidence"],
        "external_manifest_sha256": authorized_external_manifest(
            prepared, str(authorization["authorization_id"])
        )["manifest_sha256"],
        "release_directory": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "overwrite_allowed": False,
        "training_authorized": False,
    }
    for field, value in expected.items():
        if authorization[field] != value:
            raise ValueError(f"authorization {field} mismatch")
    for field in ("authorization_id", "approved_by", "approval_date"):
        if not isinstance(authorization[field], str) or not authorization[field]:
            raise ValueError(f"authorization {field} is invalid")
    if not _AUTHORIZATION_ID_RE.fullmatch(str(authorization["authorization_id"])):
        raise ValueError("authorization authorization_id is invalid")
    try:
        approval_date = date.fromisoformat(str(authorization["approval_date"]))
        expires_date = date.fromisoformat(str(authorization["expires_date"]))
    except ValueError as error:
        raise ValueError("authorization dates must use YYYY-MM-DD") from error
    today = current_date or date.today()
    if expires_date < approval_date or today > expires_date:
        raise ValueError("authorization is expired or has an invalid date range")
    body = dict(authorization)
    reported_authorization_sha256 = body.pop("authorization_sha256")
    if sha256_json(body) != reported_authorization_sha256:
        raise ValueError("authorization identity mismatch")
    expected_receipt = f"{RECEIPT_DIRECTORY}/{authorization['authorization_id']}.json"
    if authorization["receipt_path"] != expected_receipt:
        raise ValueError("authorization receipt_path mismatch")
    receipt = repository_root / expected_receipt
    _validate_protected_path(receipt, repository_root)
    if receipt.exists():
        raise ValueError("authorization has already been consumed")
    return receipt


def _probe_git_repository(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def publish_authorized_release(
    *,
    prepared: PreparedRelease,
    authorization: Mapping[str, object],
    repository_root: Path,
    repository_commit: str,
    current_date: date | None = None,
    repository_probe: Callable[[Path], tuple[str, bool]] = _probe_git_repository,
    free_bytes: Callable[[Path], int] | None = None,
    replace_directory: Callable[[Path, Path], None] = os.replace,
    replace_manifest: Callable[[Path, Path], None] = os.replace,
    mutation_hook: Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """Publish exact reviewed bytes once; leave incomplete publication quarantined."""

    root = repository_root.resolve()
    observed_commit, worktree_clean = repository_probe(root)
    if observed_commit != repository_commit:
        raise ValueError("repository commit does not match the reviewed authorization")
    if not worktree_clean:
        raise ValueError("repository worktree must be clean for publication")
    release = root / PRODUCTION_RELEASE_PATH
    external = root / PRODUCTION_MANIFEST_PATH
    receipt = validate_authorization(
        authorization=authorization,
        prepared=prepared,
        repository_commit=repository_commit,
        repository_root=root,
        current_date=current_date,
    )
    external_temporary = external.with_suffix(".json.tmp")
    receipt_temporary = receipt.with_suffix(".json.tmp")
    for path in (release, external, receipt, external_temporary, receipt_temporary):
        _validate_protected_path(path, root)
        if path.exists():
            raise FileExistsError(f"protected output already exists: {path}")
    staging = release.parent / f".{release.name}.staging-{uuid.uuid4().hex}"
    _validate_protected_path(staging, root)
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    if release.parent.exists() and any(
        release.parent.glob(f".{release.name}.staging-*")
    ):
        raise FileExistsError("an unresolved sibling staging directory exists")
    final_external_manifest = authorized_external_manifest(
        prepared, str(authorization["authorization_id"])
    )
    required_bytes = sum(len(payload) for payload in prepared.artifacts.values())
    required_bytes += len(_json_bytes(final_external_manifest))
    available = (free_bytes or (lambda path: shutil.disk_usage(path).free))(root)
    if available < required_bytes * 2:
        raise OSError("insufficient disk space for transactional publication")
    release.parent.mkdir(parents=True, exist_ok=True)
    external.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    for path in (release, external, receipt, staging):
        _validate_protected_path(path, root)
    staging.mkdir()
    directory_published = False
    try:
        for name, payload in prepared.artifacts.items():
            if name not in _ARTIFACT_NAMES:
                raise ValueError(f"unrecognized release artifact: {name}")
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _validate_staged_release(staging, prepared)
        if mutation_hook is not None:
            mutation_hook(staging)
        _validate_staged_release(staging, prepared)
        replace_directory(staging, release)
        directory_published = True
        _validate_staged_release(release, prepared)
        with external_temporary.open("xb") as handle:
            handle.write(_json_bytes(final_external_manifest))
            handle.flush()
            os.fsync(handle.fileno())
        replace_manifest(external_temporary, external)
        receipt_payload = {
            "schema_id": RECEIPT_SCHEMA_ID,
            "authorization_id": authorization["authorization_id"],
            "qualification_sha256": prepared.qualification["qualification_sha256"],
            "external_manifest_sha256": final_external_manifest["manifest_sha256"],
            "repository_commit": repository_commit,
            "implementation_sha256": prepared.qualification["implementation_sha256"],
            "authorization_sha256": authorization["authorization_sha256"],
            "release_complete": True,
            "training_authorized": False,
        }
        receipt_payload["receipt_sha256"] = sha256_json(receipt_payload)
        receipt_temporary.write_bytes(_json_bytes(receipt_payload))
        os.replace(receipt_temporary, receipt)
        return receipt_payload
    except BaseException:
        if not directory_published and staging.exists():
            shutil.rmtree(staging)
        if external_temporary.exists():
            external_temporary.unlink()
        if receipt_temporary.exists():
            receipt_temporary.unlink()
        raise


def _validate_staged_release(staging: Path, prepared: PreparedRelease) -> None:
    observed = {path.name for path in staging.iterdir()}
    if observed != set(prepared.artifacts):
        raise ValueError("staging contains missing or unbound files")
    for name, expected in prepared.artifacts.items():
        path = staging / name
        if _sha256_bytes(path.read_bytes()) != _sha256_bytes(expected):
            raise ValueError(f"staged artifact mutation detected: {name}")


def publication_status(repository_root: Path) -> str:
    """Return absent, quarantined_incomplete, or complete without mutation."""

    root = repository_root.resolve()
    release = root / PRODUCTION_RELEASE_PATH
    external = root / PRODUCTION_MANIFEST_PATH
    for path in (release, external):
        _validate_protected_path(path, root)
    if not release.exists() and not external.exists():
        return "absent"
    if release.is_dir() and not external.exists():
        return "quarantined_incomplete"
    if not release.is_dir() and external.exists():
        raise ValueError("external manifest exists without release directory")
    validate_published_release(root)
    return "complete"


def validate_published_release(repository_root: Path) -> dict[str, object]:
    """Validate both publication objects and all serialized release artifacts."""

    root = repository_root.resolve()
    release = root / PRODUCTION_RELEASE_PATH
    external = root / PRODUCTION_MANIFEST_PATH
    if not release.is_dir() or not external.is_file():
        raise ValueError("published release is incomplete")
    manifest = json.loads(external.read_text(encoding="utf-8"))
    manifest_body = dict(manifest)
    manifest_hash = manifest_body.pop("manifest_sha256", None)
    if sha256_json(manifest_body) != manifest_hash:
        raise ValueError("external manifest identity mismatch")
    if (
        manifest.get("schema_id") != EXTERNAL_MANIFEST_SCHEMA_ID
        or manifest.get("release_directory") != PRODUCTION_RELEASE_PATH
        or manifest.get("training_authorized") is not False
    ):
        raise ValueError("external manifest contract mismatch")
    authorization_id = manifest.get("authorization_id")
    if not isinstance(authorization_id, str) or not _AUTHORIZATION_ID_RE.fullmatch(
        authorization_id
    ):
        raise ValueError("external manifest authorization_id is invalid")
    receipt = root / RECEIPT_DIRECTORY / f"{authorization_id}.json"
    _validate_protected_path(receipt, root)
    if not receipt.is_file():
        raise ValueError("published release is missing its authorization receipt")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_body = dict(receipt_payload)
    receipt_hash = receipt_body.pop("receipt_sha256", None)
    if sha256_json(receipt_body) != receipt_hash:
        raise ValueError("authorization receipt identity mismatch")
    if (
        receipt_payload.get("schema_id") != RECEIPT_SCHEMA_ID
        or receipt_payload.get("authorization_id") != authorization_id
        or receipt_payload.get("external_manifest_sha256") != manifest_hash
        or receipt_payload.get("release_complete") is not True
        or receipt_payload.get("training_authorized") is not False
    ):
        raise ValueError("authorization receipt binding mismatch")
    internal = release / "internal_manifest.json"
    if not internal.is_file():
        raise ValueError("complete publication is missing its internal manifest")
    evidence = manifest.get("internal_manifest")
    if not isinstance(evidence, dict):
        raise ValueError("external manifest internal binding is invalid")
    payload = internal.read_bytes()
    if len(payload) != evidence.get("bytes") or _sha256_bytes(payload) != evidence.get(
        "sha256"
    ):
        raise ValueError("external manifest does not bind the internal manifest")
    internal_manifest = json.loads(payload)
    internal_body = dict(internal_manifest)
    internal_hash = internal_body.pop("manifest_sha256", None)
    if sha256_json(internal_body) != internal_hash:
        raise ValueError("internal manifest identity mismatch")
    if internal_manifest.get("training_authorized") is not False:
        raise ValueError("internal manifest authorizes training")
    expected_names = set(_ARTIFACT_NAMES)
    if {path.name for path in release.iterdir()} != expected_names:
        raise ValueError("published release contains missing or unbound entries")
    logical_evidence = internal_manifest.get("logical_records")
    if not isinstance(logical_evidence, dict):
        raise ValueError("logical-record evidence is invalid")
    logical_path = release / str(logical_evidence.get("path"))
    logical_bytes = logical_path.read_bytes()
    if _artifact_evidence(logical_bytes) != {
        "bytes": logical_evidence.get("bytes"),
        "sha256": logical_evidence.get("sha256"),
    }:
        raise ValueError("logical-record artifact identity mismatch")
    logical_rows = [json.loads(line) for line in logical_bytes.splitlines()]
    split_reports = internal_manifest.get("splits")
    if not isinstance(split_reports, dict):
        raise ValueError("published split evidence is invalid")
    for split in EXPECTED_SPLITS:
        report = split_reports[split]
        token_evidence = report["tokens"]
        mask_evidence = report["stored_mask"]
        token_bytes = (release / token_evidence["path"]).read_bytes()
        mask_bytes = (release / mask_evidence["path"]).read_bytes()
        if _artifact_evidence(token_bytes) != {
            "bytes": token_evidence["bytes"],
            "sha256": token_evidence["sha256"],
        }:
            raise ValueError(f"published {split} token identity mismatch")
        if _artifact_evidence(mask_bytes) != {
            "bytes": mask_evidence["bytes"],
            "sha256": mask_evidence["sha256"],
        }:
            raise ValueError(f"published {split} mask identity mismatch")
        tokens = np.frombuffer(token_bytes, dtype=np.uint16)
        masks = np.frombuffer(mask_bytes, dtype=np.uint8)
        if tokens.size != masks.size or tokens.size % RECORD_WIDTH:
            raise ValueError(f"published {split} layout mismatch")
        token_rows = tokens.reshape(-1, RECORD_WIDTH)
        mask_rows = masks.reshape(-1, RECORD_WIDTH)
        if not np.isin(mask_rows, (0, 1)).all():
            raise ValueError(f"published {split} mask is non-binary")
        rows = [row for row in logical_rows if row["split"] == split]
        by_record: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_record.setdefault(int(row["packed_record_index"]), []).append(row)
        for record_index, spans in by_record.items():
            spans.sort(key=lambda row: int(row["start"]))
            expected_start = 0
            for span in spans:
                start = int(span["start"])
                end = int(span["end"])
                target_start = int(span["target_start"])
                if start != expected_start or int(mask_rows[record_index, start]) != 0:
                    raise ValueError(f"published {split} boundary mismatch")
                if np.any(mask_rows[record_index, start:target_start]):
                    raise ValueError(f"published {split} prompt supervision")
                if not np.all(mask_rows[record_index, target_start:end]):
                    raise ValueError(f"published {split} response under-supervision")
                if int(token_rows[record_index, end - 1]) != 3:
                    raise ValueError(f"published {split} EOS mismatch")
                expected_start = end
            if np.any(token_rows[record_index, expected_start:] != 0) or np.any(
                mask_rows[record_index, expected_start:] != 0
            ):
                raise ValueError(f"published {split} PAD-tail mismatch")
    return manifest
