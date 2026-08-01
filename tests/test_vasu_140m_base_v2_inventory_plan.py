from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2_inventory_plan import (
    COUNTS,
    DIMENSIONS,
    PROMPT_DIMENSIONS,
    plan_identity,
    validate_inventory_construction_plan,
    validate_inventory_construction_plan_files,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "configs/evaluation/vasu_140m_base_v2_inventory_construction_plan.json"
)


def plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def rehash(value: dict[str, object]) -> None:
    value["plan_sha256"] = plan_identity(value)


def test_frozen_plan_and_every_bound_file_validate() -> None:
    value = plan()
    validate_inventory_construction_plan(value)
    validate_inventory_construction_plan_files(value, ROOT)
    assert value["construction_authorized"] is False
    assert value["evaluation_run_authorized"] is False
    assert value["training_authorized"] is False


def test_plan_covers_all_dimensions_and_frozen_counts() -> None:
    value = plan()
    assert set(value["dimensions"]) == DIMENSIONS
    for dimension in DIMENSIONS:
        assert value["dimensions"][dimension]["counts"] == COUNTS[dimension]
    assert value["dimensions"]["likelihood"]["count_scope"] == "per_admitted_source"
    assert PROMPT_DIMENSIONS == DIMENSIONS - {"likelihood"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["dimensions"]["arithmetic"]["counts"].update(
                held_out=999
            ),
            "frozen counts",
        ),
        (
            lambda value: value["dimensions"]["likelihood"]["source_ids"].append(
                "factual-cpt-v2-development-seed"
            ),
            "frozen assignment",
        ),
        (
            lambda value: value["dimensions"]["likelihood"].update(
                count_scope="total"
            ),
            "count scope",
        ),
        (
            lambda value: value["dimensions"]["factuality"].update(
                source_ids=[]
            ),
            "frozen assignment",
        ),
        (
            lambda value: value["dimensions"]["arithmetic"].update(
                source_ids=["factual-cpt-v2-development-seed"]
            ),
            "frozen assignment",
        ),
        (
            lambda value: value["source_catalog"][2].update(
                prompt_reuse_permitted=True
            ),
            "reuse permissions",
        ),
        (
            lambda value: value["dimensions"]["robustness"].update(
                interface="raw_continuation"
            ),
            "interface mismatch",
        ),
        (
            lambda value: value["split_policy"].update(
                semantic_family_isolation=False
            ),
            "must be true",
        ),
        (
            lambda value: value["held_out_security"].update(
                recipient_fingerprint="age1premature"
            ),
            "unkeyed",
        ),
        (
            lambda value: value["review"].update(status="accepted"),
            "review must be pending",
        ),
        (
            lambda value: value["dependencies"].pop(
                "inventory_postcommit_identity_decision"
            ),
            "dependencies are incomplete",
        ),
        (
            lambda value: value["dependencies"][
                "inventory_contract_decision"
            ].update(path="evaluation/not-a-decision.json"),
            "must be a decision document",
        ),
        (
            lambda value: value.update(construction_authorized=True),
            "construction_authorized must be false",
        ),
    ],
)
def test_plan_policy_drift_fails_closed(mutation, message: str) -> None:
    value = plan()
    mutation(value)
    rehash(value)
    with pytest.raises(ValueError, match=message):
        validate_inventory_construction_plan(value)


def test_plan_rejects_unbound_nested_mutation() -> None:
    value = plan()
    value["source_catalog"][0]["provenance"] = "mutated but structurally valid"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_inventory_construction_plan(value)


def test_source_roles_prohibit_instruction_prompt_reuse() -> None:
    value = plan()
    sources = {item["source_id"]: item for item in value["source_catalog"]}
    topic_seed = sources["ultrachat-promotion-topic-seed"]
    assert topic_seed["prompt_reuse_permitted"] is False
    assert topic_seed["held_out_derivation_permitted"] is False
    factual_seed = sources["factual-cpt-v2-development-seed"]
    assert factual_seed["held_out_derivation_permitted"] is False


def test_bound_file_hash_drift_is_rejected(tmp_path: Path) -> None:
    value = deepcopy(plan())
    fake_root = tmp_path
    bindings = [
        value["tokenizer"],
        *value["implementation"].values(),
        *value["dependencies"].values(),
    ]
    bindings.extend(item["artifact"] for item in value["source_catalog"])
    for binding in bindings:
        source = ROOT / binding["path"]
        target = fake_root / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    first = fake_root / bindings[0]["path"]
    first.write_bytes(first.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_inventory_construction_plan_files(value, fake_root)
