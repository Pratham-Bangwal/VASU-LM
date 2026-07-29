from vasu.data.arithmetic_steps import serialize_verified_steps
from vasu.data.arithmetic_v2 import generate_records, recompute_answer


def test_steps_are_deterministic_and_end_in_verified_answer() -> None:
    records = generate_records("train", 220, 42)
    for record in records:
        rendered = serialize_verified_steps(record)
        assert rendered == serialize_verified_steps(record)
        assert rendered.endswith(f"Answer: {recompute_answer(record)}")


def test_mixed_expression_has_a_bounded_intermediate_state() -> None:
    record = next(item for item in generate_records("train", 220, 42) if item["operation"] == "mixed_expression")
    assert "intermediate =" in serialize_verified_steps(record)
