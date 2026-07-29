"""Canonical, deterministic intermediate states for verified arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from vasu.data.arithmetic_v2 import recompute_answer


def _fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def serialize_verified_steps(record: Mapping[str, Any]) -> str:
    """Render bounded symbolic states derived only from verification metadata."""

    data = record["verification"]
    kind = data["kind"]
    if kind == "sum":
        values = data["values"]
        steps = [f"{values[0]} + {values[1]} = {values[0] + values[1]}"]
    elif kind == "subtract":
        values = data["values"]
        steps = [f"{values[0]} - {values[1]} = {values[0] - values[1]}"]
    elif kind == "product":
        values = data["values"]
        steps = [f"{values[0]} * {values[1]} = {values[0] * values[1]}"]
    elif kind == "exact_division":
        steps = [f"{data['dividend']} / {data['divisor']} = {data['dividend'] // data['divisor']}"]
    elif kind == "arithmetic_sequence":
        next_value = data["start"] + data["step"] * data["terms"]
        steps = [f"next = {next_value}"]
    elif kind == "percentage":
        amount = data["number"] * data["percent"] // 100
        steps = [f"{data['percent']}% of {data['number']} = {amount}"]
    elif kind == "fraction":
        first, second = Fraction(*data["first"]), Fraction(*data["second"])
        operation = data["operation"]
        if operation == "simplify":
            steps = [f"simplified = {_fraction(first)}"]
        elif operation == "compare":
            steps = [f"compare = {recompute_answer(record)}"]
        else:
            value = {"add": first + second, "subtract": first - second, "multiply": first * second}[operation]
            steps = [f"result = {_fraction(value)}"]
    elif kind == "mixed":
        a, b, c = data["values"]
        intermediate = a + b if data["operation"] == "sum_then_multiply" else a - b
        steps = [f"intermediate = {intermediate}", f"result = {recompute_answer(record)}"]
    else:
        steps = [f"result = {recompute_answer(record)}"]
    return "\n".join([*steps, f"Answer: {recompute_answer(record)}"])
