"""Small typed representations for versioned internal capability suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OBJECTIVE = "objective"
HEURISTIC = "heuristic"
HUMAN = "human"


@dataclass(frozen=True)
class CheckpointEntry:
    identifier: str
    path: str
    model_config: str
    prompt_format: str


@dataclass(frozen=True)
class CapabilityTask:
    identifier: str
    category: str
    prompt: str
    metric_kind: str
    scoring: dict[str, Any]
    provenance: dict[str, Any]
    version: str
