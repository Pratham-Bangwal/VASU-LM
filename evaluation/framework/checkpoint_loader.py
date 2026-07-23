"""Canonical checkpoint loading shared by internal capability evaluations."""

from __future__ import annotations

from pathlib import Path

import torch

from vasu.config import ModelConfig, get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer

from .registry import sha256_file
from .schemas import CheckpointEntry


def resolve_config(name: str) -> ModelConfig:
    if name == "vasu_31m":
        return ModelConfig()
    if name == "vasu_60m":
        return get_vasu_60m_config()
    raise ValueError(f"Unsupported model configuration: {name}")


def load_checkpoint_model(entry: CheckpointEntry, device: torch.device) -> tuple[VASUModel, dict[str, object]]:
    """Strictly load a canonical VASU checkpoint and report immutable metadata."""

    path = Path(entry.path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = VASUModel(resolve_config(entry.model_config)).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model" not in payload:
        raise ValueError(f"Checkpoint has no model state: {path}")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, {"path": str(path), "sha256": sha256_file(path), "global_step": payload.get("global_step")}


def load_tokenizer(path: Path = Path("assets/tokenizer.json")) -> tuple[VASUTokenizer, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    tokenizer = VASUTokenizer()
    tokenizer.load(str(path))
    return tokenizer, sha256_file(path)
