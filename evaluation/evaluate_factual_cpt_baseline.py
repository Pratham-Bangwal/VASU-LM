"""Evaluate the frozen VASU-60M parent without creating an optimizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import normalized_words, repetition_ratio
from vasu.config import get_vasu_60m_config
from vasu.data.preparation.reporting import sha256_file
from vasu.inference.generate import generate_token_ids
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.losses import language_model_loss


DEFAULT_CONFIG = Path("configs/evaluation/vasu_60m_factual_cpt_v1.json")
EXPECTED_CHECKPOINT_KEYS = {
    "epoch", "global_step", "model", "optimizer", "scheduler", "loss"
}


class FlatTokenValidationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read a flat uint16 validation stream without loading it into RAM."""

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        byte_size = self.path.stat().st_size
        if byte_size == 0 or byte_size % np.dtype(np.uint16).itemsize:
            raise ValueError("validation binary must be non-empty aligned uint16 data")
        self.token_count = byte_size // np.dtype(np.uint16).itemsize
        self._tokens: np.memmap | None = None

    def _ensure_open(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return max(0, (self.token_count - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        row = np.asarray(
            self._ensure_open()[start : start + self.sequence_length + 1],
            dtype=np.int64,
        )
        return torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{path} must contain a JSON object or array")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("format_version") != "vasu_factual_cpt_evaluation_v1":
        raise ValueError("unsupported factual-CPT evaluation configuration")
    if config.get("model_configuration") != "vasu_60m":
        raise ValueError("baseline evaluation requires the VASU-60M configuration")
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation settings must be an object")
    required = {
        "prompt_format", "do_sample", "max_new_tokens", "temperature",
        "top_k", "top_p", "use_kv_cache",
    }
    missing = sorted(required.difference(generation))
    if missing:
        raise ValueError(f"generation settings are missing: {', '.join(missing)}")
    if generation["prompt_format"] != "plain":
        raise ValueError("base-model evaluation must use raw plain prompts")


def _load_parent_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VASUModel, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not EXPECTED_CHECKPOINT_KEYS.issubset(checkpoint):
        missing = sorted(EXPECTED_CHECKPOINT_KEYS.difference(checkpoint))
        raise ValueError(f"parent checkpoint is missing keys: {', '.join(missing)}")
    model = VASUModel(get_vasu_60m_config())
    model.load_state_dict(checkpoint["model"], strict=True)
    global_step = int(checkpoint["global_step"])
    del checkpoint
    model.to(device)
    model.eval()
    return model, global_step


@torch.inference_mode()
def validation_loss(
    model: VASUModel,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Compute token-weighted next-token cross-entropy without updates."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    weighted_loss = 0.0
    token_count = 0
    batch_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss = language_model_loss(model(x), y)
        tokens = y.numel()
        weighted_loss += float(loss.item()) * tokens
        token_count += tokens
        batch_count += 1
    if token_count == 0:
        raise ValueError("validation dataset contains no target tokens")
    return {
        "loss": weighted_loss / token_count,
        "samples": len(dataset),
        "batches": batch_count,
        "tokens": token_count,
    }


def response_metrics(text: str, generated_token_count: int) -> dict[str, Any]:
    words = normalized_words(text)
    lines = [line for line in text.splitlines() if line.strip()]
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    folded = text.casefold()
    heading = bool(re.search(r"(?m)^\s*(?:#{1,6}|={2,}).+", text))
    section = bool(re.search(
        r"(?im)^\s*(references|see also|external links|further reading)\s*:?\s*$",
        text,
    ))
    citation = bool(re.search(r"\[(?:\d+|citation needed)\]", folded))
    article_lead = bool(re.match(r"^\s*[^.!?]{1,80}\s+is\s+(?:an?|the)\b", text, re.I))
    return {
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(sentences),
        "generated_tokens": generated_token_count,
        "repetition_ratio": repetition_ratio(text),
        "encyclopedic_style": {
            "heading_like": heading,
            "section_heading": section,
            "citation_like": citation,
            "article_lead_like": article_lead,
        },
    }


def aggregate_generation_metrics(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not outputs:
        raise ValueError("at least one generated output is required")
    metrics = [item["metrics"] for item in outputs]
    repetitions = [float(item["repetition_ratio"]) for item in metrics]
    lengths: dict[str, dict[str, float | int]] = {}
    for name in ("characters", "words", "sentences", "generated_tokens"):
        values = [int(item[name]) for item in metrics]
        lengths[name] = {"minimum": min(values), "maximum": max(values), "mean": mean(values)}
    style_names = ("heading_like", "section_heading", "citation_like", "article_lead_like")
    styles = {
        name: {
            "count": sum(bool(item["encyclopedic_style"][name]) for item in metrics),
            "rate": mean(bool(item["encyclopedic_style"][name]) for item in metrics),
        }
        for name in style_names
    }
    any_style = [any(item["encyclopedic_style"].values()) for item in metrics]
    styles["any_indicator"] = {"count": sum(any_style), "rate": mean(any_style)}
    return {
        "repetition": {
            "minimum": min(repetitions),
            "maximum": max(repetitions),
            "mean": mean(repetitions),
        },
        "response_lengths": lengths,
        "encyclopedic_style_indicators": styles,
    }


def _format_text_report(result: dict[str, Any]) -> str:
    lines = [
        "VASU-60M factual-CPT parent baseline",
        "=" * 72,
        f"Checkpoint: {result['checkpoint']['path']}",
        f"Global step: {result['checkpoint']['global_step']}",
        f"Checkpoint SHA-256: {result['checkpoint']['sha256']}",
        f"FineWeb validation loss: {result['validation']['fineweb']['loss']:.6f}",
        f"Wikimedia validation loss: {result['validation']['wikimedia']['loss']:.6f}",
        f"Random seed: {result['random_seed']}",
        f"Generation settings: {json.dumps(result['generation_settings'], sort_keys=True)}",
        "",
    ]
    for item in result["generations"]:
        lines.extend((
            "-" * 72,
            f"Prompt ID: {item['id']}",
            f"Category: {item['category']}",
            f"Prompt: {item['prompt']}",
            f"Metrics: {json.dumps(item['metrics'], sort_keys=True)}",
            "Continuation:",
            item["response"],
            "",
        ))
    lines.extend(("-" * 72, "Aggregate metrics:", json.dumps(result["generation_metrics"], indent=2)))
    return "\n".join(lines) + "\n"


def run_baseline(
    config_path: Path = DEFAULT_CONFIG,
    *,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
    result_path: Path | None = None,
    text_output_path: Path | None = None,
) -> dict[str, Any]:
    raw_config = _load_json(config_path)
    if not isinstance(raw_config, dict):
        raise ValueError("evaluation configuration root must be an object")
    _validate_config(raw_config)
    prompts_path = Path(raw_config["prompt_file"])
    raw_prompts = _load_json(prompts_path)
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("factual prompt file must contain a non-empty array")

    selected_checkpoint_path = checkpoint_path or Path(raw_config["parent_checkpoint"])
    tokenizer_path = Path(raw_config["tokenizer_path"])
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    model, global_step = _load_parent_model(selected_checkpoint_path, selected_device)

    validation = raw_config["validation"]
    fineweb_samples = int(validation["fineweb_samples"])
    sequence_length = get_vasu_60m_config().max_seq_len
    fineweb = ManifestTokenDataset(
        validation["fineweb_manifest"],
        "validation",
        sequence_length,
        logical_start=0,
        logical_end=fineweb_samples * sequence_length + 1,
    )
    wikimedia = FlatTokenValidationDataset(validation["wikimedia_path"], sequence_length)
    fineweb_result = validation_loss(
        model, fineweb, batch_size=int(validation["batch_size"]), device=selected_device
    )
    wikimedia_result = validation_loss(
        model, wikimedia, batch_size=int(validation["batch_size"]), device=selected_device
    )

    generation = dict(raw_config["generation"])
    outputs: list[dict[str, Any]] = []
    for index, prompt_item in enumerate(raw_prompts):
        prompt = str(prompt_item["prompt"])
        prompt_seed = int(raw_config["seed"]) + index
        torch.manual_seed(prompt_seed)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(prompt_seed)
        generated_ids = generate_token_ids(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=selected_device,
            **generation,
        )
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        outputs.append({
            "id": prompt_item["id"],
            "category": prompt_item["category"],
            "prompt": prompt,
            "expected_behavior": prompt_item["expected_behavior"],
            "seed": prompt_seed,
            "response": response,
            "metrics": response_metrics(response, len(generated_ids)),
        })

    result: dict[str, Any] = {
        "format_version": "vasu_factual_cpt_parent_baseline_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(selected_device),
        "checkpoint": {
            "path": str(selected_checkpoint_path),
            "sha256": sha256_file(selected_checkpoint_path),
            "global_step": global_step,
        },
        "evaluation_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "prompt_file": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
        "tokenizer": {"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
        "random_seed": int(raw_config["seed"]),
        "seed_policy": "base seed plus zero-based prompt index",
        "generation_settings": generation,
        "validation": {"fineweb": fineweb_result, "wikimedia": wikimedia_result},
        "factual_prompts_evaluated": len(outputs),
        "generations": outputs,
        "generation_metrics": aggregate_generation_metrics(outputs),
        "optimizer_updates_performed": 0,
    }
    selected_result_path = result_path or Path(raw_config["result_path"])
    selected_text_path = text_output_path or Path(raw_config["text_output_path"])
    _atomic_write(
        selected_result_path,
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write(selected_text_path, _format_text_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--text-output-path", type=Path)
    args = parser.parse_args()
    result = run_baseline(
        args.config,
        checkpoint_path=args.checkpoint,
        result_path=args.result_path,
        text_output_path=args.text_output_path,
    )
    print(_format_text_report(result), end="")
    configured_result = json.loads(args.config.read_text(encoding="utf-8"))["result_path"]
    print(f"JSON result: {args.result_path or configured_result}")


if __name__ == "__main__":
    main()
