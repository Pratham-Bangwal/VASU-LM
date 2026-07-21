"""Authorization-gated VASU-60M factual continued-pretraining experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vasu.config import get_vasu_60m_config
from vasu.data.preparation.reporting import sha256_file
from vasu.data.pretraining_mixture import FixedRecordTokenDataset
from vasu.model.model import VASUModel
from vasu.inference.generate import generate
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.accumulation import (
    build_accumulation_plan,
    normalize_partial_accumulation,
)
from vasu.training.losses import language_model_loss


DEFAULT_CONFIG = Path(
    "configs/training/vasu_60m_factual_cpt_wikimedia_15pct.json"
)
REQUIRED_CHECKPOINT_KEYS = {
    "epoch", "global_step", "model", "optimizer", "scheduler", "loss"
}
FINEWEB_BASELINE_LOSS = 3.356127
FINEWEB_MAX_RELATIVE_REGRESSION = 0.03
FINEWEB_LOSS_GUARDRAIL = FINEWEB_BASELINE_LOSS * (
    1.0 + FINEWEB_MAX_RELATIVE_REGRESSION
)


class FlatValidationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path, sequence_length: int) -> None:
        self.path = path
        self.sequence_length = sequence_length
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        row = np.asarray(
            self.tokens[start : start + self.sequence_length + 1], dtype=np.int64
        )
        return torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())


def load_experiment_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration root must be an object")
    if payload.get("format_version") != "vasu_factual_cpt_experiment_v1":
        raise ValueError("unsupported factual-CPT experiment configuration")
    if not isinstance(payload.get("training_authorized"), bool):
        raise ValueError("training_authorized must be boolean")
    if payload.get("model_configuration") != "vasu_60m":
        raise ValueError("experiment must use the opt-in VASU-60M configuration")
    required_values = {
        "sequence_length": 256,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "precision": "amp",
        "random_seed": 42,
        "parent_global_step": 200000,
        "validation_interval_steps": 100,
        "checkpoint_interval_steps": 200,
        "sample_generation_interval_steps": 100,
    }
    for name, expected in required_values.items():
        if payload.get(name) != expected:
            raise ValueError(f"{name} must remain {expected!r}")
    parent = Path(payload["starting_checkpoint"]).resolve()
    output = Path(payload["output_directory"]).resolve()
    if output == parent or parent in output.parents or output in parent.parents:
        raise ValueError("output directory must be isolated from parent checkpoint")
    return payload


def _validate_parent_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["starting_checkpoint"])
    if sha256_file(path) != config["starting_checkpoint_sha256"]:
        raise ValueError("parent checkpoint SHA-256 mismatch")
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    if not REQUIRED_CHECKPOINT_KEYS.issubset(checkpoint):
        raise ValueError("parent checkpoint is missing required schema keys")
    if int(checkpoint["global_step"]) != config["parent_global_step"]:
        raise ValueError("parent checkpoint global_step mismatch")
    state = checkpoint["model"]
    expected_shapes = {
        "embedding.embedding.weight": (32000, 512),
        "lm_head.weight": (32000, 512),
        "blocks.0.attention.q_proj.weight": (512, 512),
        "blocks.9.mlp.w1.weight": (2048, 512),
    }
    for key, shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"parent checkpoint architecture mismatch at {key}")
    return checkpoint


def _validate_mixture_artifacts(metadata: dict[str, Any]) -> list[str]:
    output_path = Path(metadata["output_path"])
    schedule_path = Path(metadata["selection_schedule_path"])
    tokenizer_path = Path(metadata["tokenizer_path"])
    expected = (
        (output_path, metadata["output_sha256"], "mixture"),
        (schedule_path, metadata["selection_schedule_sha256"], "source schedule"),
        (tokenizer_path, metadata["tokenizer_sha256"], "tokenizer"),
    )
    for path, expected_sha, label in expected:
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{label} SHA-256 mismatch")
    schedule = schedule_path.read_text(encoding="utf-8").splitlines()
    if len(schedule) != int(metadata["record_count"]):
        raise ValueError("source schedule record count mismatch")
    expected_sources = set(metadata["sources"])
    if set(schedule) != expected_sources:
        raise ValueError("source schedule contains unexpected source IDs")
    return schedule


def validate_experiment(path: Path, *, instantiate_model: bool) -> dict[str, Any]:
    config = load_experiment_config(path)
    metadata_path = Path(config["mixture_artifact_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_mixture_artifacts(metadata)
    if metadata["actual_supervised_token_budget"] != config[
        "actual_aligned_token_budget"
    ]:
        raise ValueError("mixture token budget differs from experiment config")
    dataset = FixedRecordTokenDataset(
        metadata["output_path"],
        config["sequence_length"],
        expected_sha256=metadata["output_sha256"],
    )
    accumulation_plan = build_accumulation_plan(
        record_count=len(dataset),
        batch_size=config["batch_size"],
        accumulation_steps=config["gradient_accumulation_steps"],
        sequence_length=config["sequence_length"],
    )
    expected_steps = accumulation_plan.optimizer_steps
    if expected_steps != config["expected_optimizer_steps"]:
        raise ValueError("expected optimizer-step count is inconsistent")
    checkpoint = _validate_parent_checkpoint(config)
    model_loaded = False
    if instantiate_model:
        model = VASUModel(get_vasu_60m_config())
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        model_loaded = True
        del model
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]
    if x0.shape != (256,) or y0.shape != x0.shape or x1.shape != y1.shape:
        raise ValueError("mixture batch-shape validation failed")
    wiki_validation = FlatValidationDataset(
        Path(config["validation"]["wikimedia_path"]), config["sequence_length"]
    )
    fineweb_validation = ManifestTokenDataset(
        config["validation"]["fineweb_manifest"],
        "validation",
        config["sequence_length"],
        logical_start=0,
        logical_end=256 * config["sequence_length"] + 1,
    )
    return {
        "training_authorized": config["training_authorized"],
        "parent_global_step": checkpoint["global_step"],
        "parent_checkpoint_compatible": True,
        "strict_model_load": model_loaded,
        "mixture_records": len(dataset),
        "first_two_sample_shapes": [list(x0.shape), list(x1.shape)],
        "expected_optimizer_steps": expected_steps,
        "trained_tokens": accumulation_plan.trained_tokens,
        "final_partial_microbatches": accumulation_plan.final_microbatches,
        "final_partial_tokens": accumulation_plan.final_tokens,
        "fineweb_validation_samples": len(fineweb_validation),
        "wikimedia_validation_samples": len(wiki_validation),
        "optimizer_updates_performed": 0,
    }


def _temperature() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def _atomic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not REQUIRED_CHECKPOINT_KEYS.issubset(saved):
        raise ValueError(f"checkpoint verification failed for {path}")
    if int(saved["global_step"]) != int(payload["global_step"]):
        raise ValueError(f"checkpoint global_step mismatch for {path}")


def _checkpoint_payload(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    global_step: int,
    loss: float,
) -> dict[str, Any]:
    """Build the existing checkpoint container without changing its schema."""

    return {
        "epoch": 0,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
    }


def _write_training_samples(
    model: VASUModel,
    tokenizer: VASUTokenizer,
    prompts_path: Path,
    output_path: Path,
    device: torch.device,
    *,
    seed: int,
) -> None:
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("sample prompt file must contain a non-empty array")
    was_training = model.training
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    lines: list[str] = []
    for item in prompts:
        prompt = str(item["prompt"])
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=60,
            temperature=0.6,
            top_k=20,
            top_p=0.9,
            do_sample=True,
            prompt_format="plain",
        )
        lines.extend((f"Prompt: {prompt}", f"Continuation: {response.strip()}", ""))
    if was_training:
        model.train()
    _atomic_text(output_path, "\n".join(lines))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _lr_factor(step: int, total: int, warmup: int, minimum_factor: float) -> float:
    if step < warmup:
        return max(minimum_factor, (step + 1) / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return minimum_factor + (1.0 - minimum_factor) * 0.5 * (
        1.0 + math.cos(math.pi * min(progress, 1.0))
    )


@torch.no_grad()
def _validation_loss(
    model: VASUModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss = language_model_loss(model(x), y)
        total += float(loss.item())
        batches += 1
    model.train()
    return total / max(1, batches)


def run_training(config_path: Path) -> None:
    """Run the isolated experiment only after explicit configuration approval."""
    config = load_experiment_config(config_path)
    if not config["training_authorized"]:
        raise PermissionError(
            "Training is not authorized. Review the prepared artifacts and set "
            "training_authorized=true explicitly before launching this entry point."
        )
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads(
        Path(config["mixture_artifact_metadata"]).read_text(encoding="utf-8")
    )
    source_schedule = _validate_mixture_artifacts(metadata)
    parent = _validate_parent_checkpoint(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved factual-CPT run")
    model = VASUModel(get_vasu_60m_config()).to(device)
    model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    total_steps = config["expected_optimizer_steps"]
    minimum_factor = config["minimum_learning_rate"] / config["learning_rate"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, total_steps, config["warmup_steps"], minimum_factor),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")
    output_dir = Path(config["output_directory"])
    experiment_step = 0
    candidates = sorted(output_dir.glob("step_*.pt")) if output_dir.exists() else []
    if candidates:
        latest = max(candidates, key=lambda item: int(item.stem.split("_")[-1]))
        resumed = torch.load(latest, map_location=device, weights_only=False)
        if not REQUIRED_CHECKPOINT_KEYS.issubset(resumed):
            raise ValueError(f"invalid experiment checkpoint: {latest}")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        experiment_step = int(resumed["global_step"])
        print(f"resume_checkpoint={latest}")
    else:
        print("resume_checkpoint=none (fresh experiment optimizer and scheduler)")
    records_per_step = config["batch_size"] * config["gradient_accumulation_steps"]
    start_record = experiment_step * records_per_step
    print(
        f"parent_global_step={config['parent_global_step']} "
        f"initial_experiment_step={experiment_step} start_record={start_record}"
    )
    if experiment_step >= total_steps:
        print(f"Experiment already complete at step {experiment_step}.")
        return
    dataset = FixedRecordTokenDataset(
        metadata["output_path"], config["sequence_length"],
        start_record=start_record, expected_sha256=metadata["output_sha256"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    fineweb_validation = ManifestTokenDataset(
        config["validation"]["fineweb_manifest"],
        "validation",
        config["sequence_length"],
        logical_start=0,
        logical_end=256 * config["sequence_length"] + 1,
    )
    wiki_validation = FlatValidationDataset(
        Path(config["validation"]["wikimedia_path"]), config["sequence_length"]
    )
    fineweb_validation_loader = DataLoader(
        fineweb_validation, batch_size=config["batch_size"], shuffle=False
    )
    wiki_validation_loader = DataLoader(
        wiki_validation, batch_size=config["batch_size"], shuffle=False
    )
    optimizer.zero_grad(set_to_none=True)
    best_validation_score = float("inf")
    best_path = output_dir / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        if REQUIRED_CHECKPOINT_KEYS.issubset(best_checkpoint):
            best_validation_score = float(best_checkpoint["loss"])
    accumulated = 0
    latest_loss = float("nan")
    record_cursor = start_record
    source_record_counts: Counter[str] = Counter(source_schedule[:start_record])
    validation_history_path = output_dir / "validation_history.jsonl"
    for micro_step, (x, y) in enumerate(loader, start=1):
        batch_records = int(x.size(0))
        source_record_counts.update(
            source_schedule[record_cursor : record_cursor + batch_records]
        )
        record_cursor += batch_records
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss = language_model_loss(model(x), y)
            scaled_loss = loss / config["gradient_accumulation_steps"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at micro_step={micro_step}")
        scaler.scale(scaled_loss).backward()
        accumulated += 1
        is_last = micro_step == len(loader)
        if accumulated == config["gradient_accumulation_steps"] or is_last:
            scaler.unscale_(optimizer)
            normalize_partial_accumulation(
                model.parameters(),
                accumulated_microbatches=accumulated,
                target_microbatches=config["gradient_accumulation_steps"],
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"]
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm before experiment_step={experiment_step + 1}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            experiment_step += 1
            accumulated = 0
            latest_loss = float(loss.item())
            temperature = _temperature()
            tokens_processed = record_cursor * config["sequence_length"]
            source_token_counts = {
                source: count * config["sequence_length"]
                for source, count in sorted(source_record_counts.items())
            }
            gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            if experiment_step % config["logging_interval_steps"] == 0:
                print(
                    f"experiment_step={experiment_step}/{total_steps} "
                    f"parent_global_step={config['parent_global_step']} "
                    f"loss={latest_loss:.6f} lr={scheduler.get_last_lr()[0]:.8g} "
                    f"grad_norm={float(gradient_norm):.6f} "
                    f"tokens_processed={tokens_processed} "
                    f"source_tokens={json.dumps(source_token_counts, sort_keys=True)} "
                    f"gpu_allocated_mib={gpu_allocated:.2f} "
                    f"gpu_reserved_mib={gpu_reserved:.2f} temperature={temperature}"
                )
            if (
                experiment_step % config["validation_interval_steps"] == 0
                or experiment_step == total_steps
            ):
                fineweb_loss = _validation_loss(
                    model, fineweb_validation_loader, device
                )
                wikimedia_loss = _validation_loss(
                    model, wiki_validation_loader, device
                )
                print(
                    f"validation step={experiment_step} "
                    f"fineweb={fineweb_loss:.6f} wikimedia={wikimedia_loss:.6f}"
                )
                validation_record = {
                    "experiment_step": experiment_step,
                    "parent_global_step": config["parent_global_step"],
                    "fineweb_validation_loss": fineweb_loss,
                    "wikimedia_validation_loss": wikimedia_loss,
                    "fineweb_relative_regression": (
                        fineweb_loss / FINEWEB_BASELINE_LOSS - 1.0
                    ),
                    "tokens_processed": tokens_processed,
                    "source_token_counts": source_token_counts,
                }
                validation_history_path.parent.mkdir(parents=True, exist_ok=True)
                with validation_history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
                if fineweb_loss > FINEWEB_LOSS_GUARDRAIL:
                    payload = _checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        global_step=experiment_step,
                        loss=latest_loss,
                    )
                    guardrail_path = output_dir / f"guardrail_stop_step_{experiment_step}.pt"
                    _atomic_checkpoint(payload, guardrail_path)
                    _atomic_checkpoint(payload, output_dir / "latest.pt")
                    raise RuntimeError(
                        "FineWeb validation guardrail exceeded: "
                        f"{fineweb_loss:.6f} > {FINEWEB_LOSS_GUARDRAIL:.6f}"
                    )
                validation_score = wikimedia_loss
                if validation_score < best_validation_score:
                    best_validation_score = validation_score
                    _atomic_checkpoint(
                        _checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            global_step=experiment_step,
                            loss=validation_score,
                        ),
                        best_path,
                    )
                    print(
                        f"checkpoint_saved={best_path} "
                        f"best_wikimedia_loss={wikimedia_loss:.6f} "
                        f"fineweb_guardrail_loss={fineweb_loss:.6f}"
                    )
            if (
                experiment_step % config["sample_generation_interval_steps"] == 0
                or experiment_step == total_steps
            ):
                _write_training_samples(
                    model,
                    tokenizer,
                    Path("evaluation/prompts_factual_cpt_v1.json"),
                    output_dir / f"samples_step_{experiment_step}.txt",
                    device,
                    seed=config["random_seed"] + experiment_step,
                )
                print(
                    f"sample_generation_saved="
                    f"{output_dir / f'samples_step_{experiment_step}.txt'}"
                )
            should_save = experiment_step % config["checkpoint_interval_steps"] == 0
            thermal_stop = temperature is not None and temperature >= config[
                "thermal_stop_celsius"
            ]
            if should_save or thermal_stop or experiment_step == total_steps:
                name = (
                    f"thermal_stop_step_{experiment_step}.pt"
                    if thermal_stop else f"step_{experiment_step}.pt"
                )
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    global_step=experiment_step,
                    loss=latest_loss,
                )
                _atomic_checkpoint(payload, output_dir / name)
                _atomic_checkpoint(payload, output_dir / "latest.pt")
                print(
                    f"checkpoint_saved={output_dir / name} "
                    f"latest_checkpoint={output_dir / 'latest.pt'}"
                )
            if thermal_stop:
                return
    print(f"Completed experiment_step={experiment_step}, loss={latest_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(validate_experiment(args.config, instantiate_model=True), indent=2))
        return
    run_training(args.config)


if __name__ == "__main__":
    main()
