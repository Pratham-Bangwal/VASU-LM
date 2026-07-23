"""Run a versioned VASU internal capability suite without model training."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from .checkpoint_loader import load_checkpoint_model, load_tokenizer
from .generation import generation_settings, run_generation
from .registry import load_checkpoint_registry, load_suite, sha256_file
from .reporting import (
    append_jsonl_atomic,
    atomic_json,
    promotion_report,
    summarize,
    validate_human_review,
)
from .scoring import score_task


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seeds(mode: str, values: list[int] | None) -> list[int | None]:
    if mode == "greedy":
        return [None]
    return values or [11, 23, 37, 41, 53]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--mode", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--parent")
    parser.add_argument(
        "--promotion-type",
        choices=("continued_pretraining", "instruction", "conversation"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--human-review-results",
        type=Path,
        help="JSON object mapping human task IDs to reviewed result objects.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject ambiguous promotion comparisons before any checkpoint is loaded."""

    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if args.parent is not None and args.promotion_type is None:
        raise ValueError("--parent requires --promotion-type.")
    if args.parent is None and args.promotion_type is not None:
        raise ValueError("--promotion-type requires --parent.")


def validate_resume_manifest(
    existing: dict[str, object], current: dict[str, object]
) -> None:
    """Reject resume when generation or promotion semantics changed."""

    for key in (
        "suite_sha256",
        "checkpoint_ids",
        "generation_settings",
        "seeds",
        "promotion",
        "human_review_results_sha256",
    ):
        if existing.get(key) != current[key]:
            raise ValueError(f"Cannot resume incompatible run: {key} differs.")


def load_human_review_results(
    path: Path | None, tasks: list[object]
) -> dict[str, dict[str, object]]:
    """Load explicit human decisions without conflating them with model scores."""

    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--human-review-results must contain a JSON object.")
    human_ids = {
        getattr(task, "identifier")
        for task in tasks
        if getattr(task, "metric_kind") == "human"
    }
    unknown = set(raw) - human_ids
    if unknown:
        raise ValueError(f"Human review file names non-human/unknown tasks: {sorted(unknown)}")
    for task_id, review in raw.items():
        try:
            validate_human_review(review)
        except ValueError as error:
            raise ValueError(f"Invalid human review for {task_id!r}: {error}") from error
    return raw


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    validate_args(args)
    registry_path = Path("evaluation/checkpoint_config.json")
    registry = load_checkpoint_registry(registry_path)
    selected = []
    for identifier in args.checkpoints:
        if identifier not in registry:
            raise ValueError(f"Unknown checkpoint {identifier!r}; valid IDs: {', '.join(registry)}")
        selected.append(registry[identifier])
    if args.parent is not None:
        if args.parent not in registry:
            raise ValueError(f"Unknown parent checkpoint {args.parent!r}.")
        if args.parent not in {entry.identifier for entry in selected}:
            selected.insert(0, registry[args.parent])
    suite_raw, tasks = load_suite(args.suite)
    human_reviews = load_human_review_results(args.human_review_results, tasks)
    promotion: dict[str, object] | None = None
    if args.promotion_type is not None:
        gates = suite_raw.get("promotion_gates")
        if not isinstance(gates, dict) or args.promotion_type not in gates:
            raise ValueError(
                f"Suite does not define the {args.promotion_type!r} promotion gate."
            )
        gate = gates[args.promotion_type]
        gate_json = json.dumps(gate, sort_keys=True, separators=(",", ":"))
        promotion = {
            "type": args.promotion_type,
            "gate": gate,
            "gate_sha256": hashlib.sha256(gate_json.encode("utf-8")).hexdigest(),
        }
    if args.categories:
        tasks = [task for task in tasks if task.category in set(args.categories)]
    if args.tasks:
        tasks = [task for task in tasks if task.identifier in set(args.tasks)]
    if not tasks:
        raise ValueError("Task filters selected no suite tasks.")
    output = args.output_dir
    if output.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output)
    manifest_path = output / "run_manifest.json"
    device = _device(args.device)
    settings = generation_settings(args.mode, args.max_new_tokens)
    manifest = {
        "framework_version": 1,
        "suite": str(args.suite), "suite_sha256": sha256_file(args.suite),
        "suite_version": suite_raw["suite_version"], "checkpoint_ids": [entry.identifier for entry in selected],
        "checkpoints": [
            {"id": entry.identifier, "path": entry.path, "sha256": sha256_file(Path(entry.path)), "model_config": entry.model_config, "prompt_format": entry.prompt_format}
            for entry in selected
        ],
        "generation_settings": settings, "seeds": _seeds(args.mode, args.seeds),
        "promotion": promotion,
        "human_review_results_sha256": (
            sha256_file(args.human_review_results)
            if args.human_review_results is not None
            else None
        ),
        "device": str(device), "torch": torch.__version__, "timestamp": datetime.now(UTC).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "git_commit": _git_commit(),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Output exists: {output}; use --resume or --overwrite.")
        validate_resume_manifest(existing, manifest)
        manifest = existing
    else:
        atomic_json(manifest_path, manifest, overwrite=args.overwrite)
    tokenizer, tokenizer_hash = load_tokenizer()
    manifest["tokenizer_sha256"] = tokenizer_hash
    atomic_json(manifest_path, manifest, overwrite=True)
    results_path = output / "per_task_results.jsonl"
    completed: set[tuple[str, str, int | None]] = set()
    if results_path.exists():
        completed = {
            (record["checkpoint_id"], record["task_id"], record["seed"])
            for record in (
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    for entry in selected:
        model, checkpoint_metadata = load_checkpoint_model(entry, device)
        for task in tasks:
            for seed in _seeds(args.mode, args.seeds):
                if (entry.identifier, task.identifier, seed) in completed:
                    continue
                response, seconds = run_generation(model=model, tokenizer=tokenizer, prompt=task.prompt, prompt_format=entry.prompt_format, device=device, settings=settings, seed=seed)
                record = {
                    "checkpoint_id": entry.identifier,
                    "checkpoint": checkpoint_metadata,
                    "task_id": task.identifier,
                    "category": task.category,
                    "seed": seed,
                    "response": response,
                    "runtime_seconds": seconds,
                    "score": score_task(task, response),
                }
                if task.identifier in human_reviews:
                    record["human_review"] = human_reviews[task.identifier]
                append_jsonl_atomic(results_path, record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    summary = summarize(records)
    atomic_json(output / "category_summary.json", summary, overwrite=True)
    if args.parent is not None:
        assert promotion is not None
        atomic_json(
            output / "promotion_report.json",
            promotion_report(
                records,
                parent=args.parent,
                candidate=selected[-1].identifier,
                promotion_type=args.promotion_type,
                gate=promotion["gate"],
            ),
            overwrite=True,
        )
    lines = [f"VASU internal capability suite: {suite_raw['suite_version']}", "", "Objective accuracy:"]
    lines.extend(f"- {name}: {data['accuracy']:.3f} ({data['count']} tasks)" for name, data in summary["objective_categories"].items())
    lines.append("\nHeuristic metrics are reported separately and are not quality scores.")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
