"""Likelihood-first evaluation for VASU-60M factual continued pretraining."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable

import torch
import torch.nn.functional as F

from evaluation.evaluate_factual_cpt_baseline import (
    FlatTokenValidationDataset,
    _load_parent_model,
    response_metrics,
    validation_loss,
)
from evaluation.metrics import normalized_words, repetition_ratio
from vasu.config import get_vasu_60m_config
from vasu.data.preparation.reporting import sha256_file
from vasu.inference.generate import generate_token_ids
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import ManifestTokenDataset


DEFAULT_CONFIG = Path("configs/evaluation/vasu_60m_factual_cpt_v2.json")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_checkpoint_hash(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual}"
        )
    return actual


def validate_result_provenance(result: dict[str, Any]) -> None:
    """Require every published result to bind all reproducibility artifacts."""

    for section in ("checkpoint", "benchmark", "evaluation_config", "tokenizer"):
        value = result.get(section)
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise ValueError(f"result is missing {section} SHA-256")
        digest = value["sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"result has invalid {section} SHA-256")


def rank_option_scores(option_scores: list[dict[str, float | int | str]]) -> dict[str, Any]:
    """Rank MC options by total and per-token conditional log-likelihood."""

    if len(option_scores) < 2:
        raise ValueError("multiple-choice scoring requires at least two options")
    raw = sorted(
        range(len(option_scores)),
        key=lambda index: float(option_scores[index]["total_log_likelihood"]),
        reverse=True,
    )
    normalized = sorted(
        range(len(option_scores)),
        key=lambda index: float(option_scores[index]["mean_log_likelihood"]),
        reverse=True,
    )
    return {
        "predicted_index": raw[0],
        "length_normalized_predicted_index": normalized[0],
        "winning_margin": (
            float(option_scores[raw[0]]["total_log_likelihood"])
            - float(option_scores[raw[1]]["total_log_likelihood"])
        ),
        "length_normalized_winning_margin": (
            float(option_scores[normalized[0]]["mean_log_likelihood"])
            - float(option_scores[normalized[1]]["mean_log_likelihood"])
        ),
    }


def _normalized_match(text: str) -> str:
    words = normalized_words(text)
    return " ".join(words)


@torch.inference_mode()
def conditional_log_likelihood(
    model: torch.nn.Module,
    prefix_ids: list[int],
    target_ids: list[int],
    device: torch.device,
) -> dict[str, float | int]:
    """Score target IDs directly under P(target | prefix)."""

    if not prefix_ids:
        raise ValueError("prefix_ids must not be empty")
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    combined = prefix_ids + target_ids
    if len(combined) > get_vasu_60m_config().max_seq_len:
        raise ValueError("scored sequence exceeds the model context length")
    inputs = torch.tensor([combined[:-1]], dtype=torch.long, device=device)
    logits = model(inputs)
    target_start = len(prefix_ids) - 1
    target_logits = logits[:, target_start:, :]
    targets = torch.tensor([target_ids], dtype=torch.long, device=device)
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(selected.sum().item())
    token_count = len(target_ids)
    average = total / token_count
    return {
        "total_log_likelihood": total,
        "mean_log_likelihood": average,
        "loss": -average,
        "perplexity": math.exp(min(50.0, -average)),
        "target_tokens": token_count,
    }


def _bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    samples: int,
    statistic: Callable[[list[float]], float] = mean,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(float(statistic(resample)))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return {"estimate": float(statistic(values)), "ci95_low": low, "ci95_high": high}


def _category_summary(
    rows: list[dict[str, Any]],
    numeric_fields: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        category: {
            "examples": len(items),
            **{
                field: mean(float(item[field]) for item in items)
                for field in numeric_fields
            },
        }
        for category, items in sorted(grouped.items())
    }


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": bool(config["do_sample"]),
        "max_new_tokens": int(config["max_new_tokens"]),
        "prompt_format": "plain",
        "use_kv_cache": True,
        "repetition_penalty": float(config["repetition_penalty"]),
    }
    if config["do_sample"]:
        kwargs.update(
            temperature=float(config["temperature"]),
            top_k=int(config["top_k"]),
            top_p=float(config["top_p"]),
        )
    return kwargs


def _generate(
    model: torch.nn.Module,
    tokenizer: VASUTokenizer,
    prompt: str,
    device: torch.device,
    config: dict[str, Any],
    seed: int,
) -> tuple[str, list[int]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    ids = generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        **_generation_kwargs(config),
    )
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), ids


def evaluate_cloze(
    model: torch.nn.Module,
    tokenizer: VASUTokenizer,
    examples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for item in examples:
        prefix_ids = tokenizer.encode(item["prompt"])
        target_ids = tokenizer.encode(item["target"])
        likelihood = conditional_log_likelihood(model, prefix_ids, target_ids, device)
        generated, generated_ids = _generate(
            model,
            tokenizer,
            item["prompt"],
            device,
            {
                "do_sample": False,
                "max_new_tokens": len(target_ids),
                "repetition_penalty": 1.1,
            },
            0,
        )
        target = item["target"].strip()
        exact = generated.strip() == target
        normalized = _normalized_match(generated) == _normalized_match(target)
        rows.append({
            **item,
            "generated": generated,
            "generated_tokens": len(generated_ids),
            "exact_match": exact,
            "normalized_match": normalized,
            **likelihood,
        })
    perplexities = [float(row["perplexity"]) for row in rows]
    return {
        "examples": rows,
        "summary": {
            "count": len(rows),
            "exact_match_accuracy": mean(float(row["exact_match"]) for row in rows),
            "normalized_match_accuracy": mean(float(row["normalized_match"]) for row in rows),
            "mean_target_log_likelihood": mean(float(row["mean_log_likelihood"]) for row in rows),
            "mean_target_perplexity": mean(perplexities),
            "category_results": _category_summary(
                rows,
                ("exact_match", "normalized_match", "mean_log_likelihood", "perplexity"),
            ),
        },
    }


def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer: VASUTokenizer,
    examples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for item in examples:
        prefix_ids = tokenizer.encode(item["prompt"])
        option_scores = []
        for option in item["options"]:
            scored = conditional_log_likelihood(
                model, prefix_ids, tokenizer.encode(option), device
            )
            option_scores.append({"option": option, **scored})
        ranking = rank_option_scores(option_scores)
        rows.append({
            **item,
            "option_scores": option_scores,
            **ranking,
            "correct": ranking["predicted_index"] == item["answer_index"],
            "length_normalized_correct": (
                ranking["length_normalized_predicted_index"] == item["answer_index"]
            ),
        })
    return {
        "examples": rows,
        "summary": {
            "count": len(rows),
            "accuracy": mean(float(row["correct"]) for row in rows),
            "length_normalized_accuracy": mean(
                float(row["length_normalized_correct"]) for row in rows
            ),
            "mean_winning_margin": mean(float(row["winning_margin"]) for row in rows),
            "mean_length_normalized_winning_margin": mean(
                float(row["length_normalized_winning_margin"]) for row in rows
            ),
            "category_results": _category_summary(
                rows,
                ("correct", "length_normalized_correct", "winning_margin"),
            ),
        },
    }


def _ngrams(words: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(words[index:index + size]) for index in range(len(words) - size + 1)]


def aggregate_generated_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distinct = {}
    repeated = {}
    for size in (1, 2, 3):
        # Do not invent n-grams across independent response boundaries.
        grams = [
            gram
            for row in rows
            for gram in _ngrams(normalized_words(row["response"]), size)
        ]
        distinct[f"distinct_{size}"] = len(set(grams)) / len(grams) if grams else 0.0
        repeated[f"repeated_{size}gram_rate"] = (
            1.0 - len(set(grams)) / len(grams) if grams else 0.0
        )
    return {
        "count": len(rows),
        "mean_repetition_ratio": mean(float(row["repetition_ratio"]) for row in rows),
        **distinct,
        **repeated,
        "premature_eos_rate": mean(float(row["premature_eos"]) for row in rows),
        "empty_output_rate": mean(float(row["empty_output"]) for row in rows),
        "mean_generated_tokens": mean(float(row["generated_tokens"]) for row in rows),
        "mean_words": mean(float(row["words"]) for row in rows),
    }


def evaluate_generations(
    model: torch.nn.Module,
    tokenizer: VASUTokenizer,
    examples: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
    seed: int,
    *,
    include_target_likelihood: bool,
) -> dict[str, Any]:
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    rows = []
    for index, item in enumerate(examples):
        response, ids = _generate(
            model, tokenizer, item["prompt"], device, config, seed + index
        )
        metrics = response_metrics(response, len(ids))
        row = {
            **item,
            "response": response,
            **metrics,
            "empty_output": not bool(response.strip()),
            "premature_eos": bool(ids and ids[-1] == eos_id and len(ids) < config["max_new_tokens"]),
            "obvious_incoherence": (
                not bool(response.strip())
                or repetition_ratio(response) > 0.65
                or len(normalized_words(response)) < 3
            ),
        }
        if include_target_likelihood:
            row["target_likelihood"] = conditional_log_likelihood(
                model,
                tokenizer.encode(item["prompt"]),
                tokenizer.encode(item["target"]),
                device,
            )
        rows.append(row)
    summary = aggregate_generated_rows(rows)
    if include_target_likelihood:
        summary["mean_continuation_log_likelihood"] = mean(
            float(row["target_likelihood"]["mean_log_likelihood"]) for row in rows
        )
        summary["mean_continuation_loss"] = -summary["mean_continuation_log_likelihood"]
    else:
        style_keys = ("heading_like", "article_lead_like", "citation_like")
        for key in style_keys:
            summary[f"{key}_rate"] = mean(
                float(row["encyclopedic_style"][key]) for row in rows
            )
        summary["obvious_incoherence_rate"] = mean(
            float(row["obvious_incoherence"]) for row in rows
        )
    return {"examples": rows, "summary": summary}


def _confidence_intervals(
    result: dict[str, Any], *, seed: int, samples: int
) -> dict[str, Any]:
    cloze = result["cloze"]["examples"]
    mc = result["multiple_choice"]["examples"]
    return {
        "cloze_exact_accuracy": _bootstrap_ci(
            [float(row["exact_match"]) for row in cloze], seed=seed, samples=samples
        ),
        "cloze_normalized_accuracy": _bootstrap_ci(
            [float(row["normalized_match"]) for row in cloze],
            seed=seed + 1,
            samples=samples,
        ),
        "target_mean_log_likelihood": _bootstrap_ci(
            [float(row["mean_log_likelihood"]) for row in cloze],
            seed=seed + 2,
            samples=samples,
        ),
        "multiple_choice_accuracy": _bootstrap_ci(
            [float(row["correct"]) for row in mc], seed=seed + 3, samples=samples
        ),
        "multiple_choice_length_normalized_accuracy": _bootstrap_ci(
            [float(row["length_normalized_correct"]) for row in mc],
            seed=seed + 4,
            samples=samples,
        ),
    }


def evaluate_checkpoint(
    *,
    checkpoint_id: str,
    checkpoint: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    benchmark: dict[str, Any],
    tokenizer: VASUTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint["path"])
    checkpoint_sha = validate_checkpoint_hash(checkpoint_path, checkpoint["sha256"])
    model, global_step = _load_parent_model(checkpoint_path, device)
    sequence_length = get_vasu_60m_config().max_seq_len
    validation_config = config["validation"]
    fineweb = ManifestTokenDataset(
        validation_config["fineweb_manifest"],
        "validation",
        sequence_length,
        logical_start=0,
        logical_end=validation_config["fineweb_samples"] * sequence_length + 1,
    )
    wikipedia = FlatTokenValidationDataset(
        validation_config["wikimedia_path"], sequence_length
    )
    batch_size = int(validation_config["batch_size"])
    result = {
        "format_version": "vasu_factual_cpt_result_v2",
        "checkpoint_id": checkpoint_id,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "global_step": global_step,
        },
        "benchmark": {
            "path": config["benchmark_path"],
            "sha256": sha256_file(Path(config["benchmark_path"])),
            "seed": benchmark["seed"],
        },
        "evaluation_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "tokenizer": {
            "path": config["tokenizer_path"],
            "sha256": sha256_file(Path(config["tokenizer_path"])),
        },
        "generation_settings": config["generation"],
        "validation": {
            "fineweb": validation_loss(model, fineweb, batch_size=batch_size, device=device),
            "wikimedia": validation_loss(model, wikipedia, batch_size=batch_size, device=device),
        },
        "cloze": evaluate_cloze(model, tokenizer, benchmark["cloze"], device),
        "multiple_choice": evaluate_multiple_choice(
            model, tokenizer, benchmark["multiple_choice"], device
        ),
        "continuation": {},
        "open_ended": {},
        "optimizer_updates_performed": 0,
    }
    for offset, (mode, generation_config) in enumerate(config["generation"].items()):
        result["continuation"][mode] = evaluate_generations(
            model,
            tokenizer,
            benchmark["continuations"],
            device,
            generation_config,
            config["seed"] + offset * 10_000,
            include_target_likelihood=True,
        )
        result["open_ended"][mode] = evaluate_generations(
            model,
            tokenizer,
            benchmark["open_ended"],
            device,
            generation_config,
            config["seed"] + 20_000 + offset * 10_000,
            include_target_likelihood=False,
        )
    result["confidence_intervals"] = _confidence_intervals(
        result, seed=config["seed"], samples=config["bootstrap_samples"]
    )
    validate_result_provenance(result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _paired_difference_ci(
    candidate: list[float],
    parent: list[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, float | bool]:
    if len(candidate) != len(parent):
        raise ValueError("paired result lengths differ")
    differences = [left - right for left, right in zip(candidate, parent, strict=True)]
    ci = _bootstrap_ci(differences, seed=seed, samples=samples)
    return {**ci, "distinguishable_from_zero": ci["ci95_low"] > 0 or ci["ci95_high"] < 0}


def build_comparison(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    parent = results["parent"]
    comparisons = {}
    for checkpoint_id in ("best", "latest"):
        candidate = results[checkpoint_id]
        comparisons[checkpoint_id] = {
            "fineweb_loss_absolute_change": (
                candidate["validation"]["fineweb"]["loss"]
                - parent["validation"]["fineweb"]["loss"]
            ),
            "fineweb_loss_relative_change": (
                candidate["validation"]["fineweb"]["loss"]
                / parent["validation"]["fineweb"]["loss"] - 1.0
            ),
            "wikimedia_loss_absolute_change": (
                candidate["validation"]["wikimedia"]["loss"]
                - parent["validation"]["wikimedia"]["loss"]
            ),
            "wikimedia_loss_relative_change": (
                candidate["validation"]["wikimedia"]["loss"]
                / parent["validation"]["wikimedia"]["loss"] - 1.0
            ),
            "cloze_exact_difference_ci": _paired_difference_ci(
                [float(row["exact_match"]) for row in candidate["cloze"]["examples"]],
                [float(row["exact_match"]) for row in parent["cloze"]["examples"]],
                seed=config["seed"],
                samples=config["bootstrap_samples"],
            ),
            "cloze_normalized_difference_ci": _paired_difference_ci(
                [float(row["normalized_match"]) for row in candidate["cloze"]["examples"]],
                [float(row["normalized_match"]) for row in parent["cloze"]["examples"]],
                seed=config["seed"] + 1,
                samples=config["bootstrap_samples"],
            ),
            "target_log_likelihood_difference_ci": _paired_difference_ci(
                [float(row["mean_log_likelihood"]) for row in candidate["cloze"]["examples"]],
                [float(row["mean_log_likelihood"]) for row in parent["cloze"]["examples"]],
                seed=config["seed"] + 2,
                samples=config["bootstrap_samples"],
            ),
            "multiple_choice_difference_ci": _paired_difference_ci(
                [float(row["correct"]) for row in candidate["multiple_choice"]["examples"]],
                [float(row["correct"]) for row in parent["multiple_choice"]["examples"]],
                seed=config["seed"] + 3,
                samples=config["bootstrap_samples"],
            ),
            "multiple_choice_length_normalized_difference_ci": _paired_difference_ci(
                [float(row["length_normalized_correct"]) for row in candidate["multiple_choice"]["examples"]],
                [float(row["length_normalized_correct"]) for row in parent["multiple_choice"]["examples"]],
                seed=config["seed"] + 4,
                samples=config["bootstrap_samples"],
            ),
        }
    return {
        "format_version": "vasu_factual_cpt_comparison_v2",
        "benchmark_sha256": parent["benchmark"]["sha256"],
        "evaluation_config_sha256": parent["evaluation_config"]["sha256"],
        "comparisons_to_parent": comparisons,
        "best_vs_latest": {
            "fineweb_loss_difference": (
                results["best"]["validation"]["fineweb"]["loss"]
                - results["latest"]["validation"]["fineweb"]["loss"]
            ),
            "wikimedia_loss_difference": (
                results["best"]["validation"]["wikimedia"]["loss"]
                - results["latest"]["validation"]["wikimedia"]["loss"]
            ),
        },
        "optimizer_updates_performed": 0,
    }


def _readable_report(result: dict[str, Any]) -> str:
    lines = [
        f"Checkpoint: {result['checkpoint_id']}",
        f"Path: {result['checkpoint']['path']}",
        f"SHA-256: {result['checkpoint']['sha256']}",
        f"FineWeb loss: {result['validation']['fineweb']['loss']:.6f}",
        f"Wikimedia loss: {result['validation']['wikimedia']['loss']:.6f}",
        f"Cloze: {json.dumps(result['cloze']['summary'], sort_keys=True)}",
        f"Multiple choice: {json.dumps(result['multiple_choice']['summary'], sort_keys=True)}",
        "",
    ]
    for section in ("continuation", "open_ended"):
        for mode, payload in result[section].items():
            lines.extend((
                f"{section} / {mode}: {json.dumps(payload['summary'], sort_keys=True)}",
                "-" * 80,
            ))
            for item in payload["examples"]:
                lines.extend((f"[{item['id']}] {item['prompt']}", item["response"], ""))
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("format_version") != "vasu_factual_cpt_evaluation_v2":
        raise ValueError("unsupported factual-CPT v2 evaluation configuration")
    benchmark_path = Path(config["benchmark_path"])
    benchmark = _load_json(benchmark_path)
    tokenizer = VASUTokenizer()
    tokenizer.load(config["tokenizer_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, Any]] = {}
    output_directory = Path(config["results_directory"])
    for checkpoint_id, checkpoint in config["checkpoints"].items():
        print(f"Evaluating {checkpoint_id}: {checkpoint['path']}")
        result = evaluate_checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint=checkpoint,
            config_path=config_path,
            config=config,
            benchmark=benchmark,
            tokenizer=tokenizer,
            device=device,
        )
        results[checkpoint_id] = result
        _atomic_json(output_directory / f"factual_cpt_v2_{checkpoint_id}.json", result)
        _atomic_text(
            output_directory / f"factual_cpt_v2_{checkpoint_id}.txt",
            _readable_report(result),
        )
        print(
            f"{checkpoint_id}: fineweb={result['validation']['fineweb']['loss']:.6f} "
            f"wikimedia={result['validation']['wikimedia']['loss']:.6f} "
            f"cloze={result['cloze']['summary']['normalized_match_accuracy']:.3f} "
            f"mc={result['multiple_choice']['summary']['accuracy']:.3f}"
        )
    comparison = build_comparison(results, config)
    _atomic_json(output_directory / "factual_cpt_v2_comparison.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    comparison = run(args.config)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
