import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

if __package__:
    from evaluation.metrics import (
        evaluate_response,
        repetition_ratio,
        validate_checks,
    )
else:
    # Direct execution places evaluation/ rather than the repository root on
    # sys.path, so import the sibling module without changing installation.
    from metrics import evaluate_response, repetition_ratio, validate_checks
from vasu.config import ModelConfig, get_vasu_60m_config
from vasu.data.prompt_templates import get_prompt_formatter
from vasu.inference.generate import generate
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


CONFIG_FILE = Path("evaluation/checkpoint_config.json")
PROMPTS_FILE = Path("evaluation/prompts.json")
SCORES_FILE = Path("evaluation/manual_scores.json")

TEXT_OUTPUT_FILE = Path("evaluation/checkpoint_comparison_corrected.txt")
JSON_OUTPUT_FILE = Path("evaluation/checkpoint_comparison_corrected.json")
SUMMARY_OUTPUT_FILE = Path(
    "evaluation/checkpoint_score_summary_corrected.json"
)

REQUIRED_SCORE_CRITERIA = (
    "relevance",
    "factuality",
    "instruction_following",
    "fluency",
    "repetition_control",
)


def clean_response(response: str) -> str:
    stop_markers = [
        "User:",
        "Assistant:",
        "Instruction:",
        "Instructions:",
        "### Instruction:",
        "### Response:",
        "Response:",
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0]

    return response.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_model(
    checkpoint_path: str,
    model_config_name: str,
    device: torch.device,
) -> VASUModel:
    model_config = resolve_model_config(model_config_name)
    model = VASUModel(model_config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def resolve_model_config(model_config_name: str) -> ModelConfig:
    """Resolve architecture IDs without loading a checkpoint."""
    if model_config_name == "vasu_31m":
        return ModelConfig()
    if model_config_name == "vasu_60m":
        return get_vasu_60m_config()
    raise ValueError(f"Unknown model config: {model_config_name}")


def resolve_checkpoint_entry(
    entry: str | dict[str, str],
) -> tuple[str, str, str]:
    """Support legacy path strings and typed checkpoint entries."""
    if isinstance(entry, str):
        return entry, "vasu_31m", "alpaca"

    checkpoint_path = entry.get("path")
    model_config_name = entry.get("model_config", "vasu_31m")
    prompt_format = entry.get("prompt_format", "alpaca")
    if not checkpoint_path:
        raise ValueError("Checkpoint entry is missing a path.")
    # Validate early so configuration errors fail before checkpoint loading.
    get_prompt_formatter(prompt_format)
    return checkpoint_path, model_config_name, prompt_format


def select_checkpoints(
    checkpoints: dict[str, str | dict[str, str]],
    requested_ids: Sequence[str] | None,
) -> dict[str, str | dict[str, str]]:
    """Return requested entries in CLI order, or all entries by default."""
    if not requested_ids:
        return checkpoints

    unknown = [name for name in requested_ids if name not in checkpoints]
    if unknown:
        valid = ", ".join(checkpoints)
        raise ValueError(
            f"Unknown checkpoint ID(s): {', '.join(unknown)}. "
            f"Valid checkpoint IDs: {valid}."
        )

    return {name: checkpoints[name] for name in requested_ids}


def effective_generation_config(
    configured: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    """Build the complete reportable generation configuration."""
    if deterministic:
        return {
            "mode": "greedy",
            "do_sample": False,
            "max_new_tokens": configured["max_new_tokens"],
            "temperature": None,
            "top_k": None,
            "top_p": None,
        }

    return {
        "mode": "sampling",
        "do_sample": True,
        "max_new_tokens": configured["max_new_tokens"],
        "temperature": configured.get("temperature", 0.8),
        "top_k": configured.get("top_k", 40),
        "top_p": configured.get("top_p", 0.9),
    }


def generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Avoid passing sampling controls when greedy decoding is selected."""
    kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": config["do_sample"],
    }
    if config["do_sample"]:
        kwargs.update(
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
        )
    return kwargs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare VASU checkpoints on the fixed prompt suite."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="ID",
        help="Evaluate only the named checkpoint IDs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--output-label",
        help=(
            "Write reports with this safe suffix instead of the default "
            "'corrected' suffix."
        ),
    )
    return parser.parse_args(argv)


def resolve_report_paths(
    output_label: str | None,
) -> tuple[Path, Path, Path]:
    label = output_label or "corrected"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(
            "output label may contain only letters, numbers, underscores, "
            "and hyphens"
        )
    return (
        Path(f"evaluation/checkpoint_comparison_{label}.txt"),
        Path(f"evaluation/checkpoint_comparison_{label}.json"),
        Path(f"evaluation/checkpoint_score_summary_{label}.json"),
    )


def count_words(text: str) -> int:
    return len(text.split())


def count_lines(text: str) -> int:
    return len(
        [
            line
            for line in text.splitlines()
            if line.strip()
        ]
    )


def basic_response_stats(response: str) -> dict[str, Any]:
    return {
        "characters": len(response),
        "words": count_words(response),
        "lines": count_lines(response),
        "repetition_ratio": round(repetition_ratio(response), 4),
        "empty": len(response.strip()) == 0,
    }


def get_manual_score(
    scores: dict[str, Any],
    checkpoint_name: str,
    prompt_id: str,
) -> dict[str, int] | None:
    checkpoint_scores = scores.get("scores", {}).get(
        checkpoint_name,
        {},
    )

    prompt_scores = checkpoint_scores.get(prompt_id)

    if not prompt_scores:
        return None

    if not isinstance(prompt_scores, dict):
        raise ValueError(
            f"Manual score for {checkpoint_name}/{prompt_id} "
            "must be an object."
        )

    missing = [
        criterion
        for criterion in REQUIRED_SCORE_CRITERIA
        if criterion not in prompt_scores
    ]
    extra = [
        criterion
        for criterion in prompt_scores
        if criterion not in REQUIRED_SCORE_CRITERIA
    ]

    if missing or extra:
        raise ValueError(
            f"Invalid criteria for {checkpoint_name}/{prompt_id}. "
            f"Missing: {missing or 'none'}; extra: {extra or 'none'}."
        )

    for criterion in REQUIRED_SCORE_CRITERIA:
        value = prompt_scores[criterion]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Score {checkpoint_name}/{prompt_id}/{criterion} "
                "must be a number from 0 to 5."
            )
        if not 0 <= value <= 5:
            raise ValueError(
                f"Score {checkpoint_name}/{prompt_id}/{criterion} "
                f"must be from 0 to 5; got {value}."
            )

    # Return criteria in a stable order in both JSON and text reports.
    return {
        criterion: prompt_scores[criterion]
        for criterion in REQUIRED_SCORE_CRITERIA
    }


def average_score(score: dict[str, int] | None) -> float | None:
    if not score:
        return None

    values = list(score.values())

    if not values:
        return None

    return round(sum(values) / len(values), 3)


def build_score_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for checkpoint_result in results:
        name = checkpoint_result["name"]

        scores = []
        automatic_scores: list[float] = []
        automatic_passed_checks = 0
        automatic_total_checks = 0

        for item in checkpoint_result.get("generations", []):
            avg = item.get("manual_score_average")

            if avg is not None:
                scores.append(avg)

            automatic = item.get("automatic_evaluation", {})
            automatic_score = automatic.get("score")
            if automatic_score is not None:
                automatic_scores.append(float(automatic_score))
            automatic_passed_checks += int(
                automatic.get("passed_checks", 0)
            )
            automatic_total_checks += int(
                automatic.get("total_checks", 0)
            )

        if scores:
            summary[name] = {
                "scored_prompts": len(scores),
                "average_score": round(
                    sum(scores) / len(scores),
                    3,
                ),
            }
        else:
            summary[name] = {
                "scored_prompts": 0,
                "average_score": None,
            }

        summary[name].update(
            {
                "automatically_evaluated_prompts": len(automatic_scores),
                "automatic_average_score": (
                    round(sum(automatic_scores) / len(automatic_scores), 3)
                    if automatic_scores
                    else None
                ),
                "automatic_passed_checks": automatic_passed_checks,
                "automatic_total_checks": automatic_total_checks,
            }
        )

    return summary


def format_text_report(
    results: list[dict[str, Any]],
    score_summary: dict[str, Any],
    generation_config: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("VASU CHECKPOINT COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Generation config: {generation_config}")
    lines.append("")

    lines.append("SCORE SUMMARY")
    lines.append("-" * 80)

    for name, summary in score_summary.items():
        average = summary["average_score"]
        average_text = "not scored" if average is None else str(average)
        lines.append(
            f"{name}: "
            f"average_score={average_text} "
            f"scored_prompts={summary['scored_prompts']} "
            f"automatic_average_score="
            f"{summary['automatic_average_score']} "
            f"automatically_evaluated_prompts="
            f"{summary['automatically_evaluated_prompts']} "
            f"automatic_checks={summary['automatic_passed_checks']}/"
            f"{summary['automatic_total_checks']}"
        )

    for checkpoint_result in results:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"CHECKPOINT: {checkpoint_result['name']}")
        lines.append(f"PATH: {checkpoint_result['path']}")
        lines.append(
            f"MODEL CONFIG: {checkpoint_result['model_config']}"
        )
        lines.append(
            f"PROMPT FORMAT: {checkpoint_result['prompt_format']}"
        )
        lines.append("=" * 80)

        if checkpoint_result.get("skipped"):
            lines.append("SKIPPED: checkpoint file not found.")
            continue

        for item in checkpoint_result["generations"]:
            lines.append("")
            lines.append("-" * 80)
            lines.append(f"Prompt ID: {item['prompt_id']}")
            lines.append(f"Category: {item['category']}")
            lines.append(f"Prompt: {item['prompt']}")
            lines.append(f"Expected: {item['expected_behavior']}")
            lines.append(f"Time: {item['time_seconds']:.2f}s")
            lines.append(f"Stats: {item['stats']}")

            if item["manual_score"] is not None:
                lines.append(
                    f"Manual score: {item['manual_score']}"
                )
                lines.append(
                    f"Manual average: "
                    f"{item['manual_score_average']}"
                )
            else:
                lines.append("Manual score: not scored")

            automatic = item.get(
                "automatic_evaluation",
                {
                    "checks": {},
                    "passed_checks": 0,
                    "total_checks": 0,
                    "score": None,
                },
            )
            automatic_score = automatic.get("score")
            if automatic_score is None:
                lines.append("Automatic score: not configured")
            else:
                lines.append(
                    f"Automatic score: {automatic_score:.3f} "
                    f"({automatic['passed_checks']}/"
                    f"{automatic['total_checks']} checks)"
                )
                failed = [
                    (name, result)
                    for name, result in automatic["checks"].items()
                    if not result["passed"]
                ]
                if failed:
                    lines.append("Failed automatic checks:")
                    for check_name, result in failed:
                        lines.append(
                            f"- {check_name}: {result['details']}"
                        )

            lines.append("")
            lines.append(item["response"])

    return "\n".join(lines)


def validate_prompt_items(prompts: Any) -> None:
    """Validate automatic checks before any tokenizer/model work begins."""
    if not isinstance(prompts, list):
        raise ValueError("prompt suite must be a list")
    for index, prompt_item in enumerate(prompts):
        if not isinstance(prompt_item, dict):
            raise ValueError(f"prompt item {index} must be an object")
        prompt_id = prompt_item.get("id", f"index {index}")
        try:
            validate_checks(prompt_item.get("checks", {}))
        except ValueError as error:
            raise ValueError(
                f"Invalid automatic checks for prompt {prompt_id}: {error}"
            ) from error


def build_generation_result(
    prompt_item: dict[str, Any],
    response: str,
    elapsed: float,
    manual_score: dict[str, int] | None,
) -> dict[str, Any]:
    """Build one backward-compatible report entry with automatic checks."""
    return {
        "prompt_id": prompt_item["id"],
        "category": prompt_item["category"],
        "prompt": prompt_item["prompt"],
        "expected_behavior": prompt_item["expected_behavior"],
        "response": response,
        "time_seconds": round(elapsed, 4),
        "stats": basic_response_stats(response),
        "manual_score": manual_score,
        "manual_score_average": average_score(manual_score),
        "automatic_evaluation": evaluate_response(prompt_item, response),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        text_output_file, json_output_file, summary_output_file = (
            resolve_report_paths(args.output_label)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = load_json(CONFIG_FILE)
    prompts = load_json(PROMPTS_FILE)
    try:
        validate_prompt_items(prompts)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if SCORES_FILE.exists():
        manual_scores = load_json(SCORES_FILE)
    else:
        manual_scores = {
            "scores": {},
        }

    all_checkpoints: dict[str, str | dict[str, str]] = config[
        "checkpoints"
    ]
    try:
        checkpoints = select_checkpoints(
            all_checkpoints,
            args.checkpoints,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    generation_config = effective_generation_config(
        config["generation"],
        deterministic=args.deterministic,
    )
    generate_kwargs = generation_kwargs(generation_config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")

    results: list[dict[str, Any]] = []

    print(f"Using device: {device}")
    print(f"Generation config: {generation_config}")

    for name, checkpoint_entry in checkpoints.items():
        (
            checkpoint_path,
            model_config_name,
            prompt_format,
        ) = resolve_checkpoint_entry(checkpoint_entry)
        path = Path(checkpoint_path)

        checkpoint_result: dict[str, Any] = {
            "name": name,
            "path": checkpoint_path,
            "model_config": model_config_name,
            "prompt_format": prompt_format,
            "generation_config": generation_config,
            "generations": [],
        }

        if not path.exists():
            checkpoint_result["skipped"] = True
            results.append(checkpoint_result)
            continue

        print(f"\nLoading checkpoint: {name}")

        model = load_model(
            checkpoint_path=checkpoint_path,
            model_config_name=model_config_name,
            device=device,
        )

        for prompt_item in prompts:
            prompt = prompt_item["prompt"]
            prompt_id = prompt_item["id"]

            print(f"Generating [{name}] -> {prompt_id}")

            start = time.time()

            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                prompt_format=prompt_format,
                **generate_kwargs,
            )

            elapsed = time.time() - start
            response = clean_response(response)

            score = get_manual_score(
                scores=manual_scores,
                checkpoint_name=name,
                prompt_id=prompt_id,
            )

            checkpoint_result["generations"].append(
                build_generation_result(
                    prompt_item=prompt_item,
                    response=response,
                    elapsed=elapsed,
                    manual_score=score,
                )
            )

        results.append(checkpoint_result)

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    score_summary = build_score_summary(results)

    text_output_file.write_text(
        format_text_report(results, score_summary, generation_config),
        encoding="utf-8",
    )

    save_json(json_output_file, results)
    save_json(summary_output_file, score_summary)

    print(f"\nSaved text report to: {text_output_file}")
    print(f"Saved JSON report to: {json_output_file}")
    print(f"Saved score summary to: {summary_output_file}")


if __name__ == "__main__":
    main()
