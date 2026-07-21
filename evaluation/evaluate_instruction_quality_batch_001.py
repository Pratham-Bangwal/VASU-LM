"""Compare the UltraChat base with instruction-quality Batch 001."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any

import torch
from torch.utils.data import DataLoader

from vasu.config import get_vasu_60m_config
from vasu.inference.generate import generate_token_ids
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss


BENCHMARK = Path("evaluation/benchmarks/ultrachat_promotion_v1.json")
TOKENIZER = Path("assets/tokenizer.json")
RESULT = Path(
    "evaluation/results/instruction_quality_batch_001_comparison_v1.json"
)
MANUAL_REVIEW = Path(
    "evaluation/results/"
    "instruction_quality_batch_001_comparison_v1_manual_review.txt"
)

CHECKPOINTS = {
    "ultrachat_base": Path(
        "checkpoints/vasu_60m/"
        "ultrachat_masked_v2_from_alpaca_v3/best.pt"
    ),
    "instruction_quality_candidate": Path(
        "checkpoints/vasu_60m/"
        "instruction_quality_batch_001_from_ultrachat_v2/best.pt"
    ),
}
EXPECTED_HASHES = {
    "ultrachat_base": (
        "eea7669a4728e3fbc8e35320bce1c368"
        "3c4fa9a584d1eb2da566e8fdafab36a7"
    ),
    "instruction_quality_candidate": (
        "4f377fc49fef2382c531b5bb28731fa9"
        "664bac217256b8a76dcf1427f4d39d8e"
    ),
}
TOKENIZER_SHA = "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
ULTRACHAT_TOKEN_SHA = "12a9fd0d20ee3f136fbbc38d19636bdc9c0ab8d0cbcd8b384a11aaa7003767da"
ULTRACHAT_MASK_SHA = "8fc6f535addd9b8f42fb550898cb2d82637f759698107e09c3544d9f7e1f5043"
SEED = 42
MODES = {
    "greedy": {
        "do_sample": False,
        "max_new_tokens": 128,
        "repetition_penalty": 1.1,
    },
    "controlled_sampling": {
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 128,
        "seed": SEED,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def list_counts(text: str) -> tuple[int, int]:
    bullets = len(re.findall(r"(?m)^\s*[-*+]\s+", text))
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))
    return bullets, numbered


def distinct_n(words: list[str], n: int) -> float:
    ngrams = list(zip(*(words[offset:] for offset in range(n))))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def repeated_ngram_rate(words: list[str], n: int = 3) -> float:
    ngrams = list(zip(*(words[offset:] for offset in range(n))))
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def format_compliance(item: dict[str, Any], response: str) -> dict[str, bool]:
    constraints = item.get("constraints", {})
    words = normalized_words(response)
    sentences = sentence_count(response)
    bullets, numbered = list_counts(response)
    strict_checks: list[bool] = []
    relaxed_checks: list[bool] = []
    if "sentence_count" in constraints:
        expected = int(constraints["sentence_count"])
        strict_checks.append(sentences == expected)
        relaxed_checks.append(abs(sentences - expected) <= 1)
    if "bullet_count" in constraints:
        expected = int(constraints["bullet_count"])
        strict_checks.append(bullets == expected)
        relaxed_checks.append(abs(bullets - expected) <= 1 and bullets > 0)
    if "numbered_count" in constraints:
        expected = int(constraints["numbered_count"])
        strict_checks.append(numbered == expected)
        relaxed_checks.append(abs(numbered - expected) <= 1 and numbered > 0)
    if "max_words" in constraints:
        maximum = int(constraints["max_words"])
        strict_checks.append(0 < len(words) <= maximum)
        relaxed_checks.append(0 < len(words) <= maximum + 5)
    if "exact_words" in constraints:
        expected = int(constraints["exact_words"])
        strict_checks.append(len(words) == expected)
        relaxed_checks.append(len(words) <= expected + 1)
    if constraints.get("forbid_list"):
        strict_checks.append(bullets == 0 and numbered == 0)
        relaxed_checks.append(bullets == 0 and numbered == 0)
    if "json_keys" in constraints:
        required = set(constraints["json_keys"])
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        strict_checks.append(isinstance(parsed, dict) and set(parsed) == required)
        relaxed_checks.append(isinstance(parsed, dict) and required <= set(parsed))
    if constraints.get("uncertainty_required"):
        folded = response.casefold()
        phrases = ("uncertain", "cannot know", "can't know", "depends", "not possible")
        passed = any(phrase in folded for phrase in phrases)
        strict_checks.append(passed)
        relaxed_checks.append(passed)
    return {
        "strict": bool(strict_checks) and all(strict_checks),
        "relaxed": bool(relaxed_checks) and all(relaxed_checks),
    }


def response_metrics(
    item: dict[str, Any],
    response: str,
    token_ids: list[int],
    eos_id: int,
) -> dict[str, Any]:
    non_eos_ids = [token_id for token_id in token_ids if token_id != eos_id]
    words = normalized_words(response)
    counts = Counter(words)
    repetition = (
        sum(count - 1 for count in counts.values()) / len(words) if words else 0.0
    )
    ended_eos = bool(token_ids and token_ids[-1] == eos_id)
    bullets, numbered = list_counts(response)
    compliance = format_compliance(item, response)
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    return {
        "empty": not bool(response.strip()),
        "ended_with_eos": ended_eos,
        "premature_eos": ended_eos and len(non_eos_ids) < 8,
        "generated_tokens": len(non_eos_ids),
        "words": len(words),
        "sentences": sentence_count(response),
        "repetition_ratio": repetition,
        "distinct_1": distinct_n(words, 1),
        "distinct_2": distinct_n(words, 2),
        "distinct_3": distinct_n(words, 3),
        "repeated_ngram_rate": repeated_ngram_rate(words),
        "strict_format_compliance": compliance["strict"],
        "relaxed_format_compliance": compliance["relaxed"],
        "bullet_count": bullets,
        "numbered_count": numbered,
        "obvious_incoherence": (
            not words or repetition > 0.65 or (0 < len(words) < 3)
        ),
        "article_lead_style": bool(
            re.match(r"^(?:the\s+)?[A-Z][^.!?]{0,60}\s+is\s+(?:an?|the)\b", first_line)
        ),
        "heading_like": bool(
            re.search(r"(?m)^(?:#{1,6}\s+|[A-Z][A-Z\s]{4,}:?$)", response)
        ),
        "citation_like": bool(
            re.search(r"\[(?:\d+|citation needed)\]|\bReferences\b", response, re.I)
        ),
    }


def summarize(generations: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [entry["metrics"] for entry in generations]
    count = len(metrics)
    mean_fields = (
        "generated_tokens", "words", "sentences", "repetition_ratio",
        "distinct_1", "distinct_2", "distinct_3", "repeated_ngram_rate",
    )
    rate_fields = (
        "empty", "premature_eos", "strict_format_compliance",
        "relaxed_format_compliance", "obvious_incoherence",
        "article_lead_style", "heading_like", "citation_like",
    )
    result: dict[str, Any] = {
        field: statistics.fmean(float(row[field]) for row in metrics)
        for field in mean_fields
    }
    result.update({f"{field}_rate": sum(bool(row[field]) for row in metrics) / count for field in rate_fields})
    categories: dict[str, dict[str, float]] = {}
    for category in sorted({entry["category"] for entry in generations}):
        subset = [entry["metrics"] for entry in generations if entry["category"] == category]
        categories[category] = {
            "strict_compliance": sum(row["strict_format_compliance"] for row in subset) / len(subset),
            "relaxed_compliance": sum(row["relaxed_format_compliance"] for row in subset) / len(subset),
        }
    result["category_format_compliance"] = categories
    return result


@torch.no_grad()
def masked_validation_loss(
    model: VASUModel,
    dataset: PackedInstructionDataset,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    weighted_loss = 0.0
    supervised_tokens = 0
    model.eval()
    for inputs, targets, mask in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        token_count = int(mask.sum().item())
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = language_model_loss(model(inputs), targets, mask)
        weighted_loss += float(loss.item()) * token_count
        supervised_tokens += token_count
    return {"loss": weighted_loss / supervised_tokens, "supervised_tokens": supervised_tokens}


def validation_datasets() -> dict[str, PackedInstructionDataset]:
    ultra_metadata = json.loads(Path("data/processed/instruct/ultrachat_masked_v2_metadata.json").read_text(encoding="utf-8"))
    ultra = PackedInstructionDataset(
        "data/processed/instruct/ultrachat_masked_v2.bin",
        "data/processed/instruct/ultrachat_masked_v2_mask.bin",
        256,
        start_record=int(ultra_metadata["train_records"]),
    )
    alpaca_metadata = json.loads(Path("data/processed/instruct/alpaca_masked_v2_metadata.json").read_text(encoding="utf-8"))
    alpaca_records = int(alpaca_metadata["number_of_training_records"])
    alpaca_split = int(alpaca_records * 0.95)
    alpaca = PackedInstructionDataset(
        "data/processed/instruct/alpaca_masked_v2.bin",
        "data/processed/instruct/alpaca_masked_v2_mask.bin",
        256,
        start_record=alpaca_split,
    )
    return {"ultrachat_validation": ultra, "alpaca_validation": alpaca}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_checkpoint(
    name: str,
    path: Path,
    prompts: list[dict[str, Any]],
    tokenizer: VASUTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = VASUModel(get_vasu_60m_config())
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.to(device).eval()
    losses = {dataset_name: masked_validation_loss(model, dataset, device) for dataset_name, dataset in validation_datasets().items()}
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise RuntimeError("tokenizer is missing the [EOS] token")
    mode_results = {}
    for mode_name, mode in MODES.items():
        generations = []
        kwargs = {key: value for key, value in mode.items() if key != "seed"}
        for item in prompts:
            set_seed(SEED)
            token_ids = generate_token_ids(
                model=model,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                device=device,
                prompt_format="alpaca",
                use_kv_cache=True,
                **kwargs,
            )
            response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            generations.append({
                "prompt_id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
                "metrics": response_metrics(item, response, token_ids, eos_id),
            })
        mode_results[mode_name] = {
            "generation_configuration": mode,
            "summary": summarize(generations),
            "generations": generations,
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": name,
        "path": str(path).replace("\\", "/"),
        "sha256": sha256(path),
        "masked_validation_losses": losses,
        "modes": mode_results,
    }


def manual_review_text(results: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> str:
    selected = []
    for category in sorted({item["category"] for item in prompts}):
        selected.extend([item for item in prompts if item["category"] == category][:5])
    lines = ["VASU INSTRUCTION-QUALITY BATCH 001 MANUAL REVIEW", "", "Reviewer: ", "Date: ", ""]
    by_name = {result["name"]: result for result in results}
    for item in selected:
        lines.extend(["=" * 80, f"Prompt ID: {item['id']}", f"Category: {item['category']}", f"Prompt: {item['prompt']}"])
        for name in CHECKPOINTS:
            mode = by_name[name]["modes"]["controlled_sampling"]
            generation = next(row for row in mode["generations"] if row["prompt_id"] == item["id"])
            lines.extend(["", f"[{name}]", generation["response"], f"Automatic metrics: {generation['metrics']}"])
        lines.extend(["", "Human preference: ", "Semantic correctness notes: ", "Conversation-quality notes: ", "Critical issue: "])
    return "\n".join(lines) + "\n"


def promotion_assessment(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_name = {result["name"]: result for result in results}
    base = by_name["ultrachat_base"]
    candidate = by_name["instruction_quality_candidate"]

    base_ultra = base["masked_validation_losses"][
        "ultrachat_validation"
    ]["loss"]
    candidate_ultra = candidate["masked_validation_losses"][
        "ultrachat_validation"
    ]["loss"]
    base_alpaca = base["masked_validation_losses"][
        "alpaca_validation"
    ]["loss"]
    candidate_alpaca = candidate["masked_validation_losses"][
        "alpaca_validation"
    ]["loss"]

    format_stable = all(
        candidate["modes"][mode]["summary"][
            "strict_format_compliance_rate"
        ]
        >= base["modes"][mode]["summary"][
            "strict_format_compliance_rate"
        ]
        for mode in MODES
    )
    repetition_stable = all(
        candidate["modes"][mode]["summary"]["repetition_ratio"]
        <= base["modes"][mode]["summary"]["repetition_ratio"] + 0.02
        for mode in MODES
    )
    empty_stable = all(
        candidate["modes"][mode]["summary"]["empty_rate"]
        <= base["modes"][mode]["summary"]["empty_rate"] + 0.01
        for mode in MODES
    )
    coherence_stable = all(
        candidate["modes"][mode]["summary"][
            "obvious_incoherence_rate"
        ]
        <= base["modes"][mode]["summary"][
            "obvious_incoherence_rate"
        ] + 0.01
        for mode in MODES
    )

    gates = {
        "instruction_format_stable_or_improved": format_stable,
        "repetition_not_materially_worse": repetition_stable,
        "empty_output_not_materially_worse": empty_stable,
        "coherence_not_materially_worse": coherence_stable,
        "ultrachat_loss_not_severely_regressed": (
            (candidate_ultra - base_ultra) / base_ultra <= 0.02
        ),
        "alpaca_loss_not_severely_regressed": (
            (candidate_alpaca - base_alpaca) / base_alpaca <= 0.02
        ),
        "manual_review_clear_improvement": False,
        "metadata_and_lineage_reproducible": True,
    }

    return {
        "gates": gates,
        "manual_review_status": (
            "pending; manual review form created"
        ),
        "ultrachat_loss_relative_change": (
            candidate_ultra - base_ultra
        ) / base_ultra,
        "alpaca_loss_relative_change": (
            candidate_alpaca - base_alpaca
        ) / base_alpaca,
        "promotion_recommended": all(gates.values()),
        "main_checkpoint_recommendation": "ultrachat_base",
        "reason": (
            "Automatic gates are provisional until semantic manual review "
            "shows a clear improvement without regression."
        ),
    }

def main() -> None:
    if sha256(TOKENIZER) != TOKENIZER_SHA:
        raise RuntimeError("tokenizer SHA-256 mismatch")
    if (
        sha256(Path("data/processed/instruct/ultrachat_masked_v2.bin"))
        != ULTRACHAT_TOKEN_SHA
    ):
        raise RuntimeError("UltraChat dataset SHA-256 mismatch")
    if (
        sha256(
            Path(
                "data/processed/instruct/"
                "ultrachat_masked_v2_mask.bin"
            )
        )
        != ULTRACHAT_MASK_SHA
    ):
        raise RuntimeError("UltraChat mask SHA-256 mismatch")

    for name, path in CHECKPOINTS.items():
        if sha256(path) != EXPECTED_HASHES[name]:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch: {name}"
            )

    benchmark = json.loads(
        BENCHMARK.read_text(encoding="utf-8")
    )
    prompts = benchmark["prompts"]
    if len(prompts) < 200:
        raise RuntimeError(
            "comparison benchmark requires at least 200 prompts"
        )

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = [
        evaluate_checkpoint(
            name,
            path,
            prompts,
            tokenizer,
            device,
        )
        for name, path in CHECKPOINTS.items()
    ]

    payload = {
        "format_version": (
            "instruction_quality_batch_001_comparison_v1"
        ),
        "benchmark": str(BENCHMARK).replace("\\", "/"),
        "benchmark_sha256": sha256(BENCHMARK),
        "tokenizer": str(TOKENIZER).replace("\\", "/"),
        "tokenizer_sha256": TOKENIZER_SHA,
        "ultrachat_dataset_sha256": ULTRACHAT_TOKEN_SHA,
        "ultrachat_mask_sha256": ULTRACHAT_MASK_SHA,
        "seed": SEED,
        "generation_modes": MODES,
        "generation_configuration_sha256": hashlib.sha256(
            json.dumps(MODES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "optimizer_updates": 0,
        "results": results,
        "promotion_assessment": promotion_assessment(results),
    }

    atomic_json(RESULT, payload)
    MANUAL_REVIEW.write_text(
        manual_review_text(results, prompts),
        encoding="utf-8",
    )

    print(f"Benchmark prompts: {len(prompts)}")
    print(f"Checkpoints compared: {len(CHECKPOINTS)}")
    print(f"Result: {RESULT}")
    print(f"Manual review: {MANUAL_REVIEW}")
    print("Optimizer updates: 0")


if __name__ == "__main__":
    main()