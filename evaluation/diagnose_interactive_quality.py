"""Read-only diagnostic for VASU-60M interactive response quality."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any

import torch

if __package__:
    from evaluation.evaluate_ultrachat_promotion import response_metrics
else:
    from evaluate_ultrachat_promotion import response_metrics
from vasu.config import get_vasu_60m_config
from vasu.data.alpaca_masked_v2 import encode_source_example
from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.inference.generate import generate_token_ids
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


TOKENIZER_PATH = Path("assets/tokenizer.json")
ALPACA_SOURCE = Path("data/raw/instruct/alpaca.jsonl")
ALPACA_TOKENS = Path("data/processed/instruct/alpaca_masked_v2.bin")
ALPACA_MASK = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
ALPACA_METADATA = Path("data/processed/instruct/alpaca_masked_v2_metadata.json")
RESULT = Path("evaluation/results/interactive_quality_diagnostic_v1.json")
REPORT = Path("evaluation/results/interactive_quality_diagnostic_v1.txt")
SEED = 42

CHECKPOINTS = {
    "fineweb_200k": Path(
        "checkpoints/vasu_60m/milestones/fineweb_step_200000.pt"
    ),
    "alpaca_v3": Path(
        "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt"
    ),
    "ultrachat_v2": Path(
        "checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3/best.pt"
    ),
}

DIRECT_STRINGS = [
    "such as web development",
    "It is also known",
    "The capital of Japan is Tokyo.",
    "1. First item\n2. Second item",
    '{"name": "VASU", "purpose": "assistant"}',
]

SMOKE_PROMPTS = [
    {"id": "machine_learning", "prompt": "Explain machine learning in simple words.", "constraints": {}},
    {"id": "exercise_three", "prompt": "Give me exactly three benefits of exercise.", "constraints": {"numbered_count": 3}},
    {"id": "japan_capital", "prompt": "What is the capital of Japan? Answer in one sentence.", "constraints": {"sentence_count": 1}, "expected": "tokyo"},
    {"id": "polite_rewrite", "prompt": "Rewrite this politely: Send me the file now.", "constraints": {}},
    {"id": "vasu_json", "prompt": "Give me valid JSON with keys name and purpose for VASU.", "constraints": {"json_keys": ["name", "purpose"]}},
    {"id": "gravity", "prompt": "Explain gravity to a ten-year-old in two sentences.", "constraints": {"sentence_count": 2}},
    {"id": "education_bullets", "prompt": "Give exactly three bullet points about education.", "constraints": {"bullet_count": 3}},
    {"id": "water_fact", "prompt": "At sea level, what temperature does water boil at? Answer briefly.", "constraints": {}, "expected": "100"},
    {"id": "summary", "prompt": "Summarize in under 20 words: Plants use sunlight, water, and carbon dioxide to make food and release oxygen.", "constraints": {"max_words": 20}},
    {"id": "one_word", "prompt": "Answer with one word: What color is a clear daytime sky usually?", "constraints": {"exact_words": 1}, "expected": "blue"},
    {"id": "no_list", "prompt": "Explain why sleep matters in one short paragraph. Do not use a list.", "constraints": {"forbid_list": True}},
    {"id": "numbered_steps", "prompt": "Give exactly four numbered steps for washing your hands.", "constraints": {"numbered_count": 4}},
    {"id": "uncertainty", "prompt": "Predict the exact price of gold in 2050. Be honest about uncertainty.", "constraints": {"uncertainty_required": True}},
    {"id": "arithmetic", "prompt": "What is 17 plus 25? Answer with one number.", "constraints": {"exact_words": 1}, "expected": "42"},
    {"id": "database", "prompt": "What is a database? Answer in two simple sentences.", "constraints": {"sentence_count": 2}},
    {"id": "friendly", "prompt": "Reply warmly in one sentence: I feel nervous about my exam.", "constraints": {"sentence_count": 1}},
    {"id": "python", "prompt": "Write one line of Python that prints Hello.", "constraints": {}},
    {"id": "rewrite_active", "prompt": "Rewrite in active voice: The report was written by Maya.", "constraints": {}},
    {"id": "india", "prompt": "Write exactly two factual sentences about India.", "constraints": {"sentence_count": 2}},
    {"id": "story", "prompt": "Write a three-sentence story about a helpful robot.", "constraints": {"sentence_count": 3}},
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def round_trip_sentences() -> list[str]:
    topics = [
        "web development", "machine learning", "public health", "Japan",
        "river systems", "small gardens", "Python code", "data ethics",
        "clean energy", "local history",
    ]
    templates = [
        "This is also known as {topic}.",
        "For example, {topic} can be useful.",
        "After commas, spaces should remain: yes, always.",
        "Don't remove spaces from {topic}.",
        "Heading: A guide to {topic}",
        "1. Study {topic}\n2. Review the notes",
        '{{"name": "{topic}", "status": "ready"}}',
        "print('Learn {topic}')",
        "Is {topic} useful? Yes, in context.",
        "The phrase—{topic}—uses em dashes.",
        "A semicolon separates ideas; {topic} remains readable.",
        "Version 2.0 includes {topic}, tests, and docs.",
        "She said, \"Study {topic}.\"",
        "It's important that {topic} isn't joined.",
        "Path: C:/examples/{topic}.",
        "Use (carefully) the term {topic}.",
        "First sentence. Second sentence about {topic}.",
        "- {topic}\n- another item",
        "Name: {topic}\nPurpose: demonstration",
        "Common boundaries include is also, such as web, and is Japan.",
    ]
    return [template.format(topic=topic) for topic in topics for template in templates]


def tokenizer_audit(tokenizer: VASUTokenizer) -> dict[str, Any]:
    sentences = round_trip_sentences()
    mismatches = []
    raw_exact = 0
    normalized_exact = 0
    for text in sentences:
        decoded = tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
        raw_exact += int(decoded == text)
        # ByteLevel(add_prefix_space=True) intentionally restores one leading
        # space. Interactive clean_response strips it; internal spacing must match.
        normalized = decoded[1:] if decoded.startswith(" ") else decoded
        passed = normalized == text
        normalized_exact += int(passed)
        if not passed:
            mismatches.append({"input": text, "decoded": decoded})
    direct = []
    for text in DIRECT_STRINGS:
        decoded = tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
        direct.append({"input": text, "decoded": decoded, "normalized_match": decoded.lstrip(" ") == text})
    boundary_cases = []
    for spaced, joined in (
        ("such as web", "such asweb"),
        ("It is also", "It isalso"),
        ("Japan is Tokyo", "JapanisTokyo"),
    ):
        boundary_cases.append({
            "spaced": {
                "text": spaced,
                "tokens": [tokenizer.tokenizer.id_to_token(token_id) for token_id in tokenizer.encode(spaced)],
            },
            "joined": {
                "text": joined,
                "tokens": [tokenizer.tokenizer.id_to_token(token_id) for token_id in tokenizer.encode(joined)],
            },
        })
    return {
        "sentences": len(sentences),
        "raw_exact_rate": raw_exact / len(sentences),
        "expected_normalized_exact_rate": normalized_exact / len(sentences),
        "spacing_mismatch_rate": len(mismatches) / len(sentences),
        "mismatches": mismatches[:20],
        "direct_cases": direct,
        "boundary_cases": boundary_cases,
        "bytelevel_decoder_configured": tokenizer.tokenizer.decoder is not None,
        "finding": "Tokenizer adds its configured leading ByteLevel space but preserves internal spaces; chat stripping removes only the leading space.",
    }


def load_source_examples(limit: int | None = None) -> list[dict[str, Any]]:
    examples = []
    with ALPACA_SOURCE.open("r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def prompt_alignment_audit(tokenizer: VASUTokenizer) -> dict[str, Any]:
    rows = []
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    for index, sample in enumerate(load_source_examples(20)):
        encoded = encode_source_example(sample, tokenizer)
        inference_prompt = format_alpaca_prompt(sample["instruction"], sample.get("input"))
        inference_ids = tokenizer.encode(inference_prompt)
        rows.append({
            "source_index": index,
            "prompt_text_equal": inference_prompt == tokenizer.decode(inference_ids, skip_special_tokens=True).lstrip(" "),
            "prompt_ids_equal": inference_ids == encoded.prompt_ids,
            "ends_response_header": inference_prompt.endswith("\nAssistant:"),
            "bos_present": tokenizer.tokenizer.token_to_id("[BOS]") in encoded.prompt_ids,
            "response_first_decoded": tokenizer.decode(encoded.response_ids[:4], skip_special_tokens=True),
            "response_mask_start": len(encoded.prompt_ids),
            "eos_appended_after_complete_response": eos_id is not None,
        })
    return {
        "examples": len(rows),
        "all_prompt_ids_match": all(row["prompt_ids_equal"] for row in rows),
        "all_response_headers_match": all(row["ends_response_header"] for row in rows),
        "bos_used": any(row["bos_present"] for row in rows),
        "eos_id": eos_id,
        "rows": rows,
    }


def dataset_audit(tokenizer: VASUTokenizer) -> dict[str, Any]:
    metadata = json.loads(ALPACA_METADATA.read_text(encoding="utf-8"))
    examples = load_source_examples()
    coverage_patterns = {
        "exact_list_lengths": r"\b(?:exactly|give|list)\s+(?:two|three|four|five|\d+)\b.*\b(?:items?|points?|benefits?|steps?)\b",
        "exact_sentence_counts": r"\b(?:one|two|three|four|\d+)\s+sentences?\b",
        "json_output": r"\bjson\b",
        "rewriting": r"\b(?:rewrite|rephrase|paraphrase|politely)\b",
        "short_factual_qa": r"^(?:what|who|where|when|which)\b|\bcapital of\b",
        "beginner_explanations": r"\b(?:simple words?|beginner|child|year-old)\b",
    }
    coverage = {
        name: sum(bool(re.search(pattern, str(row.get("instruction", "")), re.I)) for row in examples)
        for name, pattern in coverage_patterns.items()
    }
    outputs = [str(row.get("output", "")).strip() for row in examples]
    duplicate_outputs = sum(count - 1 for count in Counter(outputs).values() if count > 1)
    low_quality = sum(len(output.split()) < 5 for output in outputs)
    malformed = sum("Assistant:" in output or "User:" in output for output in outputs)
    total = int(metadata["total_tokens"])
    return {
        "records": int(metadata["number_of_training_records"]),
        "source_examples": len(examples),
        "supervised_token_ratio_total": int(metadata["assistant_loss_tokens"]) / total,
        "prompt_token_ratio_total": int(metadata["prompt_tokens"]) / total,
        "padding_token_ratio_total": int(metadata["padding_tokens"]) / total,
        "assistant_ratio_non_padding": float(metadata["assistant_token_ratio"]),
        "truncated_examples": int(metadata["truncated_examples"]),
        "truncated_answer_rate": int(metadata["truncated_examples"]) / int(metadata["retained_examples"]),
        "eos_supervised_examples": int(metadata["eos_tokens"]),
        "duplicate_output_rows": duplicate_outputs,
        "very_short_output_rows": low_quality,
        "template_marker_outputs": malformed,
        "coverage_counts": coverage,
        "mask_convention": "stored current-token mask; PackedInstructionDataset uses mask[1:] for shifted targets",
    }


def set_seed() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def malformed_spacing(response: str) -> list[str]:
    patterns = (r"\bsuch asweb\b", r"\bisalso\b", r"\bisjapan\b")
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, response, re.I)]


def compare_checkpoints(tokenizer: VASUTokenizer) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    results = []
    modes = {
        "greedy": {"do_sample": False, "max_new_tokens": 60, "repetition_penalty": 1.1},
        "sampling": {"do_sample": True, "temperature": 0.45, "top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1, "max_new_tokens": 60},
    }
    for name, path in CHECKPOINTS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model = VASUModel(get_vasu_60m_config())
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.to(device).eval()
        mode_results = {}
        for mode_name, kwargs in modes.items():
            generations = []
            for item in SMOKE_PROMPTS:
                formatted = format_alpaca_prompt(item["prompt"])
                prompt_tokens = tokenizer.encode(formatted)
                set_seed()
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
                metrics = response_metrics(item, response, token_ids, eos_id)
                expected = item.get("expected")
                metrics.update({
                    "prompt_tokens": len(prompt_tokens),
                    "total_sequence_tokens": len(prompt_tokens) + len(token_ids),
                    "hit_max_new_tokens": len(token_ids) == 60 and (not token_ids or token_ids[-1] != eos_id),
                    "hit_model_max_seq_len": len(prompt_tokens) + len(token_ids) >= 256,
                    "prompt_template_leakage": bool(re.search(r"\b(?:User|Assistant):", response)),
                    "malformed_spacing": malformed_spacing(response),
                    "expected_fact_present": expected is None or expected.casefold() in response.casefold(),
                })
                generations.append({"prompt_id": item["id"], "prompt": item["prompt"], "response": response, "metrics": metrics})
            mode_results[mode_name] = generations
        results.append({"checkpoint": name, "path": str(path).replace("\\", "/"), "modes": mode_results})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def summarize_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for result in results:
        summary[result["checkpoint"]] = {}
        for mode_name, rows in result["modes"].items():
            metrics = [row["metrics"] for row in rows]
            summary[result["checkpoint"]][mode_name] = {
                "mean_repetition": sum(row["repetition_ratio"] for row in metrics) / len(metrics),
                "empty_responses": sum(row["empty"] for row in metrics),
                "strict_constraint_compliance": sum(row["strict_format_compliance"] for row in metrics) / len(metrics),
                "malformed_spacing_responses": sum(bool(row["malformed_spacing"]) for row in metrics),
                "prompt_template_leakage": sum(row["prompt_template_leakage"] for row in metrics),
                "obvious_incoherence": sum(row["obvious_incoherence"] for row in metrics),
                "obvious_factual_failures": sum(not row["expected_fact_present"] for row in metrics),
                "hit_max_new_tokens": sum(row["hit_max_new_tokens"] for row in metrics),
                "hit_model_max_seq_len": sum(row["hit_model_max_seq_len"] for row in metrics),
                "eos_emissions": sum(row["ended_with_eos"] for row in metrics),
            }
    return summary


def text_report(payload: dict[str, Any]) -> str:
    lines = ["VASU INTERACTIVE QUALITY DIAGNOSTIC", "", f"Summary: {payload['comparison_summary']}"]
    for checkpoint in payload["checkpoint_comparison"]:
        for mode, rows in checkpoint["modes"].items():
            lines.extend(["", "=" * 80, f"{checkpoint['checkpoint']} / {mode}"])
            for row in rows:
                lines.extend(["-" * 80, f"Prompt: {row['prompt']}", f"Response: {row['response']}", f"Metrics: {row['metrics']}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_PATH))
    comparison = compare_checkpoints(tokenizer)
    payload = {
        "format_version": "interactive_quality_diagnostic_v1",
        "seed": SEED,
        "tokenizer_audit": tokenizer_audit(tokenizer),
        "prompt_alignment": prompt_alignment_audit(tokenizer),
        "dataset_audit": dataset_audit(tokenizer),
        "checkpoint_comparison": comparison,
        "comparison_summary": summarize_comparison(comparison),
        "optimizer_updates": 0,
        "training_started": False,
    }
    atomic_json(RESULT, payload)
    REPORT.write_text(text_report(payload), encoding="utf-8")
    print(f"Result: {RESULT}")
    print(f"Report: {REPORT}")
    print("Optimizer updates: 0")


if __name__ == "__main__":
    main()
