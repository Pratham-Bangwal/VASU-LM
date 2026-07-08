from datasets import load_dataset
from pathlib import Path
import json

print("Downloading UltraChat...")

dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
)

Path(
    "data/raw/instruct"
).mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    "data/raw/instruct/ultrachat.jsonl",
    "w",
    encoding="utf-8",
) as f:

    for sample in dataset:
        f.write(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    f"Saved {len(dataset):,} examples."
)