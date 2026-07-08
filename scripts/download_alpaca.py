from datasets import load_dataset
from pathlib import Path
import json

print("Downloading Alpaca dataset...")

dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train",
)

Path(
    "data/raw/instruct"
).mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    "data/raw/instruct/alpaca.jsonl",
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