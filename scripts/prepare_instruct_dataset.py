import json
from pathlib import Path

import numpy as np

from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.data.prompt_templates import format_alpaca_example


print("Loading tokenizer...")

tokenizer = VASUTokenizer()
tokenizer.load("assets/tokenizer.json")

tokens = []

print("Processing Alpaca dataset...")

with open(
    "data/raw/instruct/alpaca.jsonl",
    "r",
    encoding="utf-8",
) as f:

    for i, line in enumerate(f):

        sample = json.loads(line)

        instruction = sample["instruction"]
        inp = sample["input"]
        output = sample["output"]

        text = format_alpaca_example(
            instruction=instruction,
            input_text=inp,
            response=output,
        )

        tokens.extend(
            tokenizer.encode(text)
        )

        if (i + 1) % 5000 == 0:
            print(
                f"Processed {i+1:,} examples..."
            )

print(
    f"\nTotal tokens: {len(tokens):,}"
)

arr = np.array(
    tokens,
    dtype=np.uint16,
)

Path(
    "data/processed/instruct"
).mkdir(
    parents=True,
    exist_ok=True,
)

output_file = (
    "data/processed/instruct/alpaca.bin"
)

arr.tofile(output_file)

print(
    f"Saved to {output_file}"
)
print(
    f"Dataset size: "
    f"{arr.nbytes / 1024**2:.2f} MB"
)
