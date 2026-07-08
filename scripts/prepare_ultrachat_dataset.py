import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from vasu.tokenizer.tokenizer import VASUTokenizer


print("Loading tokenizer...")

tokenizer = VASUTokenizer()
tokenizer.load("assets/tokenizer.json")

input_file = "data/raw/instruct/ultrachat.jsonl"
output_file = "data/processed/instruct/ultrachat.bin"

Path(
    "data/processed/instruct"
).mkdir(
    parents=True,
    exist_ok=True,
)

total_tokens = 0

with open(output_file, "wb") as out:

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as f:

        for i, line in enumerate(tqdm(f)):

            sample = json.loads(line)

            messages = sample["messages"]

            text = ""

            for msg in messages:
                role = msg["role"].capitalize()

                text += (
                    f"{role}: "
                    f"{msg['content']}\n"
                )

            text += "[EOS]\n"

            ids = tokenizer.encode(text)

            arr = np.array(
                ids,
                dtype=np.uint16,
            )

            arr.tofile(out)

            total_tokens += len(ids)

            if (i + 1) % 10000 == 0:
                print(
                    f"Processed {i+1:,} examples..."
                )

print()
print(f"Total tokens: {total_tokens:,}")
print(f"Saved to {output_file}")
print(
    f"Dataset size: "
    f"{Path(output_file).stat().st_size/1024**2:.2f} MB"
)