from pathlib import Path
import numpy as np

from vasu.tokenizer.tokenizer import VASUTokenizer

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

tokenizer = VASUTokenizer()
tokenizer.load("assets/tokenizer.json")

input_file = Path("data/raw/tinystories/tinystories.txt")
output_file = Path("data/processed/tinystories.bin")

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "wb") as fout:

    total = 0

    while True:

        text = fin.read(CHUNK_SIZE)

        if not text:
            break

        ids = tokenizer.encode(text)

        arr = np.array(ids, dtype=np.uint16)

        arr.tofile(fout)

        total += len(arr)

        print(f"{total:,} tokens", end="\r")

print(f"\nFinished.")
print(f"Saved {total:,} tokens.")