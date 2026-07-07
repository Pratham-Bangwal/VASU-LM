from datasets import load_dataset
from pathlib import Path

output_dir = Path("data/raw/tinystories")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
)

output_file = output_dir / "tinystories.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for sample in dataset:
        f.write(sample["text"])
        f.write("\n")

print("Done!")
print(output_file)