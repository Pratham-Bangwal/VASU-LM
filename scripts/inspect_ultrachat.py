from datasets import load_dataset
import json

dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft",
)

sample = dataset[0]

print(sample)
print()
print(json.dumps(sample, indent=2))