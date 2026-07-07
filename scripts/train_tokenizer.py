from pathlib import Path

from vasu.tokenizer.tokenizer import VASUTokenizer

tokenizer = VASUTokenizer()

print("Training tokenizer...")

tokenizer.train(
    data_dir="data/raw/tinystories",
    vocab_size=32000,
)

Path("assets").mkdir(exist_ok=True)

tokenizer.save(
    "assets/tokenizer.json"
)

print("Done.")