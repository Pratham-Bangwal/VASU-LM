import torch

from vasu.config import (
    ModelConfig,
    TrainConfig,
)
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import TextDataset
from vasu.training.trainer import Trainer


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    print("Loading tokenizer...")

    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")

    model_config = ModelConfig()

    train_config = TrainConfig(
        epochs=5,
        learning_rate=5e-6,
        checkpoint_path="checkpoints/instruct/vasu.pt",
        checkpoint_dir="checkpoints/instruct",
    )

    print("Loading instruction dataset...")

    dataset = TextDataset(
        data_file="data/processed/instruct/alpaca.bin",
        seq_len=model_config.max_seq_len,
    )

    total_tokens = len(dataset.tokens)

    print(f"Total tokens: {total_tokens:,}")

    split = int(total_tokens * 0.95)

    train_dataset = TextDataset(
        data_file="data/processed/instruct/alpaca.bin",
        seq_len=model_config.max_seq_len,
        start=0,
        end=split,
    )

    val_dataset = TextDataset(
        data_file="data/processed/instruct/alpaca.bin",
        seq_len=model_config.max_seq_len,
        start=split,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    print("Building model...")

    model = VASUModel(model_config).to(device)

    print("Loading pretrained weights...")

    checkpoint = torch.load(
        "checkpoints/best.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        device=device,
    )

    print("Starting instruction tuning...\n")

    trainer.fit()


if __name__ == "__main__":
    main()