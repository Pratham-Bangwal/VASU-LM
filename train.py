import torch

from vasu.config import ModelConfig, TrainConfig
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import TextDataset
from vasu.training.trainer import Trainer
from vasu.training.callbacks.tensorboard import TensorBoardCallback
from vasu.training.callbacks.sample_generation import SampleGenerationCallback

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    print("Loading tokenizer...")

    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")

    model_config = ModelConfig()
    train_config = TrainConfig()

    print("Loading dataset...")

    dataset = TextDataset(
        data_file="data/processed/tinystories.bin",
        seq_len=model_config.max_seq_len,
    )

    total_tokens = len(dataset.tokens)

    split = 5_000_000

    train_dataset = TextDataset(
        data_file="data/processed/tinystories.bin",
        seq_len=model_config.max_seq_len,
        start=0,
        end=split,
    )

    val_dataset = TextDataset(
        data_file="data/processed/tinystories.bin",
        seq_len=model_config.max_seq_len,
        start=split,
        end=split+500_000,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    print("Building model...")

    model = VASUModel(model_config).to(device)
    print(
        f"VRAM after model: "
        f"{torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )

    dummy = torch.randint(
        0,
        model_config.vocab_size,
        (2, model_config.max_seq_len),
    ).to(device)

    with torch.no_grad():
        logits = model(dummy)

    print("Logits shape:", logits.shape)
    print("Logits min :", logits.min().item())
    print("Logits max :", logits.max().item())
    print("Logits mean:", logits.mean().item())
    print("Logits std :", logits.std().item())

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {params:,}")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        config=train_config,
        device=device,
        callbacks=[
            SampleGenerationCallback(),
            TensorBoardCallback()
        ],
    )

    print("Starting training...\n")

    trainer.fit()
    


if __name__ == "__main__":
    main()