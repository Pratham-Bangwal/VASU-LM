from dataclasses import dataclass


@dataclass
class VASUConfig:
    # ---------------------
    # Model
    # ---------------------
    vocab_size: int = 32000
    context_length: int = 512

    n_layers: int = 6
    n_heads: int = 8

    embedding_dim: int = 512
    hidden_dim: int = 2048

    dropout: float = 0.1

    # ---------------------
    # Training
    # ---------------------
    batch_size: int = 8
    learning_rate: float = 3e-4

    epochs: int = 10

    weight_decay: float = 0.01

    # ---------------------
    # Hardware
    # ---------------------
    device: str = "cuda"

    mixed_precision: bool = True

    # ---------------------
    # Checkpoints
    # ---------------------
    checkpoint_dir: str = "checkpoints"

    # ---------------------
    # Randomness
    # ---------------------
    seed: int = 42