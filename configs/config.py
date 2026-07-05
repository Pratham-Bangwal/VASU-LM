from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 256

    dim: int = 384
    hidden_dim: int = 1536

    n_layers: int = 8
    n_heads: int = 6

    dropout: float = 0.1

    rope_theta: int = 10000


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4

    epochs: int = 10

    weight_decay: float = 0.01

    mixed_precision: bool = True

    device: str = "cuda"

    seed: int = 42