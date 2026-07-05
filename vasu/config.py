from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 256

    dim: int = 384
    n_heads: int = 6
    n_layers: int = 8
    hidden_dim: int = 1536

    dropout: float = 0.1
    rope_theta: float = 10000.0

    bias: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10

    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    device: str = "cuda"
    seed: int = 42