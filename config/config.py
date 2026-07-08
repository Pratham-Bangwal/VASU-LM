from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Tokenizer
    vocab_size: int = 32000

    # Sequence
    max_seq_len: int = 256

    # Transformer
    dim: int = 384
    hidden_dim: int = 1536

    n_layers: int = 8
    n_heads: int = 6

    dropout: float = 0.1

    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 8

    learning_rate: float = 3e-4
    
    weight_decay: float = 0.01

    epochs: int = 10

    device: str = "cuda"

    mixed_precision: bool = True

    seed: int = 42