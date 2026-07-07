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

    batch_size: int = 4

    epochs: int = 20

    learning_rate: float = 3e-4

    weight_decay: float = 0.01

    seed: int = 42

    gradient_accumulation_steps: int = 8

    grad_clip: float = 1.0

    use_amp: bool = True

    checkpoint_path: str = "checkpoints/vasu.pt"

    instruction_tuning: bool = False