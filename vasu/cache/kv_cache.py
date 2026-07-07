import torch


class KVCache:
    def __init__(self, n_layers: int):
        self.keys: list[torch.Tensor | None] = (
            [None] * n_layers
        )

        self.values: list[torch.Tensor | None] = (
            [None] * n_layers
        )

    def get(
        self,
        layer_idx: int,
    ):
        return (
            self.keys[layer_idx],
            self.values[layer_idx],
        )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        self.keys[layer_idx] = k
        self.values[layer_idx] = v

    def clear(self):
        for i in range(len(self.keys)):
            self.keys[i] = None
            self.values[i] = None