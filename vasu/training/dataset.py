import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        data_file="data/processed/tinystories.bin",
        seq_len=256,
        start=0,
        end=None,
        stride=None
    ):

        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.tokens = np.memmap(
            data_file,
            dtype=np.uint16,
            mode="r",
        )

        if end is None:
            end = len(self.tokens)

        self.start = start
        self.end = end

        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequence length: {seq_len}")

    def __len__(self):

        return (
            self.end
            - self.start
            - self.seq_len
        ) // self.stride

    def __getitem__(self, idx):
        idx = self.start + idx * self.stride
        
        chunk = self.tokens[
            idx : idx + self.seq_len + 1
        ]

        x = torch.from_numpy(
            chunk[:-1].astype(np.int64)
        )

        y = torch.from_numpy(
            chunk[1:].astype(np.int64)
        )

        return x, y