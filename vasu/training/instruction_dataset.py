import numpy as np
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        mask_file: str,
        seq_len: int,
        start: int = 0,
        end: int | None = None,
    ):
        tokens = np.memmap(
            data_file,
            dtype=np.uint16,
            mode="r",
        )

        mask = np.memmap(
            mask_file,
            dtype=np.uint8,
            mode="r",
        )

        if end is None:
            end = len(tokens)

        self.tokens = tokens[start:end]
        self.mask = mask[start:end]

        self.seq_len = seq_len

    def __len__(self):
        return (
            len(self.tokens) - 1
        ) // self.seq_len

    def __getitem__(self, idx):

        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]
        mask = self.mask[start:end]

        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long,
        )

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long,
        )

        loss_mask = torch.tensor(
            mask[1:],
            dtype=torch.float32,
        )

        return x, y, loss_mask


class PackedInstructionDataset(Dataset):
    """Record-aware masked dataset that never slices across packed records."""

    def __init__(
        self,
        data_file: str,
        mask_file: str,
        seq_len: int,
        start_record: int = 0,
        end_record: int | None = None,
    ):
        self.seq_len = seq_len
        self.record_length = seq_len + 1
        self.tokens = np.memmap(data_file, dtype=np.uint16, mode="r")
        self.mask = np.memmap(mask_file, dtype=np.uint8, mode="r")

        if len(self.tokens) != len(self.mask):
            raise ValueError("token and mask files have different lengths")
        if len(self.tokens) % self.record_length != 0:
            raise ValueError("packed files are not divisible into records")

        self.number_of_records = len(self.tokens) // self.record_length
        if end_record is None:
            end_record = self.number_of_records
        if not 0 <= start_record <= end_record <= self.number_of_records:
            raise ValueError("invalid packed-record range")
        self.start_record = start_record
        self.end_record = end_record

    def __len__(self):
        return self.end_record - self.start_record

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        record_index = self.start_record + idx
        start = record_index * self.record_length
        end = start + self.record_length
        tokens = self.tokens[start:end]
        mask = self.mask[start:end]

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.float32)
        return x, y, loss_mask
