from pathlib import Path

import torch
from torch.utils.data import Dataset

from vasu.tokenizer import VASUTokenizer


class VASUDataset(Dataset):
    """
    Creates training samples for a decoder-only language model.

    Each sample consists of:

    Input:
    x = tokens[:-1]

    Target:
    y = tokens[1:]
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        context_length: int = 128,
    ):

        self.context_length = context_length

        tokenizer = VASUTokenizer()
        tokenizer.load(tokenizer_path)

        text = ""

        for file in Path(data_dir).glob("*.txt"):
            text += file.read_text(encoding="utf-8")
            text += "\n"

        self.tokens = tokenizer.encode(text)

    def __len__(self):

        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):

        chunk = self.tokens[idx:idx + self.context_length + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y