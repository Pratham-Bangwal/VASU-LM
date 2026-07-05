from pathlib import Path

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset for next-token prediction.

    Input:
        [t0, t1, t2, t3]

    Target:
        [t1, t2, t3, t4]
    """

    def __init__(
        self,
        tokenizer,
        data_dir: str = "data/raw",
        seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        data_path = Path(data_dir)

        text = ""

        for file in sorted(data_path.glob("*.txt")):
            text += file.read_text(
                encoding="utf-8"
            ) + "\n"

        if len(text.strip()) == 0:
            raise ValueError(
                f"No text found in {data_dir}"
            )

        self.tokens = tokenizer.encode(text).ids

    def __len__(self):
        return max(
            0,
            len(self.tokens) - self.seq_len - 1,
        )

    def __getitem__(self, idx):

        x = self.tokens[
            idx : idx + self.seq_len
        ]

        y = self.tokens[
            idx + 1 : idx + self.seq_len + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )