import numpy as np

from vasu.training.dataset import TextDataset


def test_text_dataset_returns_shifted_tokens(tmp_path):
    data_file = tmp_path / "tokens.bin"
    np.array(
        [1, 2, 3, 4, 5],
        dtype=np.uint16,
    ).tofile(data_file)

    dataset = TextDataset(
        data_file=str(data_file),
        seq_len=2,
    )

    x, y = dataset[0]

    assert x.tolist() == [1, 2]
    assert y.tolist() == [2, 3]
