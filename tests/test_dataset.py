from torch.utils.data import DataLoader

from vasu.data.dataset import VASUDataset


dataset = VASUDataset(
    data_dir="data/raw",
    tokenizer_path="checkpoints/tokenizer.json",
    context_length=16,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

x, y = next(iter(loader))

print(x.shape)
print(y.shape)

print()

print(x)

print()

print(y)
