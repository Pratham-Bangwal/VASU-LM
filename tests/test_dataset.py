from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import TextDataset


def main():

    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")

    dataset = TextDataset(
        tokenizer=tokenizer,
        seq_len=32,
    )

    print("Dataset size:", len(dataset))

    x, y = dataset[0]

    print(x.shape)
    print(y.shape)

    print(x[:10])
    print(y[:10])


if __name__ == "__main__":
    main()