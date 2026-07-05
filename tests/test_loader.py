from vasu.data.loader import VASUDataLoader


def main():
    loader = VASUDataLoader("data/raw")

    text = loader.load()

    print(text)


if __name__ == "__main__":
    main()