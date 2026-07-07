from vasu.tokenizer import VASUTokenizer


def main():

    tokenizer = VASUTokenizer()

    tokenizer.train("data/raw")

    tokenizer.save("assets/tokenizer.json")

    text = "Hello! I am building VASU."

    ids = tokenizer.encode(text)

    print("Original:")
    print(text)

    print()

    print("Token IDs:")
    print(ids)

    print()

    print("Decoded:")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()