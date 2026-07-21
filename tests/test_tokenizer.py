from vasu.tokenizer import VASUTokenizer


def test_tokenizer_train_encode_decode_roundtrip(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "sample.txt").write_text(
        "Hello VASU. Hello tokenizer.",
        encoding="utf-8",
    )

    tokenizer = VASUTokenizer()
    tokenizer.train(str(data_dir), vocab_size=64)

    text = "Hello VASU."
    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids).strip() == text
