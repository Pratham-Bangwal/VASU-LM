from vasu.data.loader import VASUDataLoader


def test_data_loader_combines_text_files(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")

    loader = VASUDataLoader(str(tmp_path))

    assert loader.load() == "first\nsecond\n"
