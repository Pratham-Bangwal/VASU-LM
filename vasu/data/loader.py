from pathlib import Path


class VASUDataLoader:
    """
    Loads raw text files from a directory for training VASU.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' does not exist."
            )

    def load(self) -> str:
        """
        Load and combine all .txt files into a single string.
        """

        text = ""

        for file in sorted(self.data_dir.glob("*.txt")):
            text += file.read_text(encoding="utf-8")
            text += "\n"

        return text