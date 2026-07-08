from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder



class VASUTokenizer:
    """
    Wrapper around Hugging Face Tokenizers.

    Responsible for:
    - Training
    - Encoding
    - Decoding
    - Saving
    - Loading
    """

    def __init__(self):

        
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, data_dir: str, vocab_size: int = 32000)->None:

        files = [str(f) for f in Path(data_dir).glob("*.txt")]

        if not files:
            raise FileNotFoundError(
            f"No .txt files found in {data_dir}"
        )

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
            ],
            initial_alphabet=ByteLevel.alphabet(),
        )

        self.tokenizer.train(files, trainer)

    def encode(
        self,
        text: str,
    ) -> list[int]:

        return self.tokenizer.encode(text).ids

    def decode(
        self,
        ids: list[int],
        **kwargs,
    ) -> str:

        return self.tokenizer.decode(
            ids,
            **kwargs
        )

    def save(
        self,
        path: str,
    ) -> None:

        file_path = Path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tokenizer.save(str(file_path))

    def load(
        self,
        path: str,
    ) -> None:

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)
        
        self.tokenizer = Tokenizer.from_file(str(file_path))