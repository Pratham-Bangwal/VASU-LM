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

    def train(self, data_dir: str, vocab_size: int = 32000):

        files = [str(f) for f in Path(data_dir).glob("*.txt")]

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
            ],
        )

        self.tokenizer.train(files, trainer)

    def encode(self, text: str):

        return self.tokenizer.encode(text).ids

    def decode(self, ids):

        return self.tokenizer.decode(ids)

    def save(self, path: str):

        self.tokenizer.save(path)

    def load(self, path: str):

        self.tokenizer = Tokenizer.from_file(path)