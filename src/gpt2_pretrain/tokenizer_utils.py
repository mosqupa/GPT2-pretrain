from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_subword_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train([str(input_path)], trainer)
    tokenizer.decoder = WordPieceDecoder(prefix="##")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def train_bpe_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    return train_subword_tokenizer(input_path, output_path, vocab_size, min_frequency)


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
