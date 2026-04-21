from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from gpt2_pretrain.data import clean_corpus_file
from gpt2_pretrain.model import GPT, GPTConfig
from gpt2_pretrain.tokenizer_utils import load_tokenizer, train_bpe_tokenizer


def test_causal_mask_and_forward_shape() -> None:
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 128)
    assert loss is not None
    assert torch.isfinite(loss)


def test_tokenizer_train_and_load() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_path = tmp / "raw.txt"
        clean_path = tmp / "clean.txt"
        tokenizer_path = tmp / "tokenizer.json"
        raw_path.write_text("人工智能正在改变世界。\n人工智能正在改变世界。\n课程作业要完整。\n", encoding="utf-8")
        kept = clean_corpus_file(raw_path, clean_path)
        assert kept == 2
        train_bpe_tokenizer(clean_path, tokenizer_path, vocab_size=64, min_frequency=1)
        tokenizer = load_tokenizer(tokenizer_path)
        encoded = tokenizer.encode("人工智能")
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        assert encoded.ids
        assert isinstance(decoded, str)
