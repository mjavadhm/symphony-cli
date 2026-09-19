"""RoBERTa BPE tokenizer, ported from RobertaTokenizer.kt.

Reads the same vocab.json / merges.txt that ship in the app's assets so the
token ids - and therefore the text embeddings - stay identical.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config

_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+")


@lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapping = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + n)
            n += 1
    return mapping


class RobertaTokenizer:
    def __init__(self, vocab_path: Path | None = None, merges_path: Path | None = None):
        vocab_path = Path(vocab_path or config.VOCAB_PATH)
        merges_path = Path(merges_path or config.MERGES_PATH)
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"tokenizer assets missing: {vocab_path}, {merges_path}")
        self.vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        merges: list[tuple[str, str]] = []
        for line in merges_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = _byte_encoder()
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        while len(word) > 1:
            best_rank = None
            best_index = -1
            for i in range(len(word) - 1):
                rank = self.ranks.get((word[i], word[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_index < 0:
                break
            word[best_index: best_index + 2] = [word[best_index] + word[best_index + 1]]
        self._cache[token] = word
        return word

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in _PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
            tokens.extend(self._bpe(encoded))
        return tokens

    def encode(self, raw_text: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (input_ids, attention_mask), both int64 of length 77."""
        text = raw_text if raw_text.startswith(" ") else " " + raw_text
        ids = [config.BOS_TOKEN_ID]
        for token in self.tokenize(text):
            token_id = self.vocab.get(token)
            if token_id is not None:
                ids.append(int(token_id))
        ids.append(config.EOS_TOKEN_ID)
        if len(ids) > config.TOKENIZER_MAX_LENGTH:
            ids = ids[: config.TOKENIZER_MAX_LENGTH - 1] + [config.EOS_TOKEN_ID]

        input_ids = np.full(config.TOKENIZER_MAX_LENGTH, config.PAD_TOKEN_ID, dtype=np.int64)
        attention_mask = np.zeros(config.TOKENIZER_MAX_LENGTH, dtype=np.int64)
        input_ids[: len(ids)] = ids
        attention_mask[: len(ids)] = 1
        return input_ids, attention_mask
