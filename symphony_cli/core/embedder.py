"""High-level embedding helpers (SemanticSearchEngine.kt)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from . import decoder, mel
from .clap_runner import ClapModelRunner, l2_normalize
from .tokenizer import RobertaTokenizer


@dataclass
class TrackEmbedding:
    file_path: str
    title: str
    artist: str
    duration_seconds: int
    chunks: list[np.ndarray] = field(default_factory=list)

    @property
    def mean_embedding(self) -> np.ndarray:
        """Mean of the individually normalized chunks, re-normalized."""
        if not self.chunks:
            return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        stacked = np.stack([l2_normalize(c) for c in self.chunks])
        return l2_normalize(stacked.mean(axis=0))


class Embedder:
    def __init__(self, runner: ClapModelRunner | None = None,
                 tokenizer: RobertaTokenizer | None = None):
        self._runner = runner
        self._tokenizer = tokenizer

    @property
    def runner(self) -> ClapModelRunner:
        if self._runner is None:
            self._runner = ClapModelRunner()
        return self._runner

    @property
    def tokenizer(self) -> RobertaTokenizer:
        if self._tokenizer is None:
            self._tokenizer = RobertaTokenizer()
        return self._tokenizer

    def embed_file(self, path: str | Path, title: str = "", artist: str = "") -> TrackEmbedding:
        path = Path(path)
        chunks = [self.runner.audio_embedding(mel.extract(chunk.samples))
                  for chunk in decoder.stream_chunks(path)]
        if not chunks:
            raise decoder.DecodeError(f"could not extract audio chunks from {path}")
        return TrackEmbedding(
            file_path=str(path),
            title=title or path.stem,
            artist=artist,
            duration_seconds=decoder.duration_seconds(path),
            chunks=chunks,
        )

    def embed_text(self, text: str) -> np.ndarray:
        input_ids, attention_mask = self.tokenizer.encode(text)
        return self.runner.text_embedding(input_ids, attention_mask)

    def embed_query(self, query: str) -> np.ndarray:
        """Prompt ensemble: average a few paraphrases, then re-normalize."""
        q = query.strip()
        total = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        for template in config.QUERY_TEMPLATES:
            embedding = self.embed_text(template.format(q=q))
            length = min(config.EMBEDDING_DIM, embedding.size)
            total[:length] += embedding[:length]
        return l2_normalize(total)
