"""Import/export in the app's exact embedding JSON format.

Shape (SemanticSearchEngine.importJsonDatabase / exportJsonDatabase):

    [{"filename": str, "title": str, "artist": str,
      "duration": int, "chunks": [[512 floats], ...]}]

Chunk offsets are not stored; the app rebuilds them as ``index * 30``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..core.embedder import TrackEmbedding


def _as_text(value) -> str:
    return "" if value is None else str(value)


def read(path: str | Path) -> Iterator[TrackEmbedding]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in data:
        chunks = [np.asarray(chunk, dtype=np.float32)
                  for chunk in (item.get("chunks") or [])]
        yield TrackEmbedding(
            file_path=_as_text(item.get("filename")),
            title=_as_text(item.get("title")),
            artist=_as_text(item.get("artist")),
            duration_seconds=int(item.get("duration") or 0),
            chunks=chunks,
        )


def write(path: str | Path, tracks: Iterable[TrackEmbedding]) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("[")
        for track in tracks:
            if count:
                handle.write(",")
            json.dump({
                "filename": track.file_path,
                "title": track.title,
                "artist": track.artist,
                "duration": int(track.duration_seconds),
                "chunks": [[float(v) for v in chunk] for chunk in track.chunks],
            }, handle, ensure_ascii=False)
            count += 1
        handle.write("]")
    return count
