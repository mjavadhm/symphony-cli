"""Golden parity tests against embeddings exported from the Android app.

Drop into tests/golden/:
  * the audio files themselves (same ones indexed in the app)
  * reference.json - the app's "Export AI Index" output for those files

The test then re-embeds each file locally and compares vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"
REFERENCE = GOLDEN / "reference.json"

COSINE_MIN = 0.999
MAX_ABS_DIFF = 1e-3


def _reference_tracks():
    if not REFERENCE.exists():
        return []
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.mark.skipif(not REFERENCE.exists(), reason="no golden reference.json yet")
@pytest.mark.parametrize("track", _reference_tracks(),
                         ids=lambda t: t.get("title", "?"))
def test_embedding_matches_app(track):
    from symphony_cli.core.embedder import Embedder

    audio = GOLDEN / Path(track["filename"]).name
    if not audio.exists():
        pytest.skip(f"audio file missing: {audio.name}")

    local = Embedder().embed_file(audio)
    expected = [np.asarray(c, dtype=np.float32) for c in track["chunks"]]

    assert len(local.chunks) == len(expected), "chunk count differs"
    for i, (got, want) in enumerate(zip(local.chunks, expected)):
        cosine = float(np.dot(got, want))
        assert cosine > COSINE_MIN, f"chunk {i}: cosine {cosine:.6f}"
        assert float(np.max(np.abs(got - want))) < MAX_ABS_DIFF, f"chunk {i}: abs diff too large"
