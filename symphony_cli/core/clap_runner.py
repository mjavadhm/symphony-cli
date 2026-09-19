"""ONNX CLAP encoders, ported from ClapModelRunner.kt."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.sqrt(np.sum(vector * vector)))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)


class ClapModelRunner:
    def __init__(self, audio_model: Path | None = None, text_model: Path | None = None):
        import onnxruntime as ort

        audio_model = Path(audio_model or config.AUDIO_MODEL_PATH)
        text_model = Path(text_model or config.TEXT_MODEL_PATH)
        missing = [p for p in (audio_model, text_model) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "ONNX models not found: " + ", ".join(str(p) for p in missing))
        options = ort.SessionOptions()
        self.audio_session = ort.InferenceSession(str(audio_model), options,
                                                  providers=["CPUExecutionProvider"])
        self.text_session = ort.InferenceSession(str(text_model), options,
                                                 providers=["CPUExecutionProvider"])

    def audio_embedding(self, mel_spectrogram: np.ndarray,
                        time_frames: int = config.MAX_FRAMES) -> np.ndarray:
        tensor = np.asarray(mel_spectrogram, dtype=np.float32).reshape(
            1, 1, time_frames, config.N_MELS)
        outputs = self.audio_session.run(None, {"input_features": tensor})
        return l2_normalize(np.asarray(outputs[0])[0])

    def text_embedding(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64).reshape(1, -1),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64).reshape(1, -1),
        }
        outputs = self.text_session.run(None, feeds)
        return l2_normalize(np.asarray(outputs[0])[0])
