"""Audio decoding, ported from AudioDecoder.kt.

The app decodes with MediaCodec at the file's native sample rate, averages the
channels into mono and then upsamples to 48 kHz with plain linear interpolation.
We reproduce the same steps with ffmpeg so the samples stay comparable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config


class DecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray   # float32, mono, 48 kHz
    offset_seconds: int


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise DecodeError("ffmpeg and ffprobe must be installed and on PATH")


def probe(path: str | Path) -> dict:
    """Returns native sample rate, channel count and duration in seconds."""
    _require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise DecodeError(f"ffprobe failed for {path}: {out.stderr.strip()[:200]}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise DecodeError(f"no audio stream in {path}")
    stream = streams[0]
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return {
        "sample_rate": int(stream.get("sample_rate") or config.SAMPLE_RATE),
        "channels": int(stream.get("channels") or 1),
        # the app uses integer-truncated seconds
        "duration_seconds": int(duration),
    }


def duration_seconds(path: str | Path) -> int:
    return probe(path)["duration_seconds"]


def _decode_range_native(path: str | Path, offset: int, length: int,
                         sample_rate: int, channels: int) -> np.ndarray:
    """Decodes [offset, offset+length) as interleaved s16 at the native rate."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-ss", str(offset), "-t", str(length), "-i", str(path),
        "-map", "a:0", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", str(channels), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise DecodeError(f"ffmpeg failed for {path}: {out.stderr.decode()[:200]}")
    raw = np.frombuffer(out.stdout, dtype="<i2")
    if raw.size == 0:
        return np.empty(0, dtype=np.float32)
    usable = (raw.size // channels) * channels
    frames = raw[:usable].reshape(-1, channels).astype(np.float32) / 32768.0
    # mono downmix: plain average of the channels, exactly like decodeChunk()
    return frames.mean(axis=1).astype(np.float32)


def resample_to_48k(samples: np.ndarray, original_rate: int) -> np.ndarray:
    """Linear interpolation, identical to resampleTo48k() in the app."""
    if original_rate == config.SAMPLE_RATE or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    ratio = config.SAMPLE_RATE / original_rate
    out_len = int(samples.size * ratio)
    if out_len <= 0:
        return np.empty(0, dtype=np.float32)
    positions = np.arange(out_len, dtype=np.float64) / ratio
    index = np.floor(positions).astype(np.int64)
    fraction = (positions - index).astype(np.float32)
    last = samples.size - 1
    safe = np.clip(index, 0, last - 1) if last >= 1 else np.zeros_like(index)
    interpolated = samples[safe] * (1.0 - fraction) + samples[safe + 1] * fraction
    # the app clamps the tail to the final sample
    interpolated = np.where(index >= last, samples[last], interpolated)
    return interpolated.astype(np.float32)


def chunk_offsets(total_seconds: int) -> list[int]:
    """Same offset selection as streamChunks()."""
    offsets = list(range(0, max(total_seconds, 0), config.CHUNK_STEP_SECONDS))
    offsets = offsets[: config.MAX_CHUNKS]
    if len(offsets) > 1:
        offsets = [o for o in offsets
                   if total_seconds - o >= config.MIN_TAIL_SECONDS]
    return offsets


def stream_chunks(path: str | Path):
    """Yields AudioChunk objects, mono float32 at 48 kHz."""
    info = probe(path)
    for offset in chunk_offsets(info["duration_seconds"]):
        native = _decode_range_native(
            path, offset, config.CHUNK_DECODE_SECONDS,
            info["sample_rate"], info["channels"],
        )
        if native.size == 0:
            continue
        yield AudioChunk(resample_to_48k(native, info["sample_rate"]), offset)


def decode_range(path: str | Path, offset: int, length: int) -> np.ndarray:
    """Arbitrary range as mono 48 kHz float32 (used by the Flow analyzer)."""
    info = probe(path)
    native = _decode_range_native(
        path, offset, length, info["sample_rate"], info["channels"])
    return resample_to_48k(native, info["sample_rate"])
