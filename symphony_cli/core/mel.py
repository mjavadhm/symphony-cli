"""Log-mel spectrogram, ported 1:1 from MelSpectrogramExtractor.kt.

Parameters match CLAP's preprocessor_config.json. The output is a flattened
float32 array shaped [1, 1, MAX_FRAMES, N_MELS], ready for the ONNX audio encoder.
"""
from __future__ import annotations

import numpy as np

from .. import config


def _hz_to_mel(hz: float | np.ndarray):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _build_filterbank() -> np.ndarray:
    n_bins = config.N_FFT // 2 + 1
    n_mels = config.N_MELS
    mel_min = _hz_to_mel(config.F_MIN)
    mel_max = _hz_to_mel(config.F_MAX)
    # the app stores the mel points as float32 before converting back to Hz
    mel_points = (mel_min + np.arange(n_mels + 2) * (mel_max - mel_min) / (n_mels + 1)).astype(np.float32)
    fft_bins = ((config.N_FFT + 1) * _mel_to_hz(mel_points) / config.SAMPLE_RATE).astype(np.float32)

    bank = np.zeros((n_mels, n_bins), dtype=np.float32)
    k = np.arange(n_bins, dtype=np.float32)
    for m in range(n_mels):
        left, center, right = fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]
        rising = (k - left) / (center - left + 1e-10)
        falling = (right - k) / (right - center + 1e-10)
        row = np.where(k < left, 0.0,
                       np.where(k <= center, rising,
                                np.where(k <= right, falling, 0.0)))
        # Slaney normalization, as in librosa
        enorm = 2.0 / (np.float32(_mel_to_hz(mel_points[m + 2])) - np.float32(_mel_to_hz(mel_points[m])))
        bank[m] = (row * enorm).astype(np.float32)
    return bank


_FILTERBANK = _build_filterbank()
# periodic Hann window, identical to the app: 0.5 * (1 - cos(2*pi*i/nFft))
_HANN = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(config.N_FFT) / config.N_FFT))).astype(np.float32)


def _power_spectrogram(samples: np.ndarray) -> np.ndarray:
    n_fft, hop = config.N_FFT, config.HOP_LENGTH
    n_bins = n_fft // 2 + 1
    n_frames = (samples.size - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((1, n_bins), dtype=np.float32)
    indices = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[indices] * _HANN[None, :]
    spectrum = np.fft.rfft(frames.astype(np.float32), n=n_fft, axis=1)
    return (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)


def _pad_or_truncate(mel: np.ndarray) -> np.ndarray:
    """repeatpad strategy: repeat existing frames until MAX_FRAMES."""
    frames = mel.shape[0]
    if frames >= config.MAX_FRAMES:
        return mel[: config.MAX_FRAMES]
    return mel[np.arange(config.MAX_FRAMES) % frames]


def extract(samples: np.ndarray) -> np.ndarray:
    """Mono 48 kHz float32 samples -> flattened [1, 1, MAX_FRAMES, N_MELS]."""
    samples = np.asarray(samples, dtype=np.float32)
    power = _power_spectrogram(samples)
    mel = power @ _FILTERBANK.T
    log_mel = (10.0 * np.log10(np.maximum(mel, config.LOG_FLOOR))).astype(np.float32)
    return _pad_or_truncate(log_mel).reshape(-1).astype(np.float32)
