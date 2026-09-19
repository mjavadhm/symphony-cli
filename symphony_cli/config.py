"""Locked parameters, ported 1:1 from the Symphony Android app.

Never change a value here without re-running the golden parity tests:
any drift silently breaks compatibility with indexes produced by the app.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- audio / chunking (AudioDecoder.kt) ---
SAMPLE_RATE = 48000
CHUNK_STEP_SECONDS = 30      # spacing between chunk starts
CHUNK_DECODE_SECONDS = 11    # how much audio is decoded per chunk
MAX_CHUNKS = 20              # app keeps only the first 20 offsets
MIN_TAIL_SECONDS = 10        # skip a trailing offset shorter than this

# --- mel spectrogram (MelSpectrogramExtractor.kt) ---
N_FFT = 1024
HOP_LENGTH = 480
N_MELS = 64
F_MIN = 50.0
F_MAX = 14000.0
MAX_FRAMES = 1001            # ~10 seconds
LOG_FLOOR = 1e-10

# --- models (ClapModelRunner.kt / ModelManager.kt) ---
EMBEDDING_DIM = 512
AUDIO_MODEL_NAME = "clap_audio_encoder_int8.onnx"
TEXT_MODEL_NAME = "clap_text_encoder_int8.onnx"

# --- tokenizer (RobertaTokenizer.kt) ---
TOKENIZER_MAX_LENGTH = 77
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 1

# --- search (SemanticSearchRepository.kt / SemanticSearchEngine.kt) ---
QUERY_TEMPLATES = ("{q}", "a {q} song", "{q} music")
CANDIDATE_CHUNKS = 200
SCORE_TOP3_WEIGHT = 0.5
SCORE_MEAN_WEIGHT = 0.3
SCORE_MAX_WEIGHT = 0.2
MMR_DIVERSITY_WEIGHT = 0.3
MMR_SKIP_ABOVE_TOPN = 100
DUPLICATE_DURATION_TOLERANCE = 2   # seconds, insertTrack()
EMBEDDED_DURATION_TOLERANCE = 5    # seconds, isTrackEmbedded()

# --- paths ---
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
MODELS_DIR = Path(os.environ.get("SYMPHONY_MODELS_DIR", REPO_ROOT / "models"))
ASSETS_DIR = Path(os.environ.get("SYMPHONY_ASSETS_DIR", REPO_ROOT / "assets"))
DATA_DIR = Path(os.environ.get("SYMPHONY_DATA_DIR", Path.home() / ".local/share/symphony-cli"))

AUDIO_MODEL_PATH = MODELS_DIR / AUDIO_MODEL_NAME
TEXT_MODEL_PATH = MODELS_DIR / TEXT_MODEL_NAME
VOCAB_PATH = ASSETS_DIR / "vocab.json"
MERGES_PATH = ASSETS_DIR / "merges.txt"
