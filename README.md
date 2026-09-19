# symphony-cli

A command-line port of the AI layer of
[Symphony AI](https://github.com/mjavadhm/symphony-ai) — CLAP-based semantic
music search, mixes and recommendations for your local library.

It is **data-compatible with the Android app**: the embedding index uses the
app's exact JSON format, so you can index a library on your desktop and import
it into the phone (or the other way round).

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 0 | scaffold, locked parameters, CLI shell | done |
| 1 | embedding parity: decoder, mel, tokenizer, CLAP | done, needs golden fixtures |
| 2 | library scan, SQLite + HNSW index, import/export | todo |
| 3 | hybrid search, MMR, `similar` | todo |
| 4 | playback + history | todo |
| 5 | recommender: daily/mood/context mixes, smart shuffle | todo |
| 6 | Flow transitions | todo |
| 7 | LLM chat | todo |
| 8 | data exchange with the app | todo |
| 9 | polish and packaging | todo |

## Requirements

* Python 3.10+
* `ffmpeg` and `ffprobe` on PATH
* The two CLAP ONNX models in `models/`:
  * `clap_audio_encoder_int8.onnx`
  * `clap_text_encoder_int8.onnx`
* The tokenizer assets in `assets/`: `vocab.json`, `merges.txt`
  (the same files the app ships in `app/src/main/assets/`)

## Install

```bash
pip install -e .
symphony doctor
```

## Usage

```bash
symphony version            # show the locked DSP/model parameters
symphony doctor             # check models, assets and ffmpeg
symphony embed song.mp3     # embed one file
symphony embed song.mp3 -o out.json   # app-compatible JSON
symphony embed-text "dark synthwave for driving at night"
```

## Parity

Every number that affects an embedding lives in `symphony_cli/config.py` and is
copied from the app. Before touching any of them, run:

```bash
pytest
```

See `tests/golden/README.md` for how to produce the reference fixtures from the
app's **Export AI Index**.

## License

AGPL-3.0, matching the upstream project.
