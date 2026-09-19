# Roadmap

A separate repo from the Android app, so nothing here is shared at build time.
The only thing that keeps the two in sync is this document plus the golden
parity tests.

## Locked decisions

These never change between phases. Every value is copied from the app and lives
in [`symphony_cli/config.py`](../symphony_cli/config.py); no magic numbers
anywhere else.

| Area | Values |
| --- | --- |
| Audio | `48000 Hz` mono · chunk step `30s` · decode `11s` per chunk · max `20` chunks · skip a trailing offset shorter than `10s` |
| Mel | `nFft=1024` · `hop=480` · `nMels=64` · `fMin=50` · `fMax=14000` · `maxFrames=1001` · periodic Hann · Slaney-normalized filterbank · `10*log10(max(x, 1e-10))` · repeat-pad |
| Models | `clap_audio_encoder_int8.onnx` + `clap_text_encoder_int8.onnx` · 512-dim L2-normalized output |
| Text | RoBERTa BPE from `vocab.json` + `merges.txt` · `max_length=77` · bos=0 eos=2 pad=1 · a leading space is prepended to the text |
| Search | prompt ensemble `q` / `a q song` / `q music` · 200 candidate chunks · `0.5*top3mean + 0.3*mean + 0.2*max` · MMR `0.3` (skipped above topN 100) |
| Exchange | exactly the app's JSON: `[{filename, title, artist, duration, chunks[][]}]`; chunk offsets are not stored and are rebuilt as `index * 30` |

## Phases

### Phase 0 — scaffold ✅

Repo layout, `pyproject.toml`, dependencies (`onnxruntime`, `numpy`, `typer`,
`rich`, `mutagen`, `hnswlib`; ffmpeg as a system dependency), `config.py` with
every locked parameter, CLI shell.

**Exit:** `symphony --help` and `symphony doctor` work.

### Phase 1 — embedding parity ✅ (pending golden fixtures)

* `core/decoder.py` — ffmpeg to PCM, channel averaging, the app's own linear resampler
* `core/mel.py` — a 1:1 port of `MelSpectrogramExtractor`
* `core/tokenizer.py` — RoBERTa BPE over the same assets
* `core/clap_runner.py` — two ONNX sessions, normalized output
* `core/embedder.py` — file/text embedding plus the prompt ensemble
* commands: `embed`, `embed-text`

**Exit gate:** for every chunk, cosine similarity with the app's export `> 0.999`
and max absolute difference `< 1e-3`. Nothing else starts until `pytest` is green.

### Phase 2 — library and index

* Filesystem walk, tags via mutagen, minimum-duration filter, blacklist/whitelist
* SQLite for `tracks` and `chunks`, HNSW cosine index over both levels
* Deduplication exactly as `insertTrack` does it: equal title (case-insensitive),
  equal normalized artist, duration difference `<= 2s`
* Commands: `index [path]` (resumable, with progress), `import-index`,
  `export-index`, `stats`
* Multi-process embedding — this is where the CLI beats the phone

**Exit gate:** import the app's export, re-export it, and have the app accept the
result (clean round trip).

### Phase 3 — search

* Prompt ensemble, hybrid score and MMR with the exact same coefficients
* `search <query>` with both limit modes: fixed count, or relative similarity
  threshold measured against the best result
* `similar <file>` via the mean embedding

**Exit gate:** on an identical library, top-10 overlaps acceptably with the app.

### Phase 4 — playback and history

Required before the recommender, because it produces the recommender's input.

* mpv (or ffplay) playback, queue, play/pause/next/prev/seek
* `playback_history` with the same columns: `completionRate`, `skipped`,
  `hourOfDay`, `dayOfWeek`, `source`, `deviceId`
* The recording rules must be lifted from `Radio.kt` verbatim

### Phase 5 — recommender

* Taste points over a 90-day window with a configurable half-life
* Daily Mixes via k-means (farthest-first init, 8 iterations) with a daily cache
* Context Mixes by hour range · Mood Mixes with prompt rotation and reroll ·
  like/dislike feedback
* Smart Shuffle (Gumbel noise) and Autoplay
* Commands: `mix daily|mood|context`, `queue smart-shuffle`, `like`/`dislike`

### Phase 6 — Flow

Head/tail acoustic features (`energy`, `centroid`, `rolloff`, `onset`), the
`track_flow` table, `analyze-flow` and `queue flow`. Independent of the models,
so it can be dropped or deferred without touching anything else.

### Phase 7 — LLM and chat

OpenAI-compatible client with SSE streaming, the same four system prompts, the
same JSON contract (`ask` / `search`), a terminal equivalent of
`ReplyStreamExtractor`, `chat` with automatic search and `save-as-mix`.

### Phase 8 — data exchange with the app

The phase-2 JSON import/export, plus history/mixes/feedback in the
`BackupManager` format. An optional sync server can come later; moving files is
enough to start.

### Phase 9 — polish

Config file, logging, readable errors, packaging (pipx or a single binary),
README. Optionally a textual TUI.

## Risks

* **Embedding drift** — the highest risk in the project, and it fails silently.
  The phase-1 golden test is the only thing that catches it.
* **Resampler differences** — ffmpeg and the app's hand-written resampler are not
  bit-identical. We decode at the native rate and reuse the app's linear
  interpolation; if the golden test still drifts, port the rest of `decodeChunk`
  more literally.
* **Track identity** — as fragile as in the app (title/artist/duration), by choice.
* **Indexing throughput** — desktop CPUs are far faster than a phone; parallelism
  should be designed in from phase 2.
