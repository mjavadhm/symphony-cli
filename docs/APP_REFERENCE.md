# Symphony app reference

What the CLI is porting, extracted from
[mjavadhm/symphony-ai](https://github.com/mjavadhm/symphony-ai). Kotlin paths
below are relative to `app/src/main/java/io/github/zyrouge/symphony/`.

## Architecture in one paragraph

`Symphony` is an `AndroidViewModel` acting as a service locator (no DI
framework). It owns `permission`, `settings`, `database`, `groove` (media
library), `radio` (playback), `translator`, `semanticSearch`, `recommendation`,
`flow`, `llm`, `llmTasks` and `spotizer`. State is exposed through `StateFlow`;
playback events go through a small `Eventer<T>`.

## AI subsystems

### Semantic search — `services/search/`

| File | Role |
| --- | --- |
| `SemanticSearchEngine.kt` | orchestration, indexing job, JSON import/export |
| `data/SemanticSearchRepository.kt` | ObjectBox storage, hybrid scoring, MMR |
| `ml/AudioDecoder.kt` | MediaCodec decoding, mono downmix, linear resample |
| `ml/MelSpectrogramExtractor.kt` | pure-Kotlin log-mel, CLAP parameters |
| `ml/RobertaTokenizer.kt` | byte-level BPE, `max_length=77` |
| `ml/ClapModelRunner.kt` | two ONNX sessions, L2-normalized output |
| `ml/ModelManager.kt` | model import and validation |

Indexing pipeline: decode a chunk every 30s (11s of audio) at 48 kHz mono, run
the mel extractor, feed the audio encoder, keep only the 512-dim vector. Each
chunk is normalized individually, then the track's mean embedding is the
normalized mean of the chunks.

Hybrid retrieval: 200 nearest chunks from HNSW, grouped by track, scored as
`0.5 * mean(top 3 chunk similarities) + 0.3 * similarity(mean embedding) +
0.2 * max chunk similarity`, then MMR-reranked with `diversityWeight = 0.3`.

### Recommender — `services/recommendation/RecommendationEngine.kt`

* Taste points from 90 days of history: weight `completionRate * recency`,
  recency `0.5 ^ (age / halfLife)` with a default 14-day half-life; skips weigh
  `-0.5 * recency`; explicit feedback is `+1` (like) and `-0.8` (dislike).
* Daily Mixes: k-means over the positive points (farthest-first init, 8
  iterations, clusters with fewer than 5 members dropped), cached per calendar
  day; `discoveryRatio` default `0.3`, `dailyMixSize` 10-100, at most 3 songs per
  artist, skipped-in-30-days and disliked tracks excluded.
* Context Mixes: hour ranges, including ranges that wrap past midnight;
  defaults are Morning 6-11 and Night 21-2.
* Mood Mixes: built-ins Sad / Workout / Chill / Night Drive / Focus, several
  prompts each, one prompt per day chosen by a seeded RNG
  (`mixId * 31 + dateHash + rerollSalt`), pool of `3 * trackCount`, weighted
  noise `jitter = 0.25 * topScore`.
* Autoplay: mean embedding of the last three queued songs, 10 similar tracks,
  excluding anything skipped in the last week.
* Smart Shuffle: `ln(1+plays) - 1.2*ln(1+skips)` plus Gumbel noise over 30 days
  of history; the current song stays first.

### Flow — `services/flow/FlowAnalyzer.kt`

Pure DSP, no model. Takes the first and last audible 5 seconds (silence and
fades trimmed with a 5% of peak-RMS threshold over 0.25s windows) and computes
four normalized features: `energy` (log RMS), `centroid`, `rolloff` (85% of
frame energy) and `onset` density. Transition distance is
`0.35*Δcentroid + 0.30*Δenergy + 0.20*Δrolloff + 0.15*Δonset`; `orderByFlow`
greedily chains each tail to the nearest head.

### LLM — `services/llm/`

`LlmClient` talks to any OpenAI-compatible `{baseUrl}/chat/completions`, with
both plain and SSE-streaming modes, and a usage mode of `Off` / `Manual` / `Auto`.
Four user-editable system prompts: `chatBehavior`, `chatStructure`,
`mixPromptsSystem`, `nameMixSystem`.

`LlmTasks` exposes `generateMixPrompts` (mix description to CLAP prompts, under
12 words each, retried when the model returns too few), `nameMix` (2-4 word
Daily Mix names, Auto mode only) and `discoverChat`, whose contract is a single
JSON object per turn:

```json
{"action": "ask",    "reply": "..."}
{"action": "search", "reply": "...", "prompts": ["...", "..."]}
```

`ReplyStreamExtractor` pulls only the `reply` field out of the JSON stream as it
arrives, unescaping as it goes, and falls back to raw text if the model does not
return JSON. Both parsers are deliberately lenient: malformed JSON becomes a
plain message rather than an error.

## Data model

| Store | Where | Contents |
| --- | --- | --- |
| `TrackEntity` | ObjectBox | filePath, title, artist, durationSeconds, meanEmbedding (HNSW 512, cosine) |
| `TrackChunkEntity` | ObjectBox | offsetSeconds, embedding (HNSW 512), relation to track |
| `playback_history` | Room | songId, playedAt, completionRate, skipped, hourOfDay, dayOfWeek, source, audioOutput, deviceId, title, artist |
| `custom_mixes` | Room | name, icon, prompt, prompts (newline separated), description, trackCount, isBuiltIn |
| `mix_contexts` | Room | name, icon, startHour, endHour, enabled |
| `mix_feedback` | Room | songId, title, artist, liked |
| `track_flow` | Room | head/tail x (energy, centroid, rolloff, onset) |
| `chat_sessions` / `chat_messages` | Room | chat history; kind = user/bot/results with prompts and songIds |

`source` values seen in history: `queue`, `daily_mix`, `mood_mix`,
`discover_prompt`, `discover_similar`.

## Interchange format

`SemanticSearchEngine.importJsonDatabase` / `exportJsonDatabase`:

```json
[{"filename": "/storage/.../song.mp3", "title": "...", "artist": "...",
  "duration": 354, "chunks": [[512 floats], ...]}]
```

* `filename` is the full device path, `artist` is the joined artist list,
  `duration` is in seconds.
* Any string may be `null` and must be read as an empty string.
* Chunk offsets are not stored; on import they become `index * 30`, so array
  order is the only thing that matters.
* On import, an existing track is replaced when the title matches
  case-insensitively, the normalized artists match, and durations differ by at
  most 2 seconds. `isTrackEmbedded` uses a looser ±5s tolerance.
* `normalizeKey` lowercases, strips bracketed segments and a trailing `feat...`,
  and collapses whitespace.

Track identity therefore rests on title + artist + duration. It is fragile, and
that fragility is inherited on purpose so both sides behave the same way.

## Android pieces the CLI replaces

| App | CLI |
| --- | --- |
| MediaCodec / MediaExtractor | ffmpeg + ffprobe |
| MediaMetadataRetriever, metaphony (JNI/TagLib) | mutagen |
| SAF / DocumentFile | ordinary filesystem walk |
| ObjectBox HNSW | hnswlib (or sqlite-vec / usearch) |
| Room | plain SQLite |
| SharedPreferences | config file |
| MediaPlayer + MediaSession | mpv / ffplay |
| Foreground service notification | terminal progress bar |
| ONNX Runtime for Android | onnxruntime, same int8 models unchanged |
