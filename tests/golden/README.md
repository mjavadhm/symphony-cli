# Golden fixtures

1. In the Symphony app, index 3-5 short songs.
2. Settings -> Semantic search -> **Export AI Index**.
3. Copy the exported JSON here as `reference.json`, and copy the same audio
   files into this folder (file names must match `filename` in the JSON).

`pytest` then verifies that the CLI reproduces the app's vectors.
The audio files are gitignored; keep them locally.
