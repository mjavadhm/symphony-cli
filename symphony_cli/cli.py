"""symphony-cli entry point."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, config

app = typer.Typer(add_completion=False, help="Symphony AI on the command line.")
console = Console()


@app.command()
def version() -> None:
    """Print the version and the locked parameters."""
    table = Table(title=f"symphony-cli {__version__}")
    table.add_column("parameter")
    table.add_column("value")
    for key in ("SAMPLE_RATE", "CHUNK_STEP_SECONDS", "CHUNK_DECODE_SECONDS",
                "N_FFT", "HOP_LENGTH", "N_MELS", "MAX_FRAMES",
                "EMBEDDING_DIM", "TOKENIZER_MAX_LENGTH"):
        table.add_row(key, str(getattr(config, key)))
    console.print(table)


@app.command()
def doctor() -> None:
    """Check models, assets and external tools."""
    import shutil

    checks = [
        ("ffmpeg", shutil.which("ffmpeg") is not None, "install ffmpeg"),
        ("ffprobe", shutil.which("ffprobe") is not None, "install ffmpeg"),
        ("audio model", config.AUDIO_MODEL_PATH.exists(), str(config.AUDIO_MODEL_PATH)),
        ("text model", config.TEXT_MODEL_PATH.exists(), str(config.TEXT_MODEL_PATH)),
        ("vocab.json", config.VOCAB_PATH.exists(), str(config.VOCAB_PATH)),
        ("merges.txt", config.MERGES_PATH.exists(), str(config.MERGES_PATH)),
    ]
    for name, ok, hint in checks:
        mark = "[green]ok[/green]" if ok else "[red]missing[/red]"
        console.print(f"{mark}  {name}" + ("" if ok else f"  -> {hint}"))


@app.command()
def embed(
    file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(None, "--output", "-o",
                                help="Write the result as app-compatible JSON."),
) -> None:
    """Embed one audio file and show the chunk count."""
    from .core.embedder import Embedder
    from .ioformat import json_index

    track = Embedder().embed_file(file)
    console.print(f"chunks: {len(track.chunks)}  duration: {track.duration_seconds}s")
    console.print(f"mean[:8]: {track.mean_embedding[:8]}")
    if output:
        json_index.write(output, [track])
        console.print(f"written -> {output}")


@app.command(name="embed-text")
def embed_text(query: str) -> None:
    """Embed a text query with the prompt ensemble."""
    from .core.embedder import Embedder

    vector = Embedder().embed_query(query)
    console.print(f"dim: {vector.size}  first 8: {vector[:8]}")


# --- placeholders for the next phases ---

@app.command()
def index(path: Path = typer.Argument(..., exists=True)) -> None:
    """Phase 2: scan a folder and build the local index."""
    raise typer.Exit(console.print("[yellow]not implemented yet (phase 2)[/yellow]") or 1)


@app.command()
def search(query: str) -> None:
    """Phase 3: hybrid semantic search."""
    raise typer.Exit(console.print("[yellow]not implemented yet (phase 3)[/yellow]") or 1)


if __name__ == "__main__":
    app()
