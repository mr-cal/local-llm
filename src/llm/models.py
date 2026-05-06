"""Model management: download, list, switch."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm.config import find_config, load_config

app = typer.Typer(help="Download, list, and switch GGUF models.")
console = Console()

# Curated list of recommended Qwen models with HuggingFace repo + filename.
# Add more entries here as new models are released.
KNOWN_MODELS: dict[str, tuple[str, str]] = {
    # alias: (hf_repo_id, gguf_filename)
    "qwen2.5-coder-7b-q8": (
        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
    ),
    "qwen2.5-coder-14b-q4": (
        "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
    ),
    "qwen2.5-coder-32b-q4": (
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
    ),
    "qwen2.5-coder-32b-q8": (
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "Qwen2.5-Coder-32B-Instruct-Q8_0.gguf",
    ),
    "qwen2.5-72b-q4": (
        "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
    ),
    "qwen3-30b-moe-q4": (
        "bartowski/Qwen3-30B-A3B-GGUF",
        "Qwen3-30B-A3B-Q4_K_M.gguf",
    ),
}


def _models_dir() -> Path:
    """Return the configured models directory, creating it if needed."""
    cfg = load_config()
    d = cfg.models_path
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_size(path: Path) -> str:
    gb = path.stat().st_size / 1_073_741_824
    return f"{gb:.1f} GB"


@app.command("list")
def list_models(
    known: Annotated[bool, typer.Option("--known", help="Also show known downloadable models.")] = False,
) -> None:
    """List locally available GGUF models."""
    models_dir = _models_dir()
    gguf_files = sorted(models_dir.glob("*.gguf"))

    cfg = load_config()
    active = cfg.models.active

    table = Table(title=f"Local models in {models_dir}", show_header=True)
    table.add_column("Active", style="cyan", width=6)
    table.add_column("Filename", style="white")
    table.add_column("Size", style="green", justify="right")

    for f in gguf_files:
        marker = "✓" if f.name == active else ""
        table.add_row(marker, f.name, _fmt_size(f))

    if not gguf_files:
        console.print(f"[yellow]No .gguf files found in {models_dir}[/yellow]")
        console.print("Run [bold]uv run llm model download --list[/bold] to see available models.")
    else:
        console.print(table)

    if known:
        kt = Table(title="Known downloadable models", show_header=True)
        kt.add_column("Alias", style="cyan")
        kt.add_column("Filename", style="white")
        kt.add_column("HF Repo", style="dim")
        for alias, (repo, filename) in KNOWN_MODELS.items():
            kt.add_row(alias, filename, repo)
        console.print(kt)


@app.command("download")
def download(
    target: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Model alias (e.g. qwen2.5-coder-14b-q4) OR "
                "HuggingFace repo ID (e.g. bartowski/Qwen2.5-Coder-14B-Instruct-GGUF)."
            )
        ),
    ] = None,
    filename: Annotated[
        str | None,
        typer.Option("--file", "-f", help="Specific GGUF filename to download (required for raw repo IDs)."),
    ] = None,
    list_known: Annotated[bool, typer.Option("--list", help="List known model aliases and exit.")] = False,
) -> None:
    """Download a GGUF model from HuggingFace."""
    if list_known:
        t = Table(title="Known model aliases", show_header=True)
        t.add_column("Alias", style="cyan")
        t.add_column("Filename")
        t.add_column("HF Repo", style="dim")
        for alias, (repo, fname) in KNOWN_MODELS.items():
            t.add_row(alias, fname, repo)
        console.print(t)
        return

    if target is None:
        console.print("[red]TARGET is required.[/red] Use --list to see available aliases.")
        raise typer.Exit(1)

    # Resolve alias → repo + filename
    if target in KNOWN_MODELS:
        repo_id, dl_filename = KNOWN_MODELS[target]
        if filename:
            dl_filename = filename  # allow override
    else:
        repo_id = target
        if not filename:
            console.print(
                "[red]--file is required when passing a raw HuggingFace repo ID.[/red]\n"
                f"Example: uv run llm model download {target} --file model.gguf"
            )
            raise typer.Exit(1)
        dl_filename = filename

    cfg = load_config()
    dest_dir = cfg.models_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]huggingface-hub not installed.[/red] Run: uv sync")
        raise typer.Exit(1) from None

    token = cfg.models.hf_token or None
    console.print(f"Downloading [bold]{dl_filename}[/bold] from [cyan]{repo_id}[/cyan] ...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=dl_filename,
        local_dir=str(dest_dir),
        token=token,
    )
    console.print(f"[green]Saved[/green] → {local_path}")
    console.print(f"Switch to it with: [bold]uv run llm model switch {dl_filename}[/bold]")


@app.command("switch")
def switch(
    model_name: Annotated[str, typer.Argument(help="GGUF filename to make active.")],
    restart: Annotated[
        bool,
        typer.Option("--restart/--no-restart", help="Restart server after switching."),
    ] = True,
) -> None:
    """Set the active model in config.toml and optionally restart the server."""

    cfg = load_config()
    model_path = cfg.models_path / model_name

    if not model_path.exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        console.print("Run [bold]uv run llm model list[/bold] to see available models.")
        raise typer.Exit(1)

    # Update config.toml in-place using regex to avoid destroying comments
    config_path = find_config()
    text = config_path.read_text()

    import re

    text = re.sub(
        r'^(active\s*=\s*)"[^"]*"',
        f'\\1"{model_name}"',
        text,
        flags=re.MULTILINE,
    )
    config_path.write_text(text)
    console.print(f"[green]Active model set to[/green] {model_name}")

    if restart:
        import os

        from llm import server as srv

        pid_path = Path(".server.pid")
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                console.print("Restarting server...")
                srv.restart()
            except (ValueError, ProcessLookupError):
                console.print("[dim]Server not running — skipping restart.[/dim]")
        else:
            console.print("[dim]Server not running — skipping restart.[/dim]")
