"""Model management: download, list, switch."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from rich.console import Console
from rich.table import Table

from llm.config import find_config, load_config

app = typer.Typer(help="Download, list, and switch GGUF models.")
console = Console()


class ModelEntry(NamedTuple):
    alias: str
    repo: str
    filename: str
    size: str
    description: str


# Curated catalog of recommended models. Add new entries here as models are released.
# Both alias and filename are accepted everywhere (download, switch).
KNOWN_MODELS: list[ModelEntry] = [
    # ── Qwen 2.5 Coder ───────────────────────────────────────────────────────
    ModelEntry(
        "qwen2.5-coder-7b-q8",
        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
        "~8 GB",
        "Qwen 2.5 Coder 7B — fastest, good for quick tasks",
    ),
    ModelEntry(
        "qwen2.5-coder-14b-q4",
        "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "~8.5 GB",
        "Qwen 2.5 Coder 14B — best speed/quality balance (default)",
    ),
    ModelEntry(
        "qwen2.5-coder-32b-q4",
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "~18 GB",
        "Qwen 2.5 Coder 32B — strong coding model",
    ),
    ModelEntry(
        "qwen2.5-coder-32b-q8",
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "Qwen2.5-Coder-32B-Instruct-Q8_0.gguf",
        "~34 GB",
        "Qwen 2.5 Coder 32B — high precision",
    ),
    ModelEntry(
        "qwen2.5-72b-q4",
        "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "~42 GB",
        "Qwen 2.5 72B — near-frontier quality (fits in 62 GB)",
    ),
    # ── Qwen 3 ───────────────────────────────────────────────────────────────
    ModelEntry(
        "qwen3-30b-moe-q4",
        "bartowski/Qwen3-30B-A3B-GGUF",
        "Qwen3-30B-A3B-Q4_K_M.gguf",
        "~17 GB",
        "Qwen 3 30B MoE — efficient MoE architecture",
    ),
    # ── Gemma 4 ──────────────────────────────────────────────────────────────
    ModelEntry(
        "gemma-4-31b-q4",
        "bartowski/google_gemma-4-31B-it-GGUF",
        "google_gemma-4-31B-it-Q4_K_M.gguf",
        "~20 GB",
        "Gemma 4 31B — newest Google model, multimodal",
    ),
    # ── Gemma 3 ──────────────────────────────────────────────────────────────
    ModelEntry(
        "gemma-3-27b-q4",
        "bartowski/google_gemma-3-27b-it-GGUF",
        "google_gemma-3-27b-it-Q4_K_M.gguf",
        "~17 GB",
        "Gemma 3 27B — strong all-rounder, multimodal",
    ),
    ModelEntry(
        "gemma-3-27b-q8",
        "bartowski/google_gemma-3-27b-it-GGUF",
        "google_gemma-3-27b-it-Q8_0.gguf",
        "~29 GB",
        "Gemma 3 27B — high precision, multimodal",
    ),
    ModelEntry(
        "gemma-3-12b-q4",
        "bartowski/google_gemma-3-12b-it-GGUF",
        "google_gemma-3-12b-it-Q4_K_M.gguf",
        "~7 GB",
        "Gemma 3 12B — fast, good quality, multimodal",
    ),
    ModelEntry(
        "gemma-3-12b-q8",
        "bartowski/google_gemma-3-12b-it-GGUF",
        "google_gemma-3-12b-it-Q8_0.gguf",
        "~13 GB",
        "Gemma 3 12B — high precision, multimodal",
    ),
]

# ── Lookup helpers ────────────────────────────────────────────────────────────


def _by_alias(alias: str) -> ModelEntry | None:
    return next((m for m in KNOWN_MODELS if m.alias == alias), None)


def _by_filename(filename: str) -> ModelEntry | None:
    return next((m for m in KNOWN_MODELS if m.filename == filename), None)


def _resolve(target: str) -> ModelEntry | None:
    """Resolve an alias or filename to a ModelEntry, or None if unknown."""
    return _by_alias(target) or _by_filename(target)


def _models_dir() -> Path:
    """Return the configured models directory, creating it if needed."""
    cfg = load_config()
    d = cfg.models_path
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt_size(path: Path) -> str:
    gb = path.stat().st_size / 1_073_741_824
    return f"{gb:.1f} GB"


def _catalog_table(title: str) -> Table:
    t = Table(title=title, show_header=True)
    t.add_column("Alias", style="cyan")
    t.add_column("Size", style="green", justify="right")
    t.add_column("Description", style="white")
    for m in KNOWN_MODELS:
        t.add_row(m.alias, m.size, m.description)
    return t


@app.command("list")
def list_models() -> None:
    """List all known models — shows download status and which is active."""
    models_dir = _models_dir()
    cfg = load_config()
    active = cfg.models.active

    # Index of locally downloaded files for fast lookup
    local: dict[str, Path] = {f.name: f for f in models_dir.glob("*.gguf")}

    table = Table(title=f"Models  (downloaded to {models_dir})", show_header=True)
    table.add_column("", width=2)  # active marker
    table.add_column("Alias", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Description", style="white")
    table.add_column("Downloaded", justify="center")

    for m in KNOWN_MODELS:
        path = local.get(m.filename)
        if path:
            dl_marker = "[green]✓[/green]"
            size = _fmt_size(path)  # actual size on disk
        else:
            dl_marker = "[dim]–[/dim]"
            size = f"[dim]{m.size}[/dim]"
        active_marker = "▶" if m.filename == active else ""
        table.add_row(active_marker, m.alias, size, m.description, dl_marker)

    # Append any locally downloaded files not in the catalog
    unknown = [f for name, f in sorted(local.items()) if name not in {m.filename for m in KNOWN_MODELS}]
    for f in unknown:
        active_marker = "▶" if f.name == active else ""
        table.add_row(active_marker, "[dim]custom[/dim]", _fmt_size(f), f.name, "[green]✓[/green]")

    console.print(table)
    console.print("[dim]▶ = active   ✓ = downloaded   – = not downloaded[/dim]")
    console.print("Download: [bold]uv run llm model download <alias>[/bold]")


@app.command("download")
def download(
    target: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Alias (e.g. gemma-4-31b-q4) or raw HuggingFace repo ID "
                "(e.g. bartowski/Qwen2.5-Coder-14B-Instruct-GGUF)."
            )
        ),
    ] = None,
    filename: Annotated[
        str | None,
        typer.Option("--file", "-f", help="GGUF filename — required when passing a raw repo ID."),
    ] = None,
) -> None:
    """Download a GGUF model from HuggingFace."""
    if target is None:
        console.print(_catalog_table("Available models  (uv run llm model download <alias>)"))
        return

    # Resolve alias first; fall back to treating target as a raw repo ID
    entry = _resolve(target)
    if entry:
        repo_id = entry.repo
        dl_filename = filename or entry.filename
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

    # Show the alias to use for switching, if known
    resolved = _by_filename(dl_filename)
    switch_target = resolved.alias if resolved else dl_filename
    console.print(f"Switch to it with: [bold]uv run llm model switch {switch_target}[/bold]")


@app.command("switch")
def switch(
    target: Annotated[str, typer.Argument(help="Alias or GGUF filename to make active.")],
    restart: Annotated[
        bool,
        typer.Option("--restart/--no-restart", help="Restart server after switching."),
    ] = True,
) -> None:
    """Set the active model in config.toml (accepts alias or filename)."""

    cfg = load_config()

    # Resolve alias → filename if needed
    entry = _resolve(target)
    model_name = entry.filename if entry else target

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
    label = f"{entry.alias} ({model_name})" if entry else model_name
    console.print(f"[green]Active model set to[/green] {label}")

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
