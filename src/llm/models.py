"""Model management: download, list, switch."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm.config import ModelEntry, find_config, load_config

app = typer.Typer(help="Download, list, and switch GGUF models.", no_args_is_help=True)
console = Console()


# ── KNOWN_MODELS catalog (default fallback) ───────────────────────────────────
# Used as a fallback when the user has no [[models.list]] entries in
# config.toml. When [[models.list]] entries are present, those take priority.

KNOWN_MODELS: list[ModelEntry] = [
    # ── Qwen 2.5 Coder ───────────────────────────────────────────────────────
    ModelEntry(
        alias="qwen2.5-coder-7b-q8",
        repo="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
        size="~8 GB",
        description="Qwen 2.5 Coder 7B - fastest, good for quick tasks",
    ),
    ModelEntry(
        alias="qwen2.5-coder-14b-q4",
        repo="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        size="~8.5 GB",
        description="Qwen 2.5 Coder 14B - best speed/quality balance (default)",
    ),
    ModelEntry(
        alias="qwen2.5-coder-32b-q4",
        repo="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        size="~18 GB",
        description="Qwen 2.5 Coder 32B - strong coding model",
    ),
    ModelEntry(
        alias="qwen2.5-coder-32b-q8",
        repo="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="Qwen2.5-Coder-32B-Instruct-Q8_0.gguf",
        size="~34 GB",
        description="Qwen 2.5 Coder 32B - high precision",
    ),
    ModelEntry(
        alias="qwen2.5-72b-q4",
        repo="bartowski/Qwen2.5-72B-Instruct-GGUF",
        filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size="~42 GB",
        description="Qwen 2.5 72B - near-frontier quality (fits in 62 GB)",
    ),
    # ── Gemma 4 ──────────────────────────────────────────────────────────────
    ModelEntry(
        alias="gemma-4-31b-q4",
        repo="bartowski/google_gemma-4-31B-it-GGUF",
        filename="google_gemma-4-31B-it-Q4_K_M.gguf",
        size="~20 GB",
        description="Gemma 4 31B - newest Google model, multimodal",
    ),
    # ── Gemma 3 ──────────────────────────────────────────────────────────────
    ModelEntry(
        alias="gemma-3-27b-q4",
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q4_K_M.gguf",
        size="~17 GB",
        description="Gemma 3 27B - strong all-rounder, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-27b-q8",
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q8_0.gguf",
        size="~29 GB",
        description="Gemma 3 27B - high precision, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-12b-q4",
        repo="bartowski/google_gemma-3-12b-it-GGUF",
        filename="google_gemma-3-12b-it-Q4_K_M.gguf",
        size="~7 GB",
        description="Gemma 3 12B - fast, good quality, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-12b-q8",
        repo="bartowski/google_gemma-3-12b-it-GGUF",
        filename="google_gemma-3-12b-it-Q8_0.gguf",
        size="~13 GB",
        description="Gemma 3 12B - high precision, multimodal",
    ),
    # ── Qwen 3 ───────────────────────────────────────────────────────────────
    ModelEntry(
        alias="qwen3-8b-q8",
        repo="bartowski/Qwen_Qwen3-8B-GGUF",
        filename="Qwen_Qwen3-8B-Q8_0.gguf",
        size="~9 GB",
        description="Qwen3 8B - fast, near-lossless quant",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-14b-q8",
        repo="bartowski/Qwen_Qwen3-14B-GGUF",
        filename="Qwen_Qwen3-14B-Q8_0.gguf",
        size="~16 GB",
        description="Qwen3 14B - near-lossless, strong coding",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-32b-q4",
        repo="bartowski/Qwen_Qwen3-32B-GGUF",
        filename="Qwen_Qwen3-32B-Q4_K_M.gguf",
        size="~20 GB",
        description="Qwen3 32B dense - top-tier coding quality",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-30b-moe-q4",
        repo="bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        filename="Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        size="~19 GB",
        description="Qwen3 30B MoE - fast TG, outperforms QwQ-32B",
        max_output=32768,
    ),
    # ── Qwen 3.6 (April 2026) ────────────────────────────────────────────────
    ModelEntry(
        alias="qwen3.6-35b-moe-q4",
        repo="bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        filename="Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        size="~21 GB",
        description="Qwen3.6 35B MoE - SWE-bench 73%, 262K ctx",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3.6-27b-q4",
        repo="bartowski/Qwen_Qwen3.6-27B-GGUF",
        filename="Qwen_Qwen3.6-27B-Q4_K_M.gguf",
        size="~18 GB",
        description="Qwen3.6 27B dense - Apr 2026, 262K context, multimodal",
        max_output=32768,
    ),
]


# ── Lookup helpers ───────────────────────────────────────────────────────────


def _by_alias(alias: str, model_list: list[ModelEntry] | None = None) -> ModelEntry | None:
    """Resolve alias → ModelEntry from a specific list (or KNOWN_MODELS)."""
    if not model_list:
        model_list = KNOWN_MODELS
    return next((m for m in model_list if m.alias == alias), None)


def _by_filename(filename: str, model_list: list[ModelEntry] | None = None) -> ModelEntry | None:
    """Resolve filename → ModelEntry from a specific list (or KNOWN_MODELS).

    Matches on both the full catalog filename (which may include a subfolder
    like ``gguf/model.gguf``) and the basename, so lookups work after the file
    has been flattened into the models directory.
    """
    if not model_list:
        model_list = KNOWN_MODELS
    basename = Path(filename).name
    return next(
        (m for m in model_list if m.filename == filename or Path(m.filename).name == basename),
        None,
    )


def _resolve(
    target: str,
    *,
    _fallback_list: list[ModelEntry] | None = None,
) -> ModelEntry | None:
    """Resolve alias or filename to a ModelEntry from config, falling back to KNOWN_MODELS.

    _fallback_list is an internal parameter for testing - when provided,
    the function uses that list directly instead of calling load_config().
    """
    if _fallback_list is not None:
        # Testing path: use the provided list directly
        entry = _by_alias(target, _fallback_list) or _by_filename(target, _fallback_list)
        if entry is None:
            entry = _by_alias(target) or _by_filename(target)
        return entry

    cfg = load_config()
    # First try config catalog
    if cfg.models.has_catalog:
        entry = cfg.models.by_alias(target) or cfg.models.by_filename(target)
        if entry:
            return entry
    # Fallback to built-in catalog
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
    """List all known models - shows download status and which is active."""
    models_dir = _models_dir()
    cfg = load_config()
    active = cfg.models.active

    # Index of locally downloaded files for fast lookup
    local: dict[str, Path] = {f.name: f for f in models_dir.glob("*.gguf")}

    # Use config catalog if available, otherwise fall back to KNOWN_MODELS
    model_list = cfg.models.entries if cfg.models.has_catalog else KNOWN_MODELS

    table = Table(title=f"Models  (downloaded to {models_dir})", show_header=True)
    table.add_column("", width=2)  # active marker
    table.add_column("Alias", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Description", style="white", max_width=60, no_wrap=False)
    table.add_column("Downloaded", justify="center")

    for m in model_list:
        path = local.get(m.filename)
        if path:
            dl_marker = "[green]✓[/green]"
            size = _fmt_size(path)  # actual size on disk
        else:
            dl_marker = "[dim]–[/dim]"
            size = f"[dim]{m.size}[/dim]"
        active_marker = "▶" if m.filename == active or m.alias == active else ""
        table.add_row(active_marker, m.alias, size, m.description, dl_marker)

    # Append any locally downloaded files not in the catalog
    catalog_filenames = {m.filename for m in model_list}
    unknown = [f for name, f in sorted(local.items()) if name not in catalog_filenames]
    for f in unknown:
        active_marker = "▶" if f.name == active else ""
        table.add_row(active_marker, "[dim]custom[/dim]", _fmt_size(f), f.name, "[green]✓[/green]")

    console.print(table)
    console.print("[dim]▶ = active   ✓ = downloaded   – = not downloaded[/dim]")
    console.print("Download: [bold]uv run llm model download <alias>[/bold]")


def _split_subfolder(dl_filename: str) -> tuple[str | None, str]:
    """Split a catalog filename into (repo subfolder, flat local filename)."""
    dl_path = Path(dl_filename)
    if len(dl_path.parts) > 1:
        return str(dl_path.parent), dl_path.name
    return None, dl_filename


def _flatten_conflicts(hf_filename: str, source_filename: str, entries: list[ModelEntry]) -> list[ModelEntry]:
    """Catalog entries that would be saved over the top of this download.

    Catalog filenames may carry a subfolder but are always saved flat, so
    "a/model.gguf" and "b/model.gguf" both land on "model.gguf".
    """
    return [e for e in entries if Path(e.filename).name == hf_filename and e.filename != source_filename]


def _remote_size(repo_id: str, filename: str, subfolder: str | None, token: str | None) -> int | None:
    """Expected download size from HuggingFace, or None if it cannot be determined."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        url = hf_hub_url(repo_id=repo_id, filename=filename, subfolder=subfolder)
        return get_hf_file_metadata(url, token=token).size
    except Exception:
        return None


def _verify_size(path: Path, expected: int | None) -> None:
    """Reject an empty or short download before it is promoted into place."""
    actual = path.stat().st_size
    if actual == 0:
        console.print(f"[red]Download is empty:[/red] {path.name}")
        raise typer.Exit(1)
    if expected is not None and actual != expected:
        console.print(f"[red]Download is incomplete:[/red] got {actual} bytes, expected {expected}.")
        raise typer.Exit(1)


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
        typer.Option("--file", "-f", help="GGUF filename - required when passing a raw repo ID."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even if the file is already present."),
    ] = False,
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

    # When the catalog filename includes a subfolder (e.g. "gguf/model.gguf"),
    # split it so hf_hub_download fetches from the right repo path but saves
    # the file flat into models_path (where llama-server expects it).
    hf_subfolder, hf_filename = _split_subfolder(dl_filename)

    catalog = cfg.models.entries if cfg.models.has_catalog else KNOWN_MODELS
    conflicts = _flatten_conflicts(hf_filename, dl_filename, catalog)
    if conflicts:
        console.print(
            f"[red]Name collision:[/red] '{dl_filename}' would be saved as "
            f"'{hf_filename}', which is already claimed by:"
        )
        for other in conflicts:
            console.print(f"  - {other.alias} ({other.filename})")
        console.print("Give one of them a distinct filename in config.toml before downloading.")
        raise typer.Exit(1)

    dest_file = dest_dir / hf_filename
    if dest_file.exists() and not force:
        console.print(f"[yellow]Already downloaded[/yellow] → {dest_file}")
        console.print("  Re-download with [bold]--force[/bold].")
        raise typer.Exit(0)

    console.print(f"Downloading [bold]{dl_filename}[/bold] from [cyan]{repo_id}[/cyan] ...")

    expected_size = _remote_size(repo_id, hf_filename, hf_subfolder, token)

    # Stage inside the destination directory so the final os.replace() is an
    # atomic same-filesystem rename: an interrupted download can never leave a
    # truncated file where llama-server expects a complete model.
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=dest_dir))
    try:
        tmp_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                subfolder=hf_subfolder,
                local_dir=str(staging),
                token=token,
            )
        )
        _verify_size(tmp_path, expected_size)
        os.replace(tmp_path, dest_file)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    local_path = dest_file
    console.print(f"[green]Saved[/green] → {local_path}")

    # Show the alias to use for switching, if known
    cfg = load_config()
    resolved = _by_filename(hf_filename, cfg.models.entries if cfg.models.has_catalog else None)
    if not resolved and cfg.models.has_catalog is False:
        resolved = _by_filename(hf_filename, KNOWN_MODELS)
    switch_target = resolved.alias if resolved else hf_filename
    console.print(f"Switch to it with: [bold]uv run llm model switch {switch_target}[/bold]")


# Matches an `active = <value>` assignment, keeping any trailing inline comment.
_ACTIVE_RE = re.compile(r"""^(?P<prefix>\s*active\s*=\s*)(?:"[^"]*"|'[^']*')(?P<suffix>.*)$""")


def _set_active_in_config(text: str, value: str) -> str | None:
    """Rewrite `active` inside the [models] table, preserving comments and layout.

    Scoped to [models] so that an `active` key in any other table (now or
    later) is left alone. Returns None if no such key exists.
    """
    if '"' in value:
        raise ValueError(f"model name contains a quote and cannot be written to TOML: {value!r}")

    lines = text.splitlines()
    in_models = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_models = stripped == "[models]"
            continue
        if not in_models:
            continue
        match = _ACTIVE_RE.match(line)
        if match:
            lines[i] = f'{match.group("prefix")}"{value}"{match.group("suffix")}'
            trailing = "\n" if text.endswith("\n") else ""
            return "\n".join(lines) + trailing
    return None


@app.command("switch")
def switch(
    target: Annotated[str, typer.Argument(help="Alias or GGUF filename to make active.")],
    restart: Annotated[
        bool,
        typer.Option("--restart/--no-restart", help="Restart server after switching."),
    ] = True,
    cost: Annotated[
        bool,
        typer.Option("--cost", help="Show cost info after switching."),
    ] = False,
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

    # Prefer the alias: `active` is documented as holding an alias, and
    # `config show` marks the active model by comparing against aliases.
    active_value = entry.alias if entry else model_name

    # Update config.toml in place to avoid destroying comments and formatting.
    config_path = find_config()
    updated = _set_active_in_config(config_path.read_text(), active_value)
    if updated is None:
        console.print(f"[red]Could not find an 'active' key under [models] in {config_path}.[/red]")
        raise typer.Exit(1)
    config_path.write_text(updated)
    label = f"{entry.alias} ({model_name})" if entry else model_name
    console.print(f"[green]Active model set to[/green] {label}")

    if cost and entry:
        console.print(f"  Cost: ${entry.cost.input:.4f} / ${entry.cost.output:.4f} per token")

    if restart:
        if not cfg.has_local_server:
            console.print("[dim]No local server to restart (client-only mode).[/dim]")
            return
        from llm import server as srv

        console.print("Restarting server...")
        srv.restart()


# ── init-catalog command ────────────────────────────────────────────────────


@app.command("show")
def show(
    target: Annotated[str, typer.Argument(help="Alias or filename to show details for.")],
) -> None:
    """Show detailed info for a model."""
    entry = _resolve(target)
    if not entry:
        console.print(f"[red]Model not found:[/red] {target}")
        console.print("Run [bold]uv run llm model list[/bold] to see available models.")
        raise typer.Exit(1)

    cfg = load_config()
    models_dir = cfg.models_path
    model_path = models_dir / entry.filename
    downloaded = model_path.exists()
    actual_size = _fmt_size(model_path) if downloaded else None

    console.print(f"\n[bold cyan]{entry.alias}[/bold cyan]")
    console.print(f"  Repository:  [cyan]{entry.repo}[/cyan]")
    console.print(f"  Filename:    {entry.filename}")
    console.print(f"  Size:        {entry.size}  (on disk: {actual_size or '–'})")
    console.print(f"  Description: {entry.description}")
    console.print(f"  Max output:  {entry.max_output} tokens")
    console.print(f"  Cost:        {entry.cost.input}/ {entry.cost.output} $ per token")
    console.print(f"  Downloaded:  {'[green]✓ yes[/green]' if downloaded else '[red]– no[/red]'}")
    console.print()

    if not downloaded:
        console.print(f"Download: [bold]uv run llm model download {entry.alias}[/bold]")


# ── cost command ────────────────────────────────────────────────────────────
