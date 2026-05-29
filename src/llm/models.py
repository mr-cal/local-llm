"""Model management: download, list, switch."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm.config import ModelEntry, find_config, load_config

app = typer.Typer(help="Download, list, and switch GGUF models.", no_args_is_help=True)
console = Console()


# ── KNOWN_MODELS catalog (dual-catalog fallback) ──────────────────────────────
# Kept as a default fallback during Phase 1 migration. When the user has
# [[models.list]] entries in config.toml, those take priority.

KNOWN_MODELS: list[ModelEntry] = [
    # ── Qwen 2.5 Coder ───────────────────────────────────────────────────────
    ModelEntry(
        alias="qwen2.5-coder-7b-q8",
        repo="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
        size="~8 GB",
        description="Qwen 2.5 Coder 7B — fastest, good for quick tasks",
    ),
    ModelEntry(
        alias="qwen2.5-coder-14b-q4",
        repo="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        size="~8.5 GB",
        description="Qwen 2.5 Coder 14B — best speed/quality balance (default)",
    ),
    ModelEntry(
        alias="qwen2.5-coder-32b-q4",
        repo="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        size="~18 GB",
        description="Qwen 2.5 Coder 32B — strong coding model",
    ),
    ModelEntry(
        alias="qwen2.5-coder-32b-q8",
        repo="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="Qwen2.5-Coder-32B-Instruct-Q8_0.gguf",
        size="~34 GB",
        description="Qwen 2.5 Coder 32B — high precision",
    ),
    ModelEntry(
        alias="qwen2.5-72b-q4",
        repo="bartowski/Qwen2.5-72B-Instruct-GGUF",
        filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size="~42 GB",
        description="Qwen 2.5 72B — near-frontier quality (fits in 62 GB)",
    ),
    # ── Gemma 4 ──────────────────────────────────────────────────────────────
    ModelEntry(
        alias="gemma-4-31b-q4",
        repo="bartowski/google_gemma-4-31B-it-GGUF",
        filename="google_gemma-4-31B-it-Q4_K_M.gguf",
        size="~20 GB",
        description="Gemma 4 31B — newest Google model, multimodal",
    ),
    # ── Gemma 3 ──────────────────────────────────────────────────────────────
    ModelEntry(
        alias="gemma-3-27b-q4",
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q4_K_M.gguf",
        size="~17 GB",
        description="Gemma 3 27B — strong all-rounder, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-27b-q8",
        repo="bartowski/google_gemma-3-27b-it-GGUF",
        filename="google_gemma-3-27b-it-Q8_0.gguf",
        size="~29 GB",
        description="Gemma 3 27B — high precision, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-12b-q4",
        repo="bartowski/google_gemma-3-12b-it-GGUF",
        filename="google_gemma-3-12b-it-Q4_K_M.gguf",
        size="~7 GB",
        description="Gemma 3 12B — fast, good quality, multimodal",
    ),
    ModelEntry(
        alias="gemma-3-12b-q8",
        repo="bartowski/google_gemma-3-12b-it-GGUF",
        filename="google_gemma-3-12b-it-Q8_0.gguf",
        size="~13 GB",
        description="Gemma 3 12B — high precision, multimodal",
    ),
    # ── Qwen 3 ───────────────────────────────────────────────────────────────
    ModelEntry(
        alias="qwen3-8b-q8",
        repo="bartowski/Qwen_Qwen3-8B-GGUF",
        filename="Qwen_Qwen3-8B-Q8_0.gguf",
        size="~9 GB",
        description="Qwen3 8B — fast, near-lossless quant",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-14b-q8",
        repo="bartowski/Qwen_Qwen3-14B-GGUF",
        filename="Qwen_Qwen3-14B-Q8_0.gguf",
        size="~16 GB",
        description="Qwen3 14B — near-lossless, strong coding",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-32b-q4",
        repo="bartowski/Qwen_Qwen3-32B-GGUF",
        filename="Qwen_Qwen3-32B-Q4_K_M.gguf",
        size="~20 GB",
        description="Qwen3 32B dense — top-tier coding quality",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3-30b-moe-q4",
        repo="bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        filename="Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        size="~19 GB",
        description="Qwen3 30B MoE — fast TG, outperforms QwQ-32B",
        max_output=32768,
    ),
    # ── Qwen 3.6 (April 2026) ────────────────────────────────────────────────
    ModelEntry(
        alias="qwen3.6-35b-moe-q4",
        repo="bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        filename="Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        size="~21 GB",
        description="Qwen3.6 35B MoE — SWE-bench 73%, 262K ctx",
        max_output=32768,
    ),
    ModelEntry(
        alias="qwen3.6-27b-q4",
        repo="bartowski/Qwen_Qwen3.6-27B-GGUF",
        filename="Qwen_Qwen3.6-27B-Q4_K_M.gguf",
        size="~18 GB",
        description="Qwen3.6 27B dense — Apr 2026, 262K context, multimodal",
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
    """Resolve filename → ModelEntry from a specific list (or KNOWN_MODELS)."""
    if not model_list:
        model_list = KNOWN_MODELS
    return next((m for m in model_list if m.filename == filename), None)


def _resolve(
    target: str,
    *,
    _fallback_list: list[ModelEntry] | None = None,
) -> ModelEntry | None:
    """Resolve alias or filename to a ModelEntry from config, falling back to KNOWN_MODELS.

    _fallback_list is an internal parameter for testing — when provided,
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
    # Dual catalog fallback: search KNOWN_MODELS
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
    cfg = load_config()
    resolved = _by_filename(dl_filename, cfg.models.entries if cfg.models.has_catalog else None)
    if not resolved and cfg.models.has_catalog is False:
        resolved = _by_filename(dl_filename, KNOWN_MODELS)
    switch_target = resolved.alias if resolved else dl_filename
    console.print(f"Switch to it with: [bold]uv run llm model switch {switch_target}[/bold]")


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


@app.command("init-catalog")
def init_catalog() -> None:
    """Copy built-in models into [[models.list]] in config.toml."""
    config_path = find_config()
    text = config_path.read_text()

    # Build catalog block from KNOWN_MODELS
    lines: list[str] = []
    for m in KNOWN_MODELS:
        lines.append("[[models.list]]")
        lines.append(f'alias = "{m.alias}"')
        lines.append(f'repo  = "{m.repo}"')
        lines.append(f'filename = "{m.filename}"')
        lines.append(f'size = "{m.size}"')
        lines.append(f'description = "{m.description}"')
        lines.append(f'max_output = {m.max_output}')
        lines.append("cost.input = 0.0")
        lines.append("cost.output = 0.0")
        lines.append("")  # blank separator

    catalog_block = "\n".join(lines)

    # Find insertion point: after active = "..." and before # ── PROXY
    marker = "# ── PROXY"
    insert_pos = text.find(marker)
    if insert_pos == -1:
        # Fallback: append before [proxy]
        proxy_marker = "[proxy]"
        insert_pos = text.find(proxy_marker)
    if insert_pos == -1:
        # Final fallback: append to end of file
        text = text.rstrip("\n") + "\n"
        insert_pos = len(text)
    else:
        insert_pos = text.rfind("\n", 0, insert_pos) + 1  # move past newline

    new_text = text[:insert_pos] + "\n" + catalog_block + text[insert_pos:]
    config_path.write_text(new_text)

    console.print(
        f"[green]✓[/green] Catalog added to [bold]{config_path}[/bold] "
        f"({len(KNOWN_MODELS)} models).\n"
        "Edit entries as needed, then run: [bold]uv run llm model list[/bold]"
    )


# ── show command ────────────────────────────────────────────────────────────


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


@app.command("catalog")
def catalog() -> None:
    """Print the model catalog from config as a table."""
    cfg = load_config()

    # Use config catalog if available, otherwise fall back to KNOWN_MODELS
    model_list = cfg.models.entries if cfg.models.has_catalog else KNOWN_MODELS

    table = Table(title=f"Model catalog ({len(model_list)} models)", show_header=True)
    table.add_column("", width=2)
    table.add_column("Alias", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Max Output", justify="right")
    table.add_column("Description", style="white", max_width=60)

    for m in model_list:
        active_marker = "▶" if m.alias == cfg.models.active else ""
        table.add_row(
            active_marker,
            m.alias,
            m.size or "–",
            str(m.max_output),
            m.description,
        )

    console.print(table)
    console.print("\n[dim]▶ = active   Use [bold]llm model switch <alias>[/bold] to change[/dim]")


@app.command("cost")
def show_cost(
    target: Annotated[str | None, typer.Argument(help="Alias or filename to show cost for."), None] = None,
) -> None:
    """Show cost information for one or all models."""
    cfg = load_config()

    # Use config catalog if available, otherwise fall back to KNOWN_MODELS
    model_list = cfg.models.entries if cfg.models.has_catalog else KNOWN_MODELS

    if target:
        entry = _resolve(target)
        if not entry:
            console.print(f"[red]Model not found:[/red] {target}")
            raise typer.Exit(1)
        console.print(f"\n[bold]{entry.alias}[/bold]")
        console.print(f"  Input:   ${entry.cost.input:.4f} / token")
        console.print(f"  Output:  ${entry.cost.output:.4f} / token")
        console.print(f"  Cache W: ${entry.cost.cache_write:.4f} / token")
        console.print(f"  Cache R: ${entry.cost.cache_read:.4f} / token")
        if entry.cost.is_zero():
            console.print("  [dim](all costs are zero — model is free)[/dim]")
    else:
        # Show all models with non-zero costs
        table = Table(title="Model Costs ($ per token)", show_header=True)
        table.add_column("Alias", style="cyan")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cache W", justify="right")
        table.add_column("Cache R", justify="right")
        any_nonzero = False
        for m in model_list:
            if m.cost.is_zero():
                continue
            any_nonzero = True
            table.add_row(
                m.alias,
                f"${m.cost.input:.4f}",
                f"${m.cost.output:.4f}",
                f"${m.cost.cache_write:.4f}",
                f"${m.cost.cache_read:.4f}",
            )
        if any_nonzero:
            console.print(table)
        else:
            console.print("[dim]All models have zero cost (free local models).[/dim]")
