"""Benchmarking: measure prompt-processing and token-generation throughput."""

from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from llm.config import load_config

app = typer.Typer(help="Benchmark inference speed.")
console = Console()

HISTORY_FILE = Path("logs/benchmark-history.csv")
HISTORY_HEADERS = ["timestamp", "model", "backend", "pp_tps", "tg_tps", "ctx", "n_tokens", "n_gpu_layers"]

_DEFAULT_PROMPT = (
    "Write a Python function that takes a list of integers and returns a new list "
    "containing only the prime numbers. Include type hints and a docstring."
)


def _ensure_history_file() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with HISTORY_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_HEADERS).writeheader()


def _append_result(row: dict[str, str | int | float]) -> None:
    _ensure_history_file()
    with HISTORY_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_HEADERS)
        writer.writerow(row)


@app.command("run")
def run(
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt text to send.")] = _DEFAULT_PROMPT,
    n_tokens: Annotated[int, typer.Option("--n-tokens", "-n", help="Max tokens to generate.")] = 200,
    raw: Annotated[bool, typer.Option("--raw", help="Also run llama-bench binary (raw throughput).")] = False,
) -> None:
    """Run an end-to-end API benchmark and record results."""
    cfg = load_config()

    console.print(f"Model      : [bold]{cfg.models.active}[/bold]")
    console.print(f"Endpoint   : {cfg.internal_url}")
    console.print(f"Max tokens : {n_tokens}")
    console.print()

    # ── API benchmark ──────────────────────────────────────────────────────
    payload = {
        "model": cfg.models.active,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens,
        "stream": False,
    }

    console.print("Sending request...", end=" ")
    t_start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{cfg.internal_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        console.print("[red]Connection refused[/red] — is the server running?")
        console.print("  Start it: [bold]uv run llm server start[/bold]")
        raise typer.Exit(1) from None
    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP {e.response.status_code}[/red]: {e.response.text[:200]}")
        raise typer.Exit(1) from None

    elapsed = time.perf_counter() - t_start
    data = resp.json()

    usage = data.get("usage", {})
    prompt_tokens: int = usage.get("prompt_tokens", 0)
    completion_tokens: int = usage.get("completion_tokens", 0)

    # llama.cpp reports timings in the response when available
    timings = data.get("timings", {})
    pp_tps_raw: float = timings.get("prompt_per_second", 0.0)
    tg_tps_raw: float = timings.get("predicted_per_second", 0.0)

    # Fallback: estimate tg_tps from wall-clock if timings not available
    tg_tps = tg_tps_raw if tg_tps_raw > 0 else (completion_tokens / elapsed if elapsed > 0 else 0)
    pp_tps = pp_tps_raw

    # ── Display results ────────────────────────────────────────────────────
    console.print("[green]done[/green]")
    console.print()

    t = Table(title="API Benchmark Results", show_header=True)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="bold white", justify="right")

    t.add_row("Wall-clock time", f"{elapsed:.2f}s")
    t.add_row("Prompt tokens", str(prompt_tokens))
    t.add_row("Completion tokens", str(completion_tokens))
    t.add_row("Prompt processing", f"{pp_tps:.1f} tok/s" if pp_tps > 0 else "n/a (use --raw)")
    t.add_row("Token generation", f"{tg_tps:.1f} tok/s")
    t.add_row("GPU layers offloaded", str(cfg.server.n_gpu_layers))

    console.print(t)

    # Show generated text
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if reply:
        console.print("\n[dim]── Generated output ──[/dim]")
        console.print(reply[:500] + ("..." if len(reply) > 500 else ""))

    # ── Record to CSV ──────────────────────────────────────────────────────
    _append_result(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": cfg.models.active,
            "backend": "llama-server",
            "pp_tps": f"{pp_tps:.1f}",
            "tg_tps": f"{tg_tps:.1f}",
            "ctx": cfg.server.n_ctx,
            "n_tokens": completion_tokens,
            "n_gpu_layers": cfg.server.n_gpu_layers,
        }
    )
    console.print(f"\n[dim]Result saved to {HISTORY_FILE}[/dim]")

    # ── Raw llama-bench (optional) ──────────────────────────────────────────
    if raw:
        _run_llama_bench(cfg)


def _run_llama_bench(cfg: object) -> None:
    """Run llama-bench for raw pp/tg throughput without HTTP overhead."""
    from llm.config import Settings

    assert isinstance(cfg, Settings)
    bench_bin = Path(cfg.server.llama_server_bin).parent / "llama-bench"
    if not bench_bin.exists():
        # Try PATH
        result = subprocess.run(["which", "llama-bench"], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[yellow]llama-bench not found — skipping raw benchmark.[/yellow]")
            console.print("  It is built alongside llama-server in llama.cpp/build/bin/")
            return
        bench_bin = Path(result.stdout.strip())

    console.print("\n[bold]Running llama-bench (raw throughput)...[/bold]")
    cmd = [
        str(bench_bin),
        "-m",
        str(cfg.model_path),
        "-ngl",
        str(cfg.server.n_gpu_layers),
        "-t",
        str(cfg.server.n_threads),
    ]
    subprocess.run(cmd, check=False)


@app.command("history")
def history(
    last: Annotated[int, typer.Option("-n", help="Show last N results.")] = 20,
) -> None:
    """Display benchmark history from logs/benchmark-history.csv."""
    if not HISTORY_FILE.exists():
        console.print("[yellow]No benchmark history found.[/yellow]")
        console.print("Run [bold]uv run llm benchmark run[/bold] to create one.")
        raise typer.Exit()

    with HISTORY_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows = rows[-last:]
    if not rows:
        console.print("[yellow]No records in history.[/yellow]")
        return

    t = Table(title=f"Benchmark History (last {len(rows)})", show_header=True)
    for col in HISTORY_HEADERS:
        t.add_column(col.replace("_", " ").title(), style="cyan" if col == "model" else "white")

    for row in rows:
        t.add_row(*[row.get(h, "") for h in HISTORY_HEADERS])

    console.print(t)
