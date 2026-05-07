"""Benchmarking: measure prompt-processing and token-generation throughput."""

from __future__ import annotations

import csv
import io
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from llm.config import find_config, load_config

app = typer.Typer(help="Benchmark inference speed.")
console = Console()

HISTORY_FILE = Path("logs/benchmark-history.csv")
HISTORY_HEADERS = ["timestamp", "model", "backend", "pp_tps", "tg_tps", "ctx", "n_tokens", "n_gpu_layers"]

_DEFAULT_PROMPT = (
    "Write a Python function that takes a list of integers and returns a new list "
    "containing only the prime numbers. Include type hints and a docstring."
)


# ── llama-bench helpers ───────────────────────────────────────────────────────


def _find_bench_bin() -> Path | None:
    result = subprocess.run(["which", "llama-bench"], capture_output=True, text=True)
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def _parse_bench_csv(text: str) -> list[dict[str, str]]:
    """Parse llama-bench CSV output, stripping ggml/llama log lines from stdout."""
    lines = [ln for ln in text.splitlines() if ln and not re.match(r"(ggml|llama|load_|main:)", ln)]
    if not lines:
        return []
    try:
        return list(csv.DictReader(io.StringIO("\n".join(lines))))
    except Exception:
        return []


def _bench_tps(
    rows: list[dict[str, str]],
    n_gpu_layers: int | None = None,
    flash_attn: int | None = None,
    type_k: str | None = None,
) -> tuple[float, float]:
    """Return (pp_tps, tg_tps) averaged over rows matching the given filters."""
    pp, tg = [], []
    for row in rows:
        if n_gpu_layers is not None and row.get("n_gpu_layers") != str(n_gpu_layers):
            continue
        if flash_attn is not None and row.get("flash_attn") != str(flash_attn):
            continue
        if type_k is not None and row.get("type_k") != type_k:
            continue
        try:
            speed = float(row["avg_ts"])  # column name in llama-bench b9047+
        except (KeyError, ValueError):
            continue
        # pp vs tg determined by which token count is non-zero
        try:
            is_pp = int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0
            is_tg = int(row.get("n_gen", 0)) > 0 and int(row.get("n_prompt", 0)) == 0
        except ValueError:
            continue
        if is_pp:
            pp.append(speed)
        elif is_tg:
            tg.append(speed)
    return (sum(pp) / len(pp) if pp else 0.0, sum(tg) / len(tg) if tg else 0.0)


def _run_llama_bench(
    bench_bin: Path,
    model_path: Path,
    n_threads: int,
    ngl_values: list[int],
    flash_attn_values: list[int],
    ctk_values: list[str],
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 2,
) -> list[dict[str, str]]:
    """Run llama-bench with given parameter combinations; return parsed CSV rows."""
    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-t",
        str(n_threads),
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-r",
        str(repetitions),
        "-o",
        "csv",
        "--progress",
        "-ngl",
        ",".join(str(v) for v in ngl_values),
        "-fa",
        ",".join(str(v) for v in flash_attn_values),
        "-ctk",
        ",".join(ctk_values),
    ]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, text=True)
    return _parse_bench_csv(result.stdout)


def _apply_config(n_gpu_layers: int, flash_attn: bool, ctk: str) -> None:
    """Update config.toml with tuned settings, preserving non-tuning extra_args."""
    cfg = load_config()
    config_path = find_config()
    text = config_path.read_text()

    text = re.sub(r"^n_gpu_layers\s*=\s*\d+", f"n_gpu_layers = {n_gpu_layers}", text, flags=re.MULTILINE)

    new_extra = _build_extra_args(flash_attn, ctk, cfg.server.extra_args)
    text = re.sub(r"^extra_args\s*=\s*\[.*?\]", f"extra_args = {new_extra!r}", text, flags=re.MULTILINE)

    config_path.write_text(text)
    console.print(
        f"[green]Config updated:[/green] n_gpu_layers={n_gpu_layers}"
        f"  flash_attn={flash_attn}  cache_type_k={ctk}"
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
        _run_llama_bench_raw(cfg)


def _run_llama_bench_raw(cfg: object) -> None:
    """Run llama-bench for raw pp/tg throughput without HTTP overhead (used by `run --raw`)."""
    from llm.config import Settings

    assert isinstance(cfg, Settings)
    bench_bin = _find_bench_bin()
    if not bench_bin:
        console.print("[yellow]llama-bench not found — skipping raw benchmark.[/yellow]")
        console.print("  Copy it to ~/.local/bin/llama-bench")
        return

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


@app.command("tune")
def tune(
    ngl: Annotated[
        str,
        typer.Option("--ngl", help="Comma-separated n_gpu_layers values. Default covers CPU→full GPU."),
    ] = "0,16,32,48,99",
    repetitions: Annotated[int, typer.Option("-r", help="Repetitions per configuration.")] = 2,
    n_prompt: Annotated[int, typer.Option("--n-prompt", help="Prompt tokens for benchmark.")] = 512,
    n_gen: Annotated[int, typer.Option("--n-gen", help="Generated tokens for benchmark.")] = 128,
    apply: Annotated[
        bool, typer.Option("--apply/--no-apply", help="Write best settings to config.toml.")
    ] = True,
) -> None:
    """3-phase optimization sweep: GPU layers → flash-attn → KV-cache quant.

    Uses llama-bench (no server needed). Optimises for token-generation speed (tg tok/s)
    since that determines interactive latency.
    """
    cfg = load_config()
    bench_bin = _find_bench_bin()
    if not bench_bin:
        console.print("[red]llama-bench not found.[/red] Copy it to ~/.local/bin/llama-bench")
        raise typer.Exit(1)

    model_path = cfg.model_path
    n_threads = cfg.server.n_threads
    ngl_list = [int(x) for x in ngl.split(",")]

    console.print(f"\n[bold cyan]╔══ Benchmark Tune: {cfg.models.active} ══╗[/bold cyan]")
    console.print(f"  bench binary : {bench_bin}")
    console.print(f"  repetitions  : {repetitions}  |  prompt: {n_prompt} tok  gen: {n_gen} tok")
    console.print(f"  ngl sweep    : {ngl_list}")
    console.print()

    # ── Phase 1: GPU layer sweep ──────────────────────────────────────────────
    console.print("[bold]Phase 1/3 — GPU layer sweep[/bold]  (flash-attn=off, ctk=f16)")
    console.print("[dim]This takes the longest — loading the model once per ngl value...[/dim]")

    rows_p1 = _run_llama_bench(
        bench_bin,
        model_path,
        n_threads,
        ngl_values=ngl_list,
        flash_attn_values=[0],
        ctk_values=["f16"],
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
    )

    t_p1 = Table(title="Phase 1: GPU layers", show_header=True)
    t_p1.add_column("ngl", style="cyan", justify="right")
    t_p1.add_column("PP tok/s", justify="right")
    t_p1.add_column("TG tok/s", justify="right", style="bold")

    best_ngl, best_tg = ngl_list[0], 0.0
    for ngl_val in ngl_list:
        pp, tg = _bench_tps(rows_p1, n_gpu_layers=ngl_val)
        marker = ""
        if tg > best_tg:
            best_tg, best_ngl = tg, ngl_val
        t_p1.add_row(str(ngl_val), f"{pp:.1f}", f"{tg:.1f}{marker}")

    # Re-mark best
    console.print(t_p1)
    console.print(f"  → Best ngl: [green]{best_ngl}[/green]  ({best_tg:.1f} tg tok/s)\n")

    # ── Phase 2: Flash-attention test ─────────────────────────────────────────
    console.print(f"[bold]Phase 2/3 — Flash-attention[/bold]  (ngl={best_ngl}, ctk=f16)")

    rows_p2 = _run_llama_bench(
        bench_bin,
        model_path,
        n_threads,
        ngl_values=[best_ngl],
        flash_attn_values=[0, 1],
        ctk_values=["f16"],
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
    )

    t_p2 = Table(title="Phase 2: Flash-attention", show_header=True)
    t_p2.add_column("flash-attn", style="cyan")
    t_p2.add_column("PP tok/s", justify="right")
    t_p2.add_column("TG tok/s", justify="right", style="bold")

    best_fa = 0
    best_tg_p2 = 0.0
    for fa_val in [0, 1]:
        pp, tg = _bench_tps(rows_p2, n_gpu_layers=best_ngl, flash_attn=fa_val)
        if tg > best_tg_p2:
            best_tg_p2, best_fa = tg, fa_val
        t_p2.add_row("on" if fa_val else "off", f"{pp:.1f}", f"{tg:.1f}")

    console.print(t_p2)
    fa_label = "on" if best_fa else "off"
    console.print(f"  → Best flash-attn: [green]{fa_label}[/green]  ({best_tg_p2:.1f} tg tok/s)\n")

    # ── Phase 3: KV-cache quantization ───────────────────────────────────────
    console.print(f"[bold]Phase 3/3 — KV-cache quantization[/bold]  (ngl={best_ngl}, fa={best_fa})")
    console.print("[dim]q8_0 halves KV-cache VRAM — useful for long contexts / OOM prevention[/dim]")

    rows_p3 = _run_llama_bench(
        bench_bin,
        model_path,
        n_threads,
        ngl_values=[best_ngl],
        flash_attn_values=[best_fa],
        ctk_values=["f16", "q8_0"],
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
    )

    t_p3 = Table(title="Phase 3: KV-cache type", show_header=True)
    t_p3.add_column("cache-type-k", style="cyan")
    t_p3.add_column("PP tok/s", justify="right")
    t_p3.add_column("TG tok/s", justify="right", style="bold")
    t_p3.add_column("VRAM note", style="dim")

    best_ctk = "f16"
    best_tg_p3 = 0.0
    for ctk_val, note in [("f16", "full precision"), ("q8_0", "~50% less VRAM")]:
        pp, tg = _bench_tps(rows_p3, n_gpu_layers=best_ngl, flash_attn=best_fa, type_k=ctk_val)
        # Prefer q8_0 if within 10% of f16 — halves KV VRAM, better long-ctx safety
        if ctk_val == "q8_0" and tg >= best_tg_p3 * 0.90:
            best_ctk, best_tg_p3 = ctk_val, tg
        elif ctk_val == "f16":
            best_tg_p3, best_ctk = tg, "f16"
        t_p3.add_row(ctk_val, f"{pp:.1f}", f"{tg:.1f}", note)

    console.print(t_p3)
    console.print(f"  → Best ctk: [green]{best_ctk}[/green]\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline_pp, baseline_tg = _bench_tps(rows_p1, n_gpu_layers=0)
    console.print("[bold cyan]╔══ Tune Summary ══╗[/bold cyan]")
    console.print(f"  Baseline  (ngl=0)     : {baseline_tg:.1f} tg tok/s")
    fa_label = "on" if best_fa else "off"
    console.print(f"  Optimised (ngl={best_ngl}, fa={fa_label}, ctk={best_ctk}) : {best_tg_p3:.1f} tg tok/s")
    if baseline_tg > 0:
        speedup = best_tg_p3 / baseline_tg
        console.print(f"  Speedup                    : [green]{speedup:.1f}×[/green]")
    console.print()

    if apply:
        _apply_config(best_ngl, bool(best_fa), best_ctk)
        console.print("\nRestart the server to apply: [bold]uv run llm server restart[/bold]")
    else:
        console.print(
            f"Run with [bold]--apply[/bold] to write to config.toml, or manually set:\n"
            f"  n_gpu_layers = {best_ngl}\n"
            f"  extra_args   = {_build_extra_args(bool(best_fa), best_ctk, cfg.server.extra_args)}"
        )


def _build_extra_args(flash_attn: bool, ctk: str, current_extra: list[str]) -> list[str]:
    """Rebuild extra_args preserving non-tuning flags, replacing tuning ones."""
    keep = [
        a for a in current_extra if a not in ("--flash-attn", "--cache-type-k", "f16", "q8_0", "q4_0", "q4_1")
    ]
    if flash_attn:
        keep.append("--flash-attn")
    if ctk != "f16":
        keep.extend(["--cache-type-k", ctk])
    return keep


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
