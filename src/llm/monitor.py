"""Memory monitor: sample system, process, and GPU memory while llama-server runs.

Spawned as a detached daemon by ``llm server start``. It appends one CSV row
every *interval* seconds while the server PID is alive, then records a final
``exited`` row the moment the server disappears. Rows older than
*retention_days* are pruned at startup and once per day, keeping the CSV small.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from llm.gpu import read_gpu_memory

MONITOR_CSV = Path("logs/memory-monitor.csv")

HEADERS = [
    "timestamp",
    "event",
    "pid",
    "embed_pid",
    "mem_total_kb",
    "mem_avail_kb",
    "mem_free_kb",
    "swap_total_kb",
    "swap_free_kb",
    "rss_kb",
    "vms_kb",
    "vm_hwm_kb",
    "vm_swap_kb",
    "gpu_vram_used_mb",
    "gpu_vram_total_mb",
    "gpu_gtt_used_mb",
    "gpu_gtt_total_mb",
]

_PRUNE_INTERVAL_S = 24 * 3600
_PROC_KEYS = {"VmRSS": "rss_kb", "VmSize": "vms_kb", "VmHWM": "vm_hwm_kb", "VmSwap": "vm_swap_kb"}


def _now_iso() -> str:
    """UTC timestamp; lexicographically sortable for the retention cutoff."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """Return True if the process exists (signal 0 succeeds)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of kB values."""
    info: dict[str, int] = {}
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return info
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.strip().split()[0] if rest.strip() else "0"
        with contextlib.suppress(ValueError):
            info[key] = int(value)
    return info


def _read_proc_status(pid: int) -> dict[str, int]:
    """Read VmRSS/VmSize/VmHWM/VmSwap (kB) from /proc/<pid>/status."""
    result: dict[str, int] = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in _PROC_KEYS and rest.strip():
            with contextlib.suppress(ValueError):
                result[_PROC_KEYS[key]] = int(rest.strip().split()[0])
    return result


def sample(server_pid: int, embed_pid: int | None, event: str = "sample") -> dict[str, object]:
    """Build one CSV row from current system, process, and GPU state."""
    mem = _read_meminfo()
    proc = _read_proc_status(server_pid)
    gpu = read_gpu_memory()
    return {
        "timestamp": _now_iso(),
        "event": event,
        "pid": server_pid,
        "embed_pid": embed_pid if embed_pid is not None else "",
        "mem_total_kb": mem.get("MemTotal", ""),
        "mem_avail_kb": mem.get("MemAvailable", ""),
        "mem_free_kb": mem.get("MemFree", ""),
        "swap_total_kb": mem.get("SwapTotal", ""),
        "swap_free_kb": mem.get("SwapFree", ""),
        "rss_kb": proc.get("rss_kb", ""),
        "vms_kb": proc.get("vms_kb", ""),
        "vm_hwm_kb": proc.get("vm_hwm_kb", ""),
        "vm_swap_kb": proc.get("vm_swap_kb", ""),
        "gpu_vram_used_mb": round(gpu.vram_used_mb, 1) if gpu else "",
        "gpu_vram_total_mb": round(gpu.vram_total_mb, 1) if gpu else "",
        "gpu_gtt_used_mb": round(gpu.gtt_used_mb, 1) if gpu else "",
        "gpu_gtt_total_mb": round(gpu.gtt_total_mb, 1) if gpu else "",
    }


def _ensure_csv(path: Path) -> None:
    """Create the CSV with its header row if it does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore").writeheader()


def append_row(path: Path, row: dict[str, object]) -> None:
    _ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore").writerow(row)


def prune_old_rows(path: Path, retention_days: int) -> int:
    """Drop rows older than *retention_days*. Returns rows removed."""
    if not path.exists() or retention_days <= 0:
        return 0
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat(timespec="seconds")
    keep: list[dict[str, str]] = []
    removed = 0
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp", "")
                if not ts or ts >= cutoff:
                    keep.append(row)
                else:
                    removed += 1
    except OSError:
        return 0
    if removed == 0:
        return 0
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(keep)
    tmp.replace(path)
    return removed


def read_recent_rows(path: Path, n: int) -> list[dict[str, str]]:
    """Return the last *n* CSV rows, oldest first."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-n:]


def run_monitor(
    server_pid: int,
    embed_pid: int | None,
    interval: int,
    retention_days: int,
    csv_path: Path = MONITOR_CSV,
) -> None:
    """Sample memory every *interval* seconds until *server_pid* exits.

    The final ``exited`` row captures system + GPU state at the moment the
    server disappeared (process counters are already gone by then).
    """
    _ensure_csv(csv_path)
    prune_old_rows(csv_path, retention_days)
    last_prune = time.monotonic()
    while _pid_alive(server_pid):
        append_row(csv_path, sample(server_pid, embed_pid))
        if time.monotonic() - last_prune >= _PRUNE_INTERVAL_S:
            prune_old_rows(csv_path, retention_days)
            last_prune = time.monotonic()
        time.sleep(interval)
    append_row(csv_path, sample(server_pid, embed_pid, event="exited"))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the detached ``python -m llm.monitor`` daemon."""
    parser = argparse.ArgumentParser(prog="llm.monitor", description="Sample memory while llama-server runs.")
    parser.add_argument("server_pid", type=int)
    parser.add_argument("--embed-pid", type=int, default=None)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--retention-days", type=int, default=90)
    parser.add_argument("--csv", type=Path, default=MONITOR_CSV)
    args = parser.parse_args(argv)
    run_monitor(args.server_pid, args.embed_pid, args.interval, args.retention_days, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
