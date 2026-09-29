"""Human-readable formatting of sizes reported by the kernel and by models."""

from __future__ import annotations

from pathlib import Path

_BYTES_PER_GIB = 1_073_741_824


def file_size(path: Path) -> str:
    """Size of *path* in GB, as shown in model listings."""
    return f"{path.stat().st_size / _BYTES_PER_GIB:.1f} GB"


def mib(value: str, unit: str) -> str:
    """Format a /proc value carrying its own unit as MiB, or '-' when missing."""
    if value == "":
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    if unit == "kb":
        number /= 1024
    return f"{number:,.0f} MiB"


def duration(seconds: int) -> str:
    """Format a process uptime as the largest sensible unit."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
