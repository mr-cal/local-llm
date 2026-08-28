"""AMD GPU memory probing via the amdgpu sysfs interface.

On the Ryzen AI 300 series the iGPU is a Vulkan device with no dedicated
VRAM: model weights live in system RAM mapped through GTT, plus a small
BIOS UMA carve-out reported as VRAM. Both pools are probed here so callers
can distinguish "GPU memory" from plain system RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SYS_DRM = Path("/sys/class/drm")


@dataclass(frozen=True)
class GpuMemory:
    """VRAM and GTT usage in MiB for the first AMD GPU found."""

    vram_used_mb: float
    vram_total_mb: float
    gtt_used_mb: float
    gtt_total_mb: float


def _read_mb(path: Path) -> float:
    """Read an amdgpu ``mem_info_*`` counter, in bytes, as MiB."""
    return int(path.read_text().strip()) / (1024 * 1024)


def _read_mb_opt(path: Path) -> float:
    """Read ``path`` as MiB, or 0.0 when it does not exist."""
    return _read_mb(path) if path.exists() else 0.0


def read_gpu_memory() -> GpuMemory | None:
    """Return VRAM and GTT usage for the first AMD GPU, or None if unavailable."""
    for card in sorted(_SYS_DRM.glob("card*")):
        dev = card / "device"
        vram_used_f = dev / "mem_info_vram_used"
        if not vram_used_f.exists():
            continue
        try:
            return GpuMemory(
                vram_used_mb=_read_mb(vram_used_f),
                vram_total_mb=_read_mb_opt(dev / "mem_info_vram_total"),
                gtt_used_mb=_read_mb_opt(dev / "mem_info_gtt_used"),
                gtt_total_mb=_read_mb_opt(dev / "mem_info_gtt_total"),
            )
        except (ValueError, OSError):
            continue
    return None


def gpu_used_mb() -> float | None:
    """Return the larger of VRAM/GTT used, in MiB, or None if unavailable."""
    mem = read_gpu_memory()
    if mem is None:
        return None
    return max(mem.vram_used_mb, mem.gtt_used_mb)


def gpu_memory_status() -> tuple[float, float] | None:
    """Return (used_mb, total_mb) of the more-utilized pool (VRAM or GTT).

    Mirrors the historical "report whichever pool is larger" behavior so
    callers can compute GPU headroom against a single pool.
    """
    mem = read_gpu_memory()
    if mem is None:
        return None
    if mem.gtt_used_mb > mem.vram_used_mb:
        return mem.gtt_used_mb, mem.gtt_total_mb
    return mem.vram_used_mb, mem.vram_total_mb
