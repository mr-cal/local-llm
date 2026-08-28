"""Tests for the shared AMD GPU memory probe."""

from __future__ import annotations

import pytest

from llm import gpu


def _make_card(tmp_path, vram_used, vram_total, gtt_used, gtt_total):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "mem_info_vram_used").write_text(str(vram_used))
    (card / "mem_info_vram_total").write_text(str(vram_total))
    (card / "mem_info_gtt_used").write_text(str(gtt_used))
    (card / "mem_info_gtt_total").write_text(str(gtt_total))
    return card


class TestReadGpuMemory:
    def test_returns_all_pools(self, tmp_path, monkeypatch):
        _make_card(
            tmp_path,
            vram_used=2 * 1024 * 1024,
            vram_total=8 * 1024 * 1024,
            gtt_used=5 * 1024 * 1024,
            gtt_total=6 * 1024 * 1024,
        )
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        mem = gpu.read_gpu_memory()
        assert mem is not None
        assert mem.vram_used_mb == pytest.approx(2.0)
        assert mem.vram_total_mb == pytest.approx(8.0)
        assert mem.gtt_used_mb == pytest.approx(5.0)
        assert mem.gtt_total_mb == pytest.approx(6.0)

    def test_missing_pool_files_default_to_zero(self, tmp_path, monkeypatch):
        card = tmp_path / "card0" / "device"
        card.mkdir(parents=True)
        (card / "mem_info_vram_used").write_text(str(2 * 1024 * 1024))
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        mem = gpu.read_gpu_memory()
        assert mem is not None
        assert mem.vram_used_mb == pytest.approx(2.0)
        assert mem.vram_total_mb == 0.0
        assert mem.gtt_used_mb == 0.0
        assert mem.gtt_total_mb == 0.0

    def test_none_when_no_card(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        assert gpu.read_gpu_memory() is None

    def test_none_when_non_amd_card(self, tmp_path, monkeypatch):
        # A card without mem_info_vram_used (e.g. virtio GPU) is skipped.
        (tmp_path / "card0" / "device").mkdir(parents=True)
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        assert gpu.read_gpu_memory() is None


class TestGpuHelpers:
    def test_gpu_used_mb_returns_larger_pool(self, tmp_path, monkeypatch):
        _make_card(
            tmp_path,
            vram_used=2 * 1024 * 1024,
            vram_total=8 * 1024 * 1024,
            gtt_used=5 * 1024 * 1024,
            gtt_total=6 * 1024 * 1024,
        )
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        assert gpu.gpu_used_mb() == pytest.approx(5.0)

    def test_gpu_used_mb_none_when_no_card(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        assert gpu.gpu_used_mb() is None

    def test_gpu_memory_status_vram_when_vram_larger(self, tmp_path, monkeypatch):
        _make_card(
            tmp_path,
            vram_used=2 * 1024 * 1024,
            vram_total=8 * 1024 * 1024,
            gtt_used=0,
            gtt_total=0,
        )
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        status = gpu.gpu_memory_status()
        assert status is not None
        used, total = status
        assert used == pytest.approx(2.0)
        assert total == pytest.approx(8.0)

    def test_gpu_memory_status_gtt_when_gtt_larger(self, tmp_path, monkeypatch):
        _make_card(
            tmp_path,
            vram_used=1 * 1024 * 1024,
            vram_total=8 * 1024 * 1024,
            gtt_used=5 * 1024 * 1024,
            gtt_total=6 * 1024 * 1024,
        )
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        status = gpu.gpu_memory_status()
        assert status is not None
        used, total = status
        assert used == pytest.approx(5.0)
        assert total == pytest.approx(6.0)

    def test_gpu_memory_status_none_when_no_card(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu, "_SYS_DRM", tmp_path)
        assert gpu.gpu_memory_status() is None
