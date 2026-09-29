"""Tests for the memory monitor sampler, pruning, and daemon loop."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta

from llm import monitor
from llm.gpu import GpuMemory


def _iso(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


class TestSample:
    def test_sample_builds_row(self, monkeypatch):
        monkeypatch.setattr(
            monitor,
            "_read_meminfo",
            lambda: {"MemTotal": 1000, "MemAvailable": 500, "MemFree": 200, "SwapTotal": 100, "SwapFree": 50},
        )
        monkeypatch.setattr(
            monitor,
            "_read_proc_status",
            lambda pid: {"rss_kb": 123, "vms_kb": 456, "vm_hwm_kb": 789, "vm_swap_kb": 0},
        )
        monkeypatch.setattr(monitor, "read_gpu_memory", lambda: GpuMemory(2.0, 8.0, 5.0, 6.0))

        row = monitor.sample(1234)

        assert row["event"] == "sample"
        assert row["pid"] == 1234
        assert row["mem_total_kb"] == 1000
        assert row["mem_avail_kb"] == 500
        assert row["rss_kb"] == 123
        assert row["gpu_vram_used_mb"] == 2.0
        assert row["gpu_gtt_used_mb"] == 5.0

    def test_sample_exited_has_empty_optional_fields(self, monkeypatch):
        monkeypatch.setattr(monitor, "_read_meminfo", lambda: {})
        monkeypatch.setattr(monitor, "_read_proc_status", lambda pid: {})
        monkeypatch.setattr(monitor, "read_gpu_memory", lambda: None)

        row = monitor.sample(1234, event="exited")

        assert row["event"] == "exited"
        assert row["mem_avail_kb"] == ""
        assert row["rss_kb"] == ""
        assert row["gpu_gtt_used_mb"] == ""


class TestPruneOldRows:
    def _write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=monitor.HEADERS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def test_removes_old_keeps_recent(self, tmp_path):
        path = tmp_path / "memory.csv"
        old = {h: "" for h in monitor.HEADERS}
        old["timestamp"] = _iso(100)
        recent = {h: "" for h in monitor.HEADERS}
        recent["timestamp"] = _iso(1)
        self._write_csv(path, [old, recent])

        removed = monitor.prune_old_rows(path, retention_days=90)

        assert removed == 1
        rows = monitor.read_recent_rows(path, 100)
        assert len(rows) == 1
        assert rows[0]["timestamp"] == recent["timestamp"]

    def test_keeps_all_when_within_retention(self, tmp_path):
        path = tmp_path / "memory.csv"
        recent = {h: "" for h in monitor.HEADERS}
        recent["timestamp"] = _iso(1)
        self._write_csv(path, [recent])
        assert monitor.prune_old_rows(path, retention_days=90) == 0

    def test_returns_zero_when_missing(self, tmp_path):
        assert monitor.prune_old_rows(tmp_path / "nope.csv", retention_days=90) == 0


class TestRunMonitor:
    def test_writes_samples_then_exit_row(self, tmp_path, monkeypatch):
        path = tmp_path / "logs" / "memory.csv"
        alive = iter([True, True, False])

        monkeypatch.setattr(monitor, "_pid_alive", lambda pid: next(alive))
        monkeypatch.setattr(monitor.time, "sleep", lambda *a: None)
        monkeypatch.setattr(monitor.time, "monotonic", lambda: 0.0)
        monkeypatch.setattr(monitor, "_read_meminfo", lambda: {"MemAvailable": 500})
        monkeypatch.setattr(monitor, "_read_proc_status", lambda pid: {"rss_kb": 100})
        monkeypatch.setattr(monitor, "read_gpu_memory", lambda: None)

        monitor.run_monitor(1234, interval=30, retention_days=90, csv_path=path)

        rows = monitor.read_recent_rows(path, 100)
        assert len(rows) == 3  # two samples + one exit row
        assert rows[0]["event"] == "sample"
        assert rows[1]["event"] == "sample"
        assert rows[2]["event"] == "exited"
