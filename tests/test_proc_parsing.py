"""Tests for the /proc parsers.

These read text the kernel formats loosely - a process name may contain
spaces and parentheses, fields may be missing, and /proc entries vanish when
a process exits - so each parser is exercised against the awkward shapes
rather than only the happy path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm import benchmark, monitor
from llm.server import _process_uptime_seconds

MEMINFO = """MemTotal:       32772836 kB
MemFree:         1234567 kB
MemAvailable:   20971520 kB
Buffers:          123456 kB
HugePages_Total:       0
"""

PROC_STATUS = """Name:\tllama-server
State:\tS (sleeping)
VmSize:\t 1234567 kB
VmRSS:\t  987654 kB
VmHWM:\t  999999 kB
VmSwap:\t       0 kB
Threads:\t12
"""


def _fake_read_text(mapping: dict[str, str]):
    def _read(self, *args, **kwargs):
        try:
            return mapping[str(self)]
        except KeyError:
            raise FileNotFoundError(str(self)) from None

    return _read


class TestReadMeminfo:
    def test_parses_kb_values(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({"/proc/meminfo": MEMINFO}))
        info = monitor._read_meminfo()
        assert info["MemAvailable"] == 20971520
        assert info["MemTotal"] == 32772836

    def test_parses_entries_without_a_kB_suffix(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({"/proc/meminfo": MEMINFO}))
        assert monitor._read_meminfo()["HugePages_Total"] == 0

    def test_returns_empty_when_unreadable(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({}))
        assert monitor._read_meminfo() == {}

    def test_skips_unparseable_lines_instead_of_failing(self, monkeypatch):
        text = "MemFree: not-a-number kB\nMemAvailable:   100 kB\n"
        monkeypatch.setattr(Path, "read_text", _fake_read_text({"/proc/meminfo": text}))
        info = monitor._read_meminfo()
        assert info == {"MemAvailable": 100}


class TestReadProcStatus:
    def test_extracts_only_the_memory_keys(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({"/proc/42/status": PROC_STATUS}))
        result = monitor._read_proc_status(42)
        assert set(result) == {"rss_kb", "vms_kb", "vm_hwm_kb", "vm_swap_kb"}
        assert result["rss_kb"] == 987654
        assert result["vm_swap_kb"] == 0

    def test_returns_empty_for_a_dead_process(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({}))
        assert monitor._read_proc_status(42) == {}


class TestAvailableMemoryMb:
    def test_converts_kb_to_mib(self, tmp_path, monkeypatch):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(MEMINFO)
        real_open = open
        monkeypatch.setattr(
            benchmark,
            "open",
            lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
            raising=False,
        )
        assert benchmark._available_memory_mb() == pytest.approx(20971520 / 1024)

    def test_returns_none_when_meminfo_is_unreadable(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("no /proc")

        monkeypatch.setattr(benchmark, "open", _boom, raising=False)
        assert benchmark._available_memory_mb() is None


class TestProcessUptimeSeconds:
    def _stat(self, starttime: int, comm: str = "llama-server") -> str:
        # After the comm field, /proc/<pid>/stat continues with state (field 3);
        # starttime is field 22, so it lands at index 19 of what remains.
        tail = " ".join(["0"] * 18 + [str(starttime)])
        return f"1234 ({comm}) S {tail}\n"

    def test_computes_uptime_from_boot_time(self, monkeypatch):
        monkeypatch.setattr(
            Path,
            "read_text",
            _fake_read_text({"/proc/1234/stat": self._stat(6000), "/proc/uptime": "1000.50 900.00\n"}),
        )
        monkeypatch.setattr("llm.server.os.sysconf", lambda name: 100)
        # starttime 6000 ticks / 100 Hz = 60s after boot; 1000.5 - 60 = 940.5
        assert _process_uptime_seconds(1234) == 940

    def test_handles_a_process_name_containing_spaces_and_parens(self, monkeypatch):
        monkeypatch.setattr(
            Path,
            "read_text",
            _fake_read_text(
                {"/proc/1234/stat": self._stat(6000, "lla (ma) server"), "/proc/uptime": "1000.50 900.00\n"}
            ),
        )
        monkeypatch.setattr("llm.server.os.sysconf", lambda name: 100)
        assert _process_uptime_seconds(1234) == 940

    def test_never_returns_a_negative_uptime(self, monkeypatch):
        monkeypatch.setattr(
            Path,
            "read_text",
            _fake_read_text({"/proc/1234/stat": self._stat(999999), "/proc/uptime": "10.0 5.0\n"}),
        )
        monkeypatch.setattr("llm.server.os.sysconf", lambda name: 100)
        assert _process_uptime_seconds(1234) == 0

    def test_returns_none_for_a_dead_process(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", _fake_read_text({"/proc/uptime": "10.0 5.0\n"}))
        assert _process_uptime_seconds(1234) is None

    def test_returns_none_for_a_truncated_stat_line(self, monkeypatch):
        monkeypatch.setattr(
            Path,
            "read_text",
            _fake_read_text({"/proc/1234/stat": "1234 (x) S 1 2 3\n", "/proc/uptime": "10.0 5.0\n"}),
        )
        assert _process_uptime_seconds(1234) is None

    def test_falls_back_to_100_hz_when_sysconf_fails(self, monkeypatch):
        def _boom(name):
            raise OSError("no SC_CLK_TCK")

        monkeypatch.setattr(
            Path,
            "read_text",
            _fake_read_text({"/proc/1234/stat": self._stat(6000), "/proc/uptime": "1000.50 900.00\n"}),
        )
        monkeypatch.setattr("llm.server.os.sysconf", _boom)
        assert _process_uptime_seconds(1234) == 940
