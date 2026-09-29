"""Tests for the shared formatting helpers."""

from __future__ import annotations

import pytest

from llm.core import fmt


class TestFileSize:
    def test_reports_gigabytes(self, tmp_path):
        f = tmp_path / "m.gguf"
        with f.open("wb") as fh:
            fh.truncate(1_073_741_824)  # sparse: no need to write a real GiB
        assert fmt.file_size(f) == "1.0 GB"

    def test_rounds_to_one_decimal(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x" * 1024)
        assert fmt.file_size(f) == "0.0 GB"


class TestMib:
    def test_missing_value_renders_as_a_dash(self):
        assert fmt.mib("", "kb") == "-"

    def test_kilobytes_are_converted(self):
        assert fmt.mib("2048", "kb") == "2 MiB"

    def test_values_already_in_mib_are_not_converted(self):
        assert fmt.mib("2048", "mb") == "2,048 MiB"

    def test_thousands_are_separated(self):
        assert fmt.mib("10485760", "kb") == "10,240 MiB"

    def test_unparseable_values_pass_through(self):
        assert fmt.mib("unknown", "kb") == "unknown"


class TestDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (45, "45s"),
            (59, "59s"),
            (60, "1m 0s"),
            (125, "2m 5s"),
            (3599, "59m 59s"),
            (3600, "1h 0m"),
            (3661, "1h 1m"),
            (86399, "23h 59m"),
            (86400, "1d 0h"),
            (90061, "1d 1h"),
            (2592000, "30d 0h"),
        ],
    )
    def test_uses_the_two_largest_units(self, seconds, expected):
        assert fmt.duration(seconds) == expected
