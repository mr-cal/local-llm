"""Unit tests for LLM streaming, token counting, and calibrated TPS."""

from __future__ import annotations

import pytest

from llm.tui import llm_stream
from llm.tui.llm_stream import (
    DEFAULT_PROMPT_TPS,
    count_tokens_async,
    load_calibrated_tps,
    update_calibrated_tps,
)


def test_calibrated_tps_fallback(monkeypatch, tmp_path):
    fake_perf = tmp_path / "perf.json"
    monkeypatch.setattr(llm_stream, "PERF_FILE", fake_perf)

    # When file does not exist, returns DEFAULT_PROMPT_TPS (40.0)
    assert load_calibrated_tps() == DEFAULT_PROMPT_TPS


def test_update_calibrated_tps(monkeypatch, tmp_path):
    fake_perf = tmp_path / "perf.json"
    monkeypatch.setattr(llm_stream, "PERF_FILE", fake_perf)

    # Initial TPS is 40.0.
    # Suppose 100 tokens evaluated in 2.0 seconds -> current_tps = 50.0.
    # Expected: 0.7 * 40.0 + 0.3 * 50.0 = 28.0 + 15.0 = 43.0.
    new_tps = update_calibrated_tps(tokens=100, eval_seconds=2.0)
    assert pytest.approx(new_tps, 0.01) == 43.0
    assert pytest.approx(load_calibrated_tps(), 0.01) == 43.0

    # Invalid values should not alter tps
    unchanged = update_calibrated_tps(tokens=0, eval_seconds=0.0)
    assert pytest.approx(unchanged, 0.01) == 43.0


@pytest.mark.anyio
async def test_count_tokens_fallback(monkeypatch):
    # Testing when server is unreachable or offline
    text = "Hello world! This is a test."
    tokens = await count_tokens_async("http://127.0.0.1:9999", text)
    # 28 chars // 4 = 7
    assert tokens == len(text) // 4
