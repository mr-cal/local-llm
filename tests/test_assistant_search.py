"""Unit tests for the multi-step search assistant."""

from __future__ import annotations

import pytest

from llm.assistant.search import AssistantSearchRunner
from llm.settings.models import Settings


@pytest.mark.anyio
async def test_search_runner_cache_tool(tmp_path):
    settings = Settings()
    settings.digest.cache_db_path = str(tmp_path / "cache.db")
    runner = AssistantSearchRunner(settings)

    runner.cache.upsert_item(
        id="test:1",
        source="github",
        item_id="1",
        title="Test item",
        content="Looking for snapd-info key",
    )

    results = await runner.execute_tool("search_cache", {"query": "snapd-info"})
    assert len(results) == 1
    assert results[0]["id"] == "test:1"


@pytest.mark.anyio
async def test_search_runner_code_tool(tmp_path, monkeypatch):
    settings = Settings()
    settings.digest.craft_dir = str(tmp_path)
    runner = AssistantSearchRunner(settings)

    monkeypatch.setattr(
        runner.craft,
        "search_code",
        lambda query, max_results=10: [{"file": "snapcraft/info.py", "line": 10, "content": "found"}],
    )

    results = await runner.execute_tool("search_code", {"query": "snapd-info"})
    assert len(results) == 1
    assert results[0]["file"] == "snapcraft/info.py"


@pytest.mark.anyio
async def test_search_runner_unknown_tool():
    settings = Settings()
    runner = AssistantSearchRunner(settings)
    res = await runner.execute_tool("nonexistent_tool", {})
    assert "Unknown tool" in res["error"]
