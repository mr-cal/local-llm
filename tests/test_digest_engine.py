"""Unit tests for the slim prompt digest engine."""

from __future__ import annotations

import datetime as dt

from llm.digest.engine import DigestEngine
from llm.digest.schedule import Slot
from llm.settings.models import Settings


def test_build_slim_prompt():
    settings = Settings()
    engine = DigestEngine(settings)

    items = [
        {
            "source": "discourse",
            "title": "snapd-info discussion",
            "author": "cal",
            "content": "Proposal for snapd-info key.",
        },
        {
            "source": "github",
            "title": "PR #1234",
            "author": "alice",
            "content": "Add feature X.",
        },
    ]
    thread_updates = [
        {
            "source": "mattermost",
            "title": "Dev channel",
            "delta_replies": 4,
        }
    ]

    prompt = engine._build_slim_prompt(
        slot=Slot.AFTERNOON,
        today_str="2025-05-12",
        items=items,
        thread_updates=thread_updates,
    )

    assert "Afternoon Digest for 2025-05-12" in prompt
    assert "snapd-info discussion" in prompt
    assert "Dev channel" in prompt
    assert "4 new replies" in prompt


def test_run_digest_end_to_end(monkeypatch, tmp_path):
    settings = Settings()
    settings.digest.cache_db_path = str(tmp_path / "cache.db")
    settings.digest.digest_output_dir = str(tmp_path / "digests")

    engine = DigestEngine(settings)

    # Mock source ingestors to return controlled items
    monkeypatch.setattr("llm.digest.engine.ingest_discourse", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "llm.digest.engine.ingest_github",
        lambda *args, **kwargs: [
            {"source": "github", "title": "Issue #42", "author": "bob", "content": "Fix bug"}
        ],
    )
    monkeypatch.setattr("llm.digest.engine.ingest_mattermost", lambda *args, **kwargs: [])
    monkeypatch.setattr("llm.digest.engine.ingest_matrix", lambda *args, **kwargs: [])
    monkeypatch.setattr("llm.digest.engine.ingest_google", lambda *args, **kwargs: [])
    monkeypatch.setattr("llm.digest.engine.ingest_jira", lambda *args, **kwargs: [])

    # Mock _call_llm
    monkeypatch.setattr(
        engine,
        "_call_llm",
        lambda base_url, prompt: "# Morning Digest 2025-05-12\n\n- Issue #42 fixed by bob.",
    )

    now = dt.datetime(2025, 5, 12, 5, 30, 0, tzinfo=dt.UTC)
    summary = engine.run_digest(slot=Slot.MORNING, now=now)

    assert "# Morning Digest 2025-05-12" in summary
    assert "Issue #42 fixed by bob" in summary

    # Verify cached
    cached = engine.cache.get_digest("2025-05-12", "morning")
    assert cached == summary

    # Verify written to disk
    out_file = tmp_path / "digests" / "2025-05-12-morning.md"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == summary
