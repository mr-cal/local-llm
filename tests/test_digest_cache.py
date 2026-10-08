"""Unit tests for the SQLite activity cache, FTS5, vector search, and thread deltas."""

from __future__ import annotations

import pytest

from llm.digest.cache import (
    ActivityCache,
    _cosine_similarity,
    _pack_vector,
    _unpack_vector,
)


def test_vector_packing_and_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    packed = _pack_vector(v1)
    unpacked = _unpack_vector(packed)
    assert len(unpacked) == 3
    assert pytest.approx(unpacked[0]) == 1.0

    # Cosine similarity
    assert pytest.approx(_cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(_cosine_similarity(v1, v3)) == 0.0


def test_activity_cache_upsert_and_fts(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = ActivityCache(db_file)

    # Insert items
    cache.upsert_item(
        id="discourse:123",
        source="discourse",
        item_id="123",
        title="snapd-info specification thread",
        content="Discussion regarding the snapd-info key and format in craft yaml",
        author="cal",
        timestamp="2025-05-12T05:00:00Z",
    )
    cache.upsert_item(
        id="mattermost:456",
        source="mattermost",
        item_id="456",
        title="Weekly sync",
        content="Let's discuss rockcraft release testing and QA",
        author="alice",
        timestamp="2025-05-12T05:10:00Z",
    )

    # Retrieve single item
    item = cache.get_item("discourse:123")
    assert item is not None
    assert item["title"] == "snapd-info specification thread"
    assert item["author"] == "cal"

    # Search with FTS5
    results = cache.search_text("snapd-info")
    assert len(results) == 1
    assert results[0]["id"] == "discourse:123"

    # Search with filter on source
    results_mm = cache.search_text("rockcraft", source="mattermost")
    assert len(results_mm) == 1
    assert results_mm[0]["id"] == "mattermost:456"

    # Search non-matching
    assert len(cache.search_text("nonexistent_term")) == 0


def test_vector_and_hybrid_search(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = ActivityCache(db_file)

    cache.upsert_item(
        id="item:1",
        source="github",
        item_id="1",
        title="Feature A",
        content="Implementation details for feature A",
    )
    cache.upsert_item(
        id="item:2",
        source="github",
        item_id="2",
        title="Bug B",
        content="Crash on startup in feature B",
    )

    # Store embeddings
    v_query = [1.0, 0.0, 0.0]
    v_item1 = [0.95, 0.05, 0.0]
    v_item2 = [0.0, 0.1, 0.9]

    cache.upsert_embedding("item:1", v_item1)
    cache.upsert_embedding("item:2", v_item2)

    vec_results = cache.search_vector(v_query, limit=2)
    assert len(vec_results) == 2
    assert vec_results[0]["id"] == "item:1"
    assert vec_results[0]["similarity_score"] > 0.9

    # Hybrid search
    hybrid = cache.hybrid_search("Feature", query_vector=v_query, limit=5)
    assert len(hybrid) > 0
    assert hybrid[0]["id"] == "item:1"


def test_thread_delta_tracking(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = ActivityCache(db_file)

    date = "2025-05-12"
    # Morning snapshot
    cache.record_thread_snapshot(
        thread_id="th_1",
        date=date,
        slot="morning",
        reply_count=5,
        source="mattermost",
        title="Refactoring discussion",
    )
    # Afternoon snapshot with 3 new replies (total 8)
    cache.record_thread_snapshot(
        thread_id="th_1",
        date=date,
        slot="afternoon",
        reply_count=8,
        source="mattermost",
        title="Refactoring discussion",
    )

    # Afternoon snapshot for a thread that had no morning record (new thread)
    cache.record_thread_snapshot(
        thread_id="th_2",
        date=date,
        slot="afternoon",
        reply_count=2,
        source="mattermost",
        title="Lunch plans",
    )

    deltas = cache.get_thread_updates_since_morning(date)
    assert len(deltas) == 2
    th1 = next(d for d in deltas if d["thread_id"] == "th_1")
    assert th1["delta_replies"] == 3
    assert th1["am_count"] == 5
    assert th1["pm_count"] == 8

    th2 = next(d for d in deltas if d["thread_id"] == "th_2")
    assert th2["delta_replies"] == 2
    assert th2["am_count"] == 0


def test_digest_history(tmp_path):
    db_file = tmp_path / "cache.db"
    cache = ActivityCache(db_file)

    cache.save_digest("2025-05-12", "morning", "# Morning Digest\nAll good.")
    cached = cache.get_digest("2025-05-12", "morning")
    assert cached == "# Morning Digest\nAll good."
    assert cache.get_digest("2025-05-12", "afternoon") is None

    digests = cache.list_digests()
    assert len(digests) == 1
    assert digests[0]["id"] == "2025-05-12_morning"
