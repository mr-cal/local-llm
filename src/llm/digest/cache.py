"""SQLite cache with FTS5 and vector semantic embeddings for digest items."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any

import httpx

sqlite_vec: Any = None
with contextlib.suppress(ImportError):
    import sqlite_vec


def _pack_vector(vector: list[float]) -> bytes:
    """Pack list of floats into binary buffer."""
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    """Unpack binary buffer into list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class ActivityCache:
    """Persistent SQLite cache storing items, thread snapshots, and embeddings."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if sqlite_vec is not None:
            with contextlib.suppress(Exception):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    thread_id TEXT,
                    title TEXT,
                    author TEXT,
                    content TEXT,
                    timestamp TEXT,
                    url TEXT,
                    metadata_json TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS item_embeddings (
                    item_id TEXT PRIMARY KEY,
                    dimensions INTEGER,
                    vector BLOB,
                    FOREIGN KEY (item_id) REFERENCES items (id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS digest_history (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thread_snapshots (
                    thread_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    source TEXT,
                    title TEXT,
                    reply_count INTEGER NOT NULL,
                    last_reply_at TEXT,
                    PRIMARY KEY (thread_id, date, slot)
                )
            """)

            # Create FTS5 virtual table if supported
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    id UNINDEXED,
                    title,
                    content,
                    author,
                    source UNINDEXED,
                    content='items',
                    content_rowid='rowid'
                )
            """)
            # Triggers to keep FTS index synced with items table
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
                    INSERT INTO items_fts(rowid, id, title, content, author, source)
                    VALUES (new.rowid, new.id, new.title, new.content, new.author, new.source);
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
                    INSERT INTO items_fts(items_fts, rowid, id, title, content, author, source)
                    VALUES('delete', old.rowid, old.id, old.title, old.content, old.author, old.source);
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
                    INSERT INTO items_fts(items_fts, rowid, id, title, content, author, source)
                    VALUES('delete', old.rowid, old.id, old.title, old.content, old.author, old.source);
                    INSERT INTO items_fts(rowid, id, title, content, author, source)
                    VALUES (new.rowid, new.id, new.title, new.content, new.author, new.source);
                END;
            """)
            conn.commit()

    def upsert_item(
        self,
        id: str,
        source: str,
        item_id: str,
        content: str,
        thread_id: str | None = None,
        title: str | None = None,
        author: str | None = None,
        timestamp: str | None = None,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace an ingested item in the cache."""
        meta_str = json.dumps(metadata) if metadata else "{}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO items (
                    id, source, item_id, thread_id, title, author, content, timestamp, url, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, source, item_id, thread_id, title, author, content, timestamp, url, meta_str),
            )
            conn.commit()

    def get_item(self, id: str) -> dict[str, Any] | None:
        """Retrieve a single item by id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM items WHERE id = ?", (id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("metadata_json"):
                with contextlib.suppress(Exception):
                    res["metadata"] = json.loads(res["metadata_json"])
            return res

    def search_text(self, query: str, source: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search using SQLite FTS5."""
        if not query.strip():
            return []

        # Sanitize query for FTS5 (escape double quotes)
        safe_query = '"' + query.replace('"', '""') + '"'
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if source:
                sql = """
                    SELECT i.*, fts.rank
                    FROM items_fts fts
                    JOIN items i ON fts.id = i.id
                    WHERE items_fts MATCH ? AND i.source = ?
                    ORDER BY fts.rank
                    LIMIT ?
                """
                params = (safe_query, source, limit)
            else:
                sql = """
                    SELECT i.*, fts.rank
                    FROM items_fts fts
                    JOIN items i ON fts.id = i.id
                    WHERE items_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """
                params = (safe_query, limit)

            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # Fallback to LIKE if FTS query syntax error occurs
                like_query = f"%{query}%"
                cursor.execute(
                    """
                    SELECT *, 0.0 as rank FROM items
                    WHERE (title LIKE ? OR content LIKE ?)
                    LIMIT ?
                    """,
                    (like_query, like_query, limit),
                )
                rows = cursor.fetchall()

            results = []
            for r in rows:
                item = dict(r)
                if item.get("metadata_json"):
                    with contextlib.suppress(Exception):
                        item["metadata"] = json.loads(item["metadata_json"])
                results.append(item)
            return results

    def upsert_embedding(self, item_id: str, vector: list[float]) -> None:
        """Store embedding vector for an item."""
        packed = _pack_vector(vector)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO item_embeddings (item_id, dimensions, vector)
                VALUES (?, ?, ?)
                """,
                (item_id, len(vector), packed),
            )
            conn.commit()

    def search_vector(self, query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Cosine similarity search over stored embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT item_id, dimensions, vector FROM item_embeddings")
            rows = cursor.fetchall()

            scored: list[tuple[float, str]] = []
            for r in rows:
                item_id = r["item_id"]
                blob = r["vector"]
                vec = _unpack_vector(blob)
                score = _cosine_similarity(query_vector, vec)
                scored.append((score, item_id))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_ids = [item_id for _, item_id in scored[:limit]]

            if not top_ids:
                return []

            placeholders = ",".join("?" for _ in top_ids)
            cursor.execute(f"SELECT * FROM items WHERE id IN ({placeholders})", top_ids)
            items_by_id = {row["id"]: dict(row) for row in cursor.fetchall()}

            results = []
            for score, item_id in scored[:limit]:
                if item_id in items_by_id:
                    res = items_by_id[item_id]
                    res["similarity_score"] = score
                    results.append(res)
            return results

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Combine FTS5 keyword results with semantic vector similarity."""
        text_results = self.search_text(query, limit=limit * 2)
        if not query_vector:
            return text_results[:limit]

        vector_results = self.search_vector(query_vector, limit=limit * 2)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        items_map: dict[str, dict[str, Any]] = {}

        k = 60.0
        for rank, item in enumerate(text_results):
            iid = item["id"]
            items_map[iid] = item
            rrf_scores[iid] = rrf_scores.get(iid, 0.0) + (1.0 / (k + rank + 1))

        for rank, item in enumerate(vector_results):
            iid = item["id"]
            items_map[iid] = item
            rrf_scores[iid] = rrf_scores.get(iid, 0.0) + (1.0 / (k + rank + 1))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [items_map[iid] for iid, _ in sorted_items[:limit]]

    def record_thread_snapshot(
        self,
        thread_id: str,
        date: str,
        slot: str,
        reply_count: int,
        last_reply_at: str | None = None,
        source: str = "",
        title: str = "",
    ) -> None:
        """Record thread reply counts for morning/afternoon delta calculation."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO thread_snapshots (
                    thread_id, date, slot, source, title, reply_count, last_reply_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, date, slot, source, title, reply_count, last_reply_at),
            )
            conn.commit()

    def get_thread_updates_since_morning(self, date: str) -> list[dict[str, Any]]:
        """Compare afternoon thread state against morning state for *date*."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    pm.thread_id,
                    pm.source,
                    pm.title,
                    pm.reply_count AS pm_count,
                    COALESCE(am.reply_count, 0) AS am_count,
                    (pm.reply_count - COALESCE(am.reply_count, 0)) AS delta_replies,
                    pm.last_reply_at
                FROM thread_snapshots pm
                LEFT JOIN thread_snapshots am
                    ON pm.thread_id = am.thread_id
                    AND pm.date = am.date
                    AND am.slot = 'morning'
                WHERE pm.date = ?
                  AND pm.slot = 'afternoon'
                  AND (pm.reply_count - COALESCE(am.reply_count, 0)) > 0
                ORDER BY delta_replies DESC
                """,
                (date,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def save_digest(self, date: str, slot: str, markdown: str) -> None:
        """Save generated markdown digest."""
        now_str = dt.datetime.now(dt.UTC).isoformat()
        digest_id = f"{date}_{slot}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO digest_history (id, date, slot, markdown, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (digest_id, date, slot, markdown, now_str),
            )
            conn.commit()

    def get_digest(self, date: str, slot: str) -> str | None:
        """Retrieve stored markdown digest for *date* and *slot*."""
        digest_id = f"{date}_{slot}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT markdown FROM digest_history WHERE id = ?", (digest_id,))
            row = cursor.fetchone()
            return row["markdown"] if row else None

    def list_digests(self, limit: int = 30) -> list[dict[str, Any]]:
        """List historical digests."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, date, slot, created_at, length(markdown) as size
                FROM digest_history
                ORDER BY date DESC, slot DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]


def fetch_embedding_from_server(text: str, base_url: str = "http://127.0.0.1:8081") -> list[float]:
    """Call the local llama-server embedding endpoint."""
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    payload = {"input": text}
    try:
        resp = httpx.post(url, json=payload, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception:
        # Also try /embedding endpoint if /v1/embeddings is not matched
        fallback_url = f"{base_url.rstrip('/')}/embedding"
        resp = httpx.post(fallback_url, json={"content": text}, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])
