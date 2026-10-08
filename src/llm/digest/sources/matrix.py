"""Matrix source ingestor using Client-Server API v3."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import MatrixSettings


def ingest_matrix(
    settings: MatrixSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Fetch messages and room events from Matrix homeserver."""
    if not settings.is_authenticated():
        return []

    items: list[dict[str, Any]] = []
    base_url = settings.homeserver.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.access_token}",
        "Accept": "application/json",
    }

    start_ms = int(start_time.timestamp() * 1000)

    try:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            # If explicit rooms configured, query their message history
            rooms_to_check = settings.rooms

            # If no rooms configured, query joined rooms
            if not rooms_to_check:
                r_joined = client.get(f"{base_url}/_matrix/client/v3/joined_rooms")
                if r_joined.status_code == 200:
                    rooms_to_check = r_joined.json().get("joined_rooms", [])

            for room_id in rooms_to_check:
                if digest_filter and digest_filter.should_ignore_channel(room_id):
                    continue

                # Fetch messages in room
                resp = client.get(
                    f"{base_url}/_matrix/client/v3/rooms/{room_id}/messages",
                    params={"dir": "b", "limit": 40},
                )
                if resp.status_code != 200:
                    continue

                chunk = resp.json().get("chunk", [])
                for event in chunk:
                    if event.get("type") != "m.room.message":
                        continue

                    origin_server_ts = event.get("origin_server_ts", 0)
                    if origin_server_ts < start_ms:
                        # Reached events older than start_time
                        continue

                    sender = event.get("sender", "")
                    if digest_filter and digest_filter.should_ignore_author(sender):
                        continue

                    content_body = event.get("content", {}).get("body", "")
                    if not content_body:
                        continue

                    event_id = event.get("event_id", "")
                    ts_iso = dt.datetime.fromtimestamp(origin_server_ts / 1000.0, dt.UTC).isoformat()
                    iid = f"matrix_{event_id}"

                    item = {
                        "id": iid,
                        "source": "matrix",
                        "item_id": event_id,
                        "thread_id": f"matrix_room_{room_id}",
                        "title": f"Matrix: {room_id}",
                        "author": sender,
                        "content": content_body,
                        "timestamp": ts_iso,
                        "url": f"{base_url}/#/room/{room_id}",
                        "metadata": {"room_id": room_id},
                    }
                    items.append(item)
                    if cache:
                        cache.upsert_item(**item)
    except Exception:
        pass

    return items
