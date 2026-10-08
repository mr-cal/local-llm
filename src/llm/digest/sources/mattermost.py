"""Mattermost source ingestor using REST API v4."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import MattermostSettings


def ingest_mattermost(
    settings: MattermostSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Fetch unread posts, mentions, and thread updates from Mattermost."""
    if not settings.is_authenticated():
        return []

    items: list[dict[str, Any]] = []
    base_url = settings.url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.token}",
        "Accept": "application/json",
    }

    since_ms = int(start_time.timestamp() * 1000)

    try:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            # 1. Check current user
            me_resp = client.get(f"{base_url}/api/v4/users/me")
            if me_resp.status_code != 200:
                return []

            # 2. Get user's channels
            channels_resp = client.get(f"{base_url}/api/v4/users/me/channels")
            if channels_resp.status_code != 200:
                return []
            channels = channels_resp.json()

            for chan in channels:
                chan_id = chan.get("id")
                chan_name = chan.get("name", "")
                chan_display = chan.get("display_name", chan_name)
                chan_type = chan.get("type")  # 'D'=direct, 'O'=open, 'P'=private

                # Channel filtering:
                # Direct messages ('D') always included.
                # Other channels: included if configured in settings.channels or if settings.channels is empty
                if chan_type != "D" and settings.channels and chan_name not in settings.channels:
                    continue

                if digest_filter and digest_filter.should_ignore_channel(chan_name):
                    continue

                # Fetch posts in channel since start_time
                posts_resp = client.get(
                    f"{base_url}/api/v4/channels/{chan_id}/posts",
                    params={"since": since_ms, "per_page": 60},
                )
                if posts_resp.status_code != 200:
                    continue

                posts_data = posts_resp.json()
                order = posts_data.get("order", [])
                posts_dict = posts_data.get("posts", {})

                # Count replies for thread tracking
                thread_reply_counts: dict[str, int] = {}

                for pid in order:
                    post = posts_dict.get(pid, {})
                    author_id = post.get("user_id", "")
                    message = post.get("message", "").strip()
                    if not message:
                        continue

                    root_id = post.get("root_id") or pid
                    thread_reply_counts[root_id] = thread_reply_counts.get(root_id, 0) + 1

                    create_at = post.get("create_at", 0)
                    ts_iso = dt.datetime.fromtimestamp(create_at / 1000.0, dt.UTC).isoformat()
                    iid = f"mattermost_{pid}"

                    item = {
                        "id": iid,
                        "source": "mattermost",
                        "item_id": pid,
                        "thread_id": f"mattermost_thread_{root_id}",
                        "title": f"Mattermost #{chan_display}",
                        "author": author_id,
                        "content": message,
                        "timestamp": ts_iso,
                        "url": f"{base_url}/pl/{pid}",
                        "metadata": {
                            "channel_id": chan_id,
                            "channel_name": chan_name,
                            "root_id": root_id,
                        },
                    }
                    items.append(item)
                    if cache:
                        cache.upsert_item(**item)

                # Record thread snapshots in cache
                if cache:
                    today_str = start_time.strftime("%Y-%m-%d")
                    slot_name = "morning" if start_time.hour < 12 else "afternoon"
                    for tid, count in thread_reply_counts.items():
                        cache.record_thread_snapshot(
                            thread_id=f"mattermost_thread_{tid}",
                            date=today_str,
                            slot=slot_name,
                            reply_count=count,
                            source="mattermost",
                            title=f"Mattermost #{chan_display} thread",
                        )
    except Exception:
        pass

    return items
