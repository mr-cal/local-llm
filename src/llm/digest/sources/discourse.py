"""Discourse source ingestor."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from llm.discourse.client import DiscourseClient

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import DiscourseSettings


def ingest_discourse(
    settings: DiscourseSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Fetch Discourse posts within [start_time, end_time] and store in cache."""
    if not settings.is_authenticated():
        return []

    client = DiscourseClient(base_url=settings.url, session_cookie=settings.session_cookie)
    try:
        topics = client.get_posts_since(start_time)
    except Exception:
        return []

    items = []
    for topic in topics:
        title = topic.get("title", "")
        topic_id = str(topic.get("id", ""))
        url = f"{settings.url.rstrip('/')}/t/{topic.get('slug', topic_id)}/{topic_id}"

        # Topic level noise filter
        if digest_filter and digest_filter.should_ignore_topic(title):
            continue

        # Category filter
        if settings.tracked_categories != "all" and settings.tracked_categories:
            cat_id = str(topic.get("category_id", ""))
            if cat_id not in settings.tracked_categories:
                continue

        # Tag filter
        if settings.tracked_tags != "all" and settings.tracked_tags:
            topic_tags = {str(tg).lower() for tg in topic.get("tags", [])}
            tracked = {t.lower() for t in settings.tracked_tags}
            if not topic_tags.intersection(tracked):
                continue

        for post in topic.get("recent_posts", []):
            author = post.get("username", "")
            if digest_filter and digest_filter.should_ignore_author(author):
                continue

            created_at_str = post.get("created_at", "")
            post_id = str(post.get("id", ""))
            item_id = f"discourse_{topic_id}_{post_id}"
            content = post.get("cooked", "") or post.get("raw", "")

            item = {
                "id": item_id,
                "source": "discourse",
                "item_id": post_id,
                "thread_id": f"discourse_topic_{topic_id}",
                "title": f"Discourse: {title}",
                "author": author,
                "content": content,
                "timestamp": created_at_str,
                "url": url,
                "metadata": {
                    "topic_id": topic_id,
                    "post_number": post.get("post_number"),
                },
            }
            items.append(item)
            if cache:
                cache.upsert_item(**item)

    return items
