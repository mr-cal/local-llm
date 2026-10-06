"""MCP server implementation for Discourse."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from llm.discourse.client import DiscourseClient, get_default_summary_window


def create_discourse_mcp_server(
    url: str | None = None,
    session_cookie: str | None = None,
) -> MCPServer:
    """Create and return a configured MCPServer for Discourse."""
    base_url = url or os.getenv("DISCOURSE_URL", "https://canonical.discourse.com")
    cookie = session_cookie or os.getenv("DISCOURSE_SESSION_COOKIE", "")

    client = DiscourseClient(base_url=base_url, session_cookie=cookie)
    mcp = MCPServer("discourse")

    @mcp.tool()
    def get_discourse_user_status() -> dict[str, Any]:
        """Check session cookie authentication status and return current user details."""
        return client.check_auth()

    @mcp.tool()
    def list_latest_topics(limit: int = 25) -> list[dict[str, Any]]:
        """List the latest active topics from the Discourse forum."""
        raw_topics = client.get_latest_topics(limit=limit)
        return [
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "slug": t.get("slug"),
                "url": f"{client.base_url}/t/{t.get('slug', 'topic')}/{t.get('id')}",
                "posts_count": t.get("posts_count"),
                "bumped_at": t.get("bumped_at"),
                "created_at": t.get("created_at"),
                "category_id": t.get("category_id"),
            }
            for t in raw_topics
        ]

    @mcp.tool()
    def get_topic_thread(topic_id: int) -> dict[str, Any]:
        """Fetch the full content and stream of posts for a specific topic ID."""
        data = client.get_topic(topic_id)
        post_stream = data.get("post_stream", {}).get("posts", [])
        posts = [
            {
                "id": p.get("id"),
                "post_number": p.get("post_number"),
                "username": p.get("username"),
                "created_at": p.get("created_at"),
                "cooked": p.get("cooked"),
            }
            for p in post_stream
        ]
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "url": f"{client.base_url}/t/{data.get('slug', 'topic')}/{topic_id}",
            "posts_count": data.get("posts_count"),
            "posts": posts,
        }

    @mcp.tool()
    def search_forum(query: str) -> dict[str, Any]:
        """Search the Discourse forum for posts and topics matching a query."""
        return client.search(query)

    @mcp.tool()
    def get_daily_summary_posts(lookback_hours: int | None = None) -> dict[str, Any]:
        """Fetch all topics and posts active within the summary window.

        If lookback_hours is not specified, dynamically uses 72 hours on Mondays (covering
        the weekend) and 24 hours on all other days.
        """
        import datetime as dt

        if lookback_hours is not None:
            since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=lookback_hours)
            hours = lookback_hours
        else:
            since, hours = get_default_summary_window()

        topics = client.get_posts_since(since=since)
        return {
            "window_hours": hours,
            "since": since.isoformat(),
            "total_active_topics": len(topics),
            "topics": topics,
        }

    return mcp
