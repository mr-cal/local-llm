"""Discourse HTTP API client using session cookie authentication."""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def get_default_summary_window(now: dt.datetime | None = None) -> tuple[dt.datetime, int]:
    """Calculate the summary start datetime and duration in hours.

    On Monday (weekday == 0), look back 72 hours (covering Friday, Saturday, Sunday).
    On Tuesday through Friday (weekday 1..4), look back 24 hours.
    On Saturday or Sunday (weekday 5..6), look back 24 hours.
    """
    current = now or dt.datetime.now(dt.UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.UTC)

    # Monday is 0 in Python
    hours = 72 if current.weekday() == 0 else 24
    start_time = current - dt.timedelta(hours=hours)
    return start_time, hours


class DiscourseClient:
    """Read-only client for Discourse forums using session cookie authentication."""

    def __init__(
        self,
        base_url: str,
        session_cookie: str,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_cookie = session_cookie.strip()
        self.timeout = timeout
        self.user_agent = user_agent

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.session_cookie:
            headers["Cookie"] = f"_t={self.session_cookie}"
        return headers

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    def check_auth(self) -> dict[str, Any]:
        """Verify session authentication by checking current user session."""
        try:
            data = self._request("session/current.json")
            user = data.get("current_user", {})
            return {
                "authenticated": bool(user),
                "username": user.get("username", "anonymous"),
                "name": user.get("name", ""),
                "admin": user.get("admin", False),
            }
        except httpx.HTTPStatusError as exc:
            return {
                "authenticated": False,
                "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            }
        except Exception as exc:
            return {
                "authenticated": False,
                "error": str(exc),
            }

    def get_latest_topics(self, limit: int = 30) -> list[dict[str, Any]]:
        """Fetch latest topics from /latest.json."""
        data = self._request("latest.json")
        topic_list = data.get("topic_list", {})
        topics = topic_list.get("topics", [])
        return topics[:limit]

    def get_top_topics(self, period: str = "daily", limit: int = 20) -> list[dict[str, Any]]:
        """Fetch top topics for a given period ('daily', 'weekly', 'monthly')."""
        data = self._request(f"top.json?period={period}")
        topic_list = data.get("topic_list", {})
        topics = topic_list.get("topics", [])
        return topics[:limit]

    def get_topic(self, topic_id: int) -> dict[str, Any]:
        """Fetch full details and post stream for a topic."""
        return self._request(f"t/{topic_id}.json")

    def search(self, query: str) -> dict[str, Any]:
        """Search topics and posts with query."""
        return self._request("search.json", params={"q": query})

    def get_posts_since(self, since: dt.datetime, max_topics: int = 30) -> list[dict[str, Any]]:
        """Get topics and their posts created or bumped since a given datetime."""
        if since.tzinfo is None:
            since = since.replace(tzinfo=dt.UTC)

        latest = self.get_latest_topics(limit=max_topics)
        recent_topics: list[dict[str, Any]] = []

        for topic in latest:
            bumped_str = topic.get("bumped_at") or topic.get("created_at")
            if not bumped_str:
                continue

            try:
                # Discourse timestamps are ISO 8601 strings (e.g., 2026-10-06T15:20:00.000Z)
                bumped_at = dt.datetime.fromisoformat(bumped_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            if bumped_at >= since:
                recent_topics.append(topic)

        # For the filtered topics, collect summary details
        results: list[dict[str, Any]] = []
        for t in recent_topics:
            t_id = t["id"]
            try:
                details = self.get_topic(t_id)
                post_stream = details.get("post_stream", {}).get("posts", [])
                # Filter posts within the time window
                matching_posts = []
                for p in post_stream:
                    p_created = p.get("created_at")
                    if p_created:
                        try:
                            p_time = dt.datetime.fromisoformat(p_created.replace("Z", "+00:00"))
                            if p_time >= since:
                                matching_posts.append(
                                    {
                                        "id": p.get("id"),
                                        "post_number": p.get("post_number"),
                                        "username": p.get("username"),
                                        "created_at": p_created,
                                        "cooked": p.get("cooked", ""),
                                    }
                                )
                        except ValueError:
                            pass

                results.append(
                    {
                        "id": t_id,
                        "title": t.get("title", ""),
                        "slug": t.get("slug", ""),
                        "url": f"{self.base_url}/t/{t.get('slug', 'topic')}/{t_id}",
                        "posts_count": t.get("posts_count", 0),
                        "created_at": t.get("created_at"),
                        "bumped_at": t.get("bumped_at"),
                        "category_id": t.get("category_id"),
                        "recent_posts": matching_posts,
                    }
                )
            except Exception:
                # Still include topic overview if full thread fetch failed
                results.append(
                    {
                        "id": t_id,
                        "title": t.get("title", ""),
                        "slug": t.get("slug", ""),
                        "url": f"{self.base_url}/t/{t.get('slug', 'topic')}/{t_id}",
                        "posts_count": t.get("posts_count", 0),
                        "created_at": t.get("created_at"),
                        "bumped_at": t.get("bumped_at"),
                        "category_id": t.get("category_id"),
                        "recent_posts": [],
                    }
                )

        return results
