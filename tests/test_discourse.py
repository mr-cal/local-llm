"""Unit tests for Discourse integration and MCP server."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

from llm.discourse.client import DiscourseClient, get_default_summary_window
from llm.discourse.server import create_discourse_mcp_server
from llm.settings.models import HermesSettings


def test_get_default_summary_window_monday() -> None:
    # 2026-10-05 is a Monday
    monday = dt.datetime(2026, 10, 5, 12, 0, 0, tzinfo=dt.UTC)
    start_time, hours = get_default_summary_window(monday)
    assert hours == 72
    assert start_time == dt.datetime(2026, 10, 2, 12, 0, 0, tzinfo=dt.UTC)


def test_get_default_summary_window_weekday() -> None:
    # 2026-10-06 is a Tuesday
    tuesday = dt.datetime(2026, 10, 6, 12, 0, 0, tzinfo=dt.UTC)
    start_time, hours = get_default_summary_window(tuesday)
    assert hours == 24
    assert start_time == dt.datetime(2026, 10, 5, 12, 0, 0, tzinfo=dt.UTC)


def test_hermes_settings_has_discourse() -> None:
    cfg = HermesSettings()
    assert not cfg.has_discourse()

    cfg.discourse_url = "https://canonical.discourse.com"
    cfg.discourse_session_cookie = "mock_t_cookie"
    assert cfg.has_discourse()


def test_discourse_client_headers() -> None:
    client = DiscourseClient(
        base_url="https://canonical.discourse.com",
        session_cookie="cookie_xyz",
    )
    headers = client._headers()
    assert headers["Cookie"] == "_t=cookie_xyz"
    assert "User-Agent" in headers
    assert headers["Accept"] == "application/json"


@patch("llm.discourse.client.httpx.Client")
def test_discourse_client_check_auth(mock_client_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_instance

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "current_user": {
            "username": "testuser",
            "name": "Test User",
            "admin": False,
        }
    }
    mock_instance.get.return_value = mock_resp

    client = DiscourseClient("https://canonical.discourse.com", "cookie_xyz")
    status = client.check_auth()

    assert status["authenticated"] is True
    assert status["username"] == "testuser"
    assert status["admin"] is False


@patch("llm.discourse.client.httpx.Client")
def test_discourse_client_get_posts_since(mock_client_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_instance

    latest_resp = MagicMock()
    latest_resp.json.return_value = {
        "topic_list": {
            "topics": [
                {
                    "id": 101,
                    "title": "Recent Topic",
                    "slug": "recent-topic",
                    "bumped_at": "2026-10-06T10:00:00.000Z",
                    "posts_count": 2,
                },
                {
                    "id": 102,
                    "title": "Old Topic",
                    "slug": "old-topic",
                    "bumped_at": "2026-10-01T10:00:00.000Z",
                    "posts_count": 1,
                },
            ]
        }
    }

    topic_resp = MagicMock()
    topic_resp.json.return_value = {
        "id": 101,
        "title": "Recent Topic",
        "slug": "recent-topic",
        "post_stream": {
            "posts": [
                {
                    "id": 201,
                    "post_number": 1,
                    "username": "alice",
                    "created_at": "2026-10-06T10:00:00.000Z",
                    "cooked": "<p>Hello world</p>",
                }
            ]
        },
    }

    mock_instance.get.side_effect = [latest_resp, topic_resp]

    client = DiscourseClient("https://canonical.discourse.com", "cookie_xyz")
    since = dt.datetime(2026, 10, 5, 0, 0, 0, tzinfo=dt.UTC)
    topics = client.get_posts_since(since=since)

    assert len(topics) == 1
    assert topics[0]["id"] == 101
    assert topics[0]["title"] == "Recent Topic"
    assert len(topics[0]["recent_posts"]) == 1


def test_discourse_mcp_server_init() -> None:
    server = create_discourse_mcp_server(
        url="https://canonical.discourse.com",
        session_cookie="test_cookie",
    )
    assert server.name == "discourse"
