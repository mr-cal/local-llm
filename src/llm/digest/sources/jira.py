"""Atlassian Jira source ingestor and search client."""

from __future__ import annotations

import base64
import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import JiraSettings


def _get_auth_header(settings: JiraSettings) -> dict[str, str]:
    if settings.email and settings.api_token:
        cred = f"{settings.email}:{settings.api_token}"
        encoded = base64.b64encode(cred.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    if settings.api_token:
        return {"Authorization": f"Bearer {settings.api_token}"}
    return {}


def ingest_jira(
    settings: JiraSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Fetch Jira assigned issues, sprint tickets, and recent changes."""
    if not settings.is_authenticated():
        return []

    items: list[dict[str, Any]] = []
    base_url = settings.url.rstrip("/")
    headers = {
        **_get_auth_header(settings),
        "Accept": "application/json",
    }

    date_str = start_time.strftime("%Y-%m-%d %H:%M")
    # JQL: updated >= start_time AND (assignee = currentUser() OR project IN (...))
    jql_parts = [f"updated >= '{date_str}'"]
    if settings.projects:
        proj_list = ",".join(f"'{p}'" for p in settings.projects)
        jql_parts.append(f"(assignee = currentUser() OR project IN ({proj_list}))")
    else:
        jql_parts.append("assignee = currentUser()")

    jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"

    try:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            resp = client.get(
                f"{base_url}/rest/api/3/search",
                params={
                    "jql": jql,
                    "maxResults": 30,
                    "fields": "summary,status,assignee,updated,description,comment",
                },
            )
            if resp.status_code != 200:
                # Try v2 if v3 is not available
                resp = client.get(
                    f"{base_url}/rest/api/2/search",
                    params={
                        "jql": jql,
                        "maxResults": 30,
                        "fields": "summary,status,assignee,updated,description,comment",
                    },
                )
                if resp.status_code != 200:
                    return []

            data = resp.json()
            issues = data.get("issues", [])
            for issue in issues:
                key = issue.get("key")
                fields = issue.get("fields", {})
                summary = fields.get("summary", "")
                status_name = fields.get("status", {}).get("name", "")
                assignee_info = fields.get("assignee") or {}
                assignee_name = assignee_info.get("displayName", "Unassigned")
                updated = fields.get("updated", "")

                if digest_filter and digest_filter.should_ignore_author(assignee_name):
                    continue

                iid = f"jira_{key}"
                content = f"Status: {status_name}\nAssignee: {assignee_name}\nSummary: {summary}"
                url = f"{base_url}/browse/{key}"

                item = {
                    "id": iid,
                    "source": "jira",
                    "item_id": key,
                    "thread_id": f"jira_issue_{key}",
                    "title": f"Jira {key}: {summary}",
                    "author": assignee_name,
                    "content": content,
                    "timestamp": updated,
                    "url": url,
                    "metadata": {"key": key, "status": status_name},
                }
                items.append(item)
                if cache:
                    cache.upsert_item(**item)
    except Exception:
        pass

    return items


def search_jira(settings: JiraSettings, query: str, limit: int = 15) -> list[dict[str, Any]]:
    """Search Jira issues using text query."""
    if not settings.is_authenticated():
        return []

    base_url = settings.url.rstrip("/")
    headers = {
        **_get_auth_header(settings),
        "Accept": "application/json",
    }
    safe_q = query.replace("'", "\\'")
    jql = f"text ~ '{safe_q}' ORDER BY updated DESC"

    try:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            resp = client.get(
                f"{base_url}/rest/api/3/search",
                params={"jql": jql, "maxResults": limit, "fields": "summary,status,assignee,updated"},
            )
            if resp.status_code != 200:
                resp = client.get(
                    f"{base_url}/rest/api/2/search",
                    params={"jql": jql, "maxResults": limit, "fields": "summary,status,assignee,updated"},
                )
                if resp.status_code != 200:
                    return []

            data = resp.json()
            out = []
            for iss in data.get("issues", []):
                fields = iss.get("fields", {})
                out.append(
                    {
                        "key": iss.get("key"),
                        "summary": fields.get("summary"),
                        "status": fields.get("status", {}).get("name"),
                        "assignee": (fields.get("assignee") or {}).get("displayName"),
                        "updated": fields.get("updated"),
                        "url": f"{base_url}/browse/{iss.get('key')}",
                    }
                )
            return out
    except Exception:
        return []


def get_jira_issue(settings: JiraSettings, key: str) -> dict[str, Any]:
    """Retrieve full details of a specific Jira issue."""
    if not settings.is_authenticated():
        return {}

    base_url = settings.url.rstrip("/")
    headers = {
        **_get_auth_header(settings),
        "Accept": "application/json",
    }

    try:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            resp = client.get(f"{base_url}/rest/api/3/issue/{key}")
            if resp.status_code != 200:
                resp = client.get(f"{base_url}/rest/api/2/issue/{key}")
                if resp.status_code != 200:
                    return {}
            return resp.json()
    except Exception:
        return {}
