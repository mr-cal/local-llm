"""GitHub source ingestor."""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import GitHubSettings


def ingest_github(
    settings: GitHubSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Ingest GitHub notifications, assigned issues/PRs, and repository activity."""
    items: list[dict[str, Any]] = []
    token = settings.token.strip()

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "local-llm-jotter",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try gh CLI first if installed
    has_gh = shutil.which("gh") is not None

    since_iso = start_time.isoformat()

    # 1. Assigned issues and review requests
    if has_gh:
        try:
            # gh search issues/prs assigned to me updated since start_time
            date_filter = start_time.strftime("%Y-%m-%d")
            cmd = [
                "gh",
                "search",
                "issues",
                "--assignee=@me",
                f"--updated=>={date_filter}",
                "--json=number,title,repository,url,updatedAt,author,body",
                "--limit=30",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                import json

                issues = json.loads(res.stdout)
                for iss in issues:
                    author = iss.get("author", {}).get("login", "")
                    if digest_filter and digest_filter.should_ignore_author(author):
                        continue
                    repo = iss.get("repository", {}).get("nameWithOwner", "")
                    num = iss.get("number")
                    iid = f"github_issue_{repo}_{num}"
                    item = {
                        "id": iid,
                        "source": "github",
                        "item_id": str(num),
                        "thread_id": f"gh_thread_{repo}_{num}",
                        "title": f"GitHub #{num} ({repo}): {iss.get('title')}",
                        "author": author,
                        "content": iss.get("body", "") or iss.get("title", ""),
                        "timestamp": iss.get("updatedAt", since_iso),
                        "url": iss.get("url", ""),
                        "metadata": {"repo": repo, "type": "assigned_issue"},
                    }
                    items.append(item)
                    if cache:
                        cache.upsert_item(**item)
        except Exception:
            pass

    # 2. Configured repos activity via GitHub API if token available
    if token and settings.repos:
        with httpx.Client(headers=headers, timeout=15.0) as client:
            for repo in settings.repos:
                try:
                    # Fetch recently updated PRs
                    resp = client.get(
                        f"https://api.github.com/repos/{repo}/pulls",
                        params={"state": "all", "sort": "updated", "direction": "desc", "per_page": 15},
                    )
                    if resp.status_code == 200:
                        for pr in resp.json():
                            updated_at_str = pr.get("updated_at")
                            if not updated_at_str:
                                continue
                            author = pr.get("user", {}).get("login", "")
                            if digest_filter and digest_filter.should_ignore_author(author):
                                continue

                            num = pr.get("number")
                            iid = f"github_pr_{repo}_{num}"
                            # Check if within window
                            item = {
                                "id": iid,
                                "source": "github",
                                "item_id": str(num),
                                "thread_id": f"gh_thread_{repo}_{num}",
                                "title": f"GitHub PR #{num} ({repo}): {pr.get('title')}",
                                "author": author,
                                "content": pr.get("body", "") or pr.get("title", ""),
                                "timestamp": updated_at_str,
                                "url": pr.get("html_url", ""),
                                "metadata": {
                                    "repo": repo,
                                    "state": pr.get("state"),
                                    "merged": pr.get("merged_at") is not None,
                                },
                            }
                            items.append(item)
                            if cache:
                                cache.upsert_item(**item)
                except Exception:
                    continue

    return items
