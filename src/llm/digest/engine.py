"""Slim prompt engine and single-pass summarizer for daily digests."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx

from llm.digest.cache import ActivityCache
from llm.digest.filter import DigestFilter
from llm.digest.schedule import Slot, get_lookback_window
from llm.digest.sources.discourse import ingest_discourse
from llm.digest.sources.github import ingest_github
from llm.digest.sources.google import ingest_google
from llm.digest.sources.jira import ingest_jira
from llm.digest.sources.matrix import ingest_matrix
from llm.digest.sources.mattermost import ingest_mattermost

if TYPE_CHECKING:
    from llm.settings.models import Settings


class DigestEngine:
    """Orchestrates source ingestion, prompt assembly, and LLM summarization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = ActivityCache(self.settings.digest.cache_db_path)
        self.filter = DigestFilter(self.settings.digest.filter)

    def run_digest(
        self,
        slot: Slot = Slot.MORNING,
        now: dt.datetime | None = None,
        llm_url: str | None = None,
    ) -> str:
        """Run complete digest pipeline for *slot* and return markdown summary."""
        if now is None:
            now = dt.datetime.now(dt.UTC)

        start_time, end_time = get_lookback_window(now=now, slot=slot)
        today_str = now.strftime("%Y-%m-%d")

        # 1. Gather items from active sources
        items: list[dict[str, Any]] = []

        # Discourse only included in morning slot
        if slot == Slot.MORNING:
            items.extend(
                ingest_discourse(
                    self.settings.discourse,
                    start_time,
                    end_time,
                    digest_filter=self.filter,
                    cache=self.cache,
                )
            )

        items.extend(
            ingest_github(
                self.settings.github,
                start_time,
                end_time,
                digest_filter=self.filter,
                cache=self.cache,
            )
        )
        items.extend(
            ingest_mattermost(
                self.settings.mattermost,
                start_time,
                end_time,
                digest_filter=self.filter,
                cache=self.cache,
            )
        )
        items.extend(
            ingest_matrix(
                self.settings.matrix,
                start_time,
                end_time,
                digest_filter=self.filter,
                cache=self.cache,
            )
        )
        items.extend(
            ingest_google(
                self.settings.google,
                start_time,
                end_time,
                digest_filter=self.filter,
                cache=self.cache,
            )
        )
        items.extend(
            ingest_jira(
                self.settings.jira,
                start_time,
                end_time,
                digest_filter=self.filter,
                cache=self.cache,
            )
        )

        # 2. Check thread updates if afternoon slot
        thread_updates: list[dict[str, Any]] = []
        if slot == Slot.AFTERNOON:
            thread_updates = self.cache.get_thread_updates_since_morning(today_str)

        # 3. Assemble compact slim prompt
        prompt = self._build_slim_prompt(
            slot=slot,
            today_str=today_str,
            items=items,
            thread_updates=thread_updates,
        )

        # 4. Generate summary with LLM
        target_url = (llm_url or self.settings.client_url).rstrip("/")
        markdown_summary = self._call_llm(target_url, prompt)

        # 5. Persist digest
        self.cache.save_digest(today_str, slot.value, markdown_summary)
        out_dir = self.settings.digest.resolved_digest_output_dir
        out_file = out_dir / f"{today_str}-{slot.value}.md"
        out_file.write_text(markdown_summary, encoding="utf-8")

        return markdown_summary

    def _build_slim_prompt(
        self,
        slot: Slot,
        today_str: str,
        items: list[dict[str, Any]],
        thread_updates: list[dict[str, Any]],
    ) -> str:
        """Construct a compact prompt minimizing tokens."""
        title = f"{slot.value.capitalize()} Digest for {today_str}"
        lines = [
            f"# Summarize Activity: {title}",
            "Provide a concise, high-signal bullet-point summary organized by category:",
            "- Priority Action Items & Direct Mentions",
            "- Meetings & Calendar (Google)",
            "- Code & Reviews (GitHub, Craft)",
            "- Team Discussions & Threads (Discourse, Mattermost, Matrix)",
            "- Jira Issues & Tickets",
        ]

        if thread_updates:
            lines.append("\n## Thread Activity Delta (Since Morning):")
            for tu in thread_updates:
                lines.append(
                    f"- {tu.get('source')}: '{tu.get('title')}' had {tu.get('delta_replies')} new replies."
                )

        lines.append(f"\n## Activity Items ({len(items)} items):")
        if not items:
            lines.append("No new items in this window.")
        else:
            for it in items[:60]:  # limit to top 60 items to keep prompt tight
                src = it.get("source", "")
                author = it.get("author", "")
                title_item = it.get("title", "")
                content_excerpt = (it.get("content", "") or "")[:200].replace("\n", " ")
                lines.append(f"[{src}] {title_item} (by {author}): {content_excerpt}")

        lines.append("\nSummary:")
        return "\n".join(lines)

    def _call_llm(self, base_url: str, prompt: str) -> str:
        """Call LLM completion endpoint."""
        url = f"{base_url}/chat/completions"
        headers = {}
        if self.settings.client_api_key:
            headers["Authorization"] = f"Bearer {self.settings.client_api_key}"

        payload = {
            "model": "local",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Jotter, an executive assistant. "
                        "Generate clean, high-signal markdown summaries without filler."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            # Fallback markdown when server is offline or unreachable
            return (
                f"# Digest (Offline Mode)\n\n"
                f"Could not reach LLM server at {url}: {exc}\n\n"
                f"## Raw Items\n{prompt}"
            )
