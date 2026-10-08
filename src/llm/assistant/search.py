"""Multi-step search assistant across local cache, code, and external services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from llm.digest.cache import ActivityCache
from llm.digest.sources.craft import CraftInspector
from llm.digest.sources.google import get_drive_file_activity, search_google_drive
from llm.digest.sources.jira import get_jira_issue, search_jira
from llm.discourse.client import DiscourseClient
from llm.tui.llm_stream import ToolCallEvent

if TYPE_CHECKING:
    from llm.settings.models import Settings


class AssistantSearchRunner:
    """Runs interactive search queries across cache, services, and craft code."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = ActivityCache(self.settings.digest.cache_db_path)
        self.craft = CraftInspector(self.settings.digest.craft_dir)

    async def execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a search tool and return its result."""
        if name == "search_cache":
            query = args.get("query", "")
            return self.cache.search_text(query, limit=args.get("limit", 10))

        if name == "search_code":
            query = args.get("query", "")
            return self.craft.search_code(query, max_results=args.get("limit", 10))

        if name == "validate_code_reference":
            symbol = args.get("symbol", "")
            return self.craft.validate_reference(symbol)

        if name == "search_discourse":
            if not self.settings.discourse.is_authenticated():
                return {"error": "Discourse not configured"}
            client = DiscourseClient(
                base_url=self.settings.discourse.url,
                session_cookie=self.settings.discourse.session_cookie,
            )
            return client.search(args.get("query", ""))

        if name == "read_discourse_topic":
            if not self.settings.discourse.is_authenticated():
                return {"error": "Discourse not configured"}
            client = DiscourseClient(
                base_url=self.settings.discourse.url,
                session_cookie=self.settings.discourse.session_cookie,
            )
            return client.get_topic(int(args.get("topic_id", 0)))

        if name == "search_google_drive":
            return search_google_drive(self.settings.google, args.get("query", ""))

        if name == "get_drive_file_activity":
            return get_drive_file_activity(self.settings.google, args.get("file_id", ""))

        if name == "search_jira":
            return search_jira(self.settings.jira, args.get("query", ""))

        if name == "get_jira_issue":
            return get_jira_issue(self.settings.jira, args.get("key", ""))

        return {"error": f"Unknown tool: {name}"}

    async def search_and_stream(self, user_query: str) -> AsyncIterator[Any]:
        """Perform search across local cache, code, and external tools, yielding trace and answer."""
        yield ToolCallEvent(tool_name="search_cache", args={"query": user_query}, status="running")
        cache_results = await self.execute_tool("search_cache", {"query": user_query, "limit": 5})
        yield ToolCallEvent(
            tool_name="search_cache",
            args={"query": user_query},
            status="completed",
            result=f"Found {len(cache_results)} matching cached items",
        )

        code_results = []
        if any(kw in user_query.lower() for kw in ["code", "function", "snap", "craft", "key", "config"]):
            yield ToolCallEvent(tool_name="search_code", args={"query": user_query}, status="running")
            code_results = await self.execute_tool("search_code", {"query": user_query, "limit": 5})
            yield ToolCallEvent(
                tool_name="search_code",
                args={"query": user_query},
                status="completed",
                result=f"Found {len(code_results)} code matches in craft repos",
            )

        # Assemble summary context
        context_parts = []
        if cache_results:
            context_parts.append("### Relevant Cached Items:")
            for item in cache_results:
                context_parts.append(
                    f"- [{item.get('source')}] {item.get('title')}: {item.get('content', '')[:300]}"
                )

        if code_results:
            context_parts.append("\n### Relevant Code in ~/dev/craft:")
            for match in code_results:
                context_parts.append(f"- `{match.get('file')}:{match.get('line')}`: {match.get('content')}")

        context_text = "\n".join(context_parts) if context_parts else "No direct matches found in cache."

        # Query local LLM for grounded answer
        system_msg = (
            "You are Jotter, an executive assistant. Use the provided search results and code grounding "
            "to answer the user's question directly, concisely, and cite source references."
        )
        user_prompt = f"Context:\n{context_text}\n\nUser Question: {user_query}"

        from llm.tui.llm_stream import stream_chat_completion

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]
        async for event in stream_chat_completion(
            base_url=self.settings.client_url,
            messages=messages,
            api_key=self.settings.client_api_key,
        ):
            yield event
