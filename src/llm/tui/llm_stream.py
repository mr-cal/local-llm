"""LLM Streaming Client with exact token counting and calibrated progress estimation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

PERF_FILE = Path("~/.config/local-llm/perf.json").expanduser()
DEFAULT_PROMPT_TPS = 40.0


@dataclass
class PromptEvalProgress:
    tokens: int
    estimated_total_sec: float
    elapsed_sec: float
    progress_pct: float


@dataclass
class PromptEvalFinished:
    tokens: int
    actual_eval_sec: float
    actual_tps: float


@dataclass
class ReasoningToken:
    token: str


@dataclass
class AnswerToken:
    token: str


@dataclass
class ToolCallEvent:
    tool_name: str
    args: dict[str, Any]
    status: str
    result: Any | None = None


def load_calibrated_tps() -> float:
    """Read prompt evaluation speed from perf.json, defaulting to 40.0 tps."""
    if PERF_FILE.exists():
        try:
            data = json.loads(PERF_FILE.read_text(encoding="utf-8"))
            val = float(data.get("prompt_eval_tps", DEFAULT_PROMPT_TPS))
            if val > 0:
                return val
        except Exception:
            pass
    return DEFAULT_PROMPT_TPS


def update_calibrated_tps(tokens: int, eval_seconds: float) -> float:
    """Update running average of prompt evaluation speed."""
    if eval_seconds <= 0 or tokens <= 0:
        return load_calibrated_tps()

    current_tps = tokens / eval_seconds
    old_tps = load_calibrated_tps()
    # Exponential moving average (alpha=0.3)
    new_tps = 0.7 * old_tps + 0.3 * current_tps

    PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_eval_tps": round(new_tps, 2),
                "last_run": {
                    "tokens": tokens,
                    "seconds": round(eval_seconds, 2),
                    "tps": round(current_tps, 2),
                },
            },
            f,
            indent=2,
        )
    return new_tps


async def count_tokens_async(base_url: str, text: str) -> int:
    """Query llama-server /tokenize endpoint to get exact token count."""
    url = f"{base_url.rstrip('/')}/tokenize"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"content": text})
            if resp.status_code == 200:
                data = resp.json()
                return len(data.get("tokens", []))
    except Exception:
        pass
    # Rough fallback estimation: ~4 chars per token
    return max(1, len(text) // 4)


async def stream_chat_completion(
    base_url: str,
    messages: list[dict[str, str]],
    api_key: str = "",
    temperature: float = 0.3,
) -> AsyncIterator[Any]:
    """Stream chat completion yielding progress, reasoning tokens, and answer tokens."""
    # 1. Exact prompt token count
    prompt_text = "".join(m.get("content", "") for m in messages)
    token_count = await count_tokens_async(base_url, prompt_text)
    calibrated_tps = load_calibrated_tps()
    est_total_sec = token_count / calibrated_tps if calibrated_tps > 0 else 1.0

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": "local",
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    start_time = time.monotonic()
    first_token_received = False
    in_think_block = False

    async with httpx.AsyncClient(timeout=180.0) as client:
        # Start background timer for progress reporting until first token
        async def report_progress() -> AsyncIterator[PromptEvalProgress]:
            while not first_token_received:
                elapsed = time.monotonic() - start_time
                pct = min(99.0, (elapsed / est_total_sec) * 100.0) if est_total_sec > 0 else 50.0
                yield PromptEvalProgress(
                    tokens=token_count,
                    estimated_total_sec=est_total_sec,
                    elapsed_sec=elapsed,
                    progress_pct=pct,
                )
                await asyncio.sleep(0.1)

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")

                        if not first_token_received:
                            first_token_received = True
                            actual_eval = time.monotonic() - start_time
                            actual_tps = token_count / actual_eval if actual_eval > 0 else calibrated_tps
                            update_calibrated_tps(token_count, actual_eval)
                            yield PromptEvalFinished(
                                tokens=token_count,
                                actual_eval_sec=actual_eval,
                                actual_tps=actual_tps,
                            )

                        if reasoning:
                            yield ReasoningToken(reasoning)

                        if content:
                            if "<think>" in content:
                                in_think_block = True
                                content = content.replace("<think>", "")
                            if "</think>" in content:
                                in_think_block = False
                                parts = content.split("</think>", 1)
                                if parts[0]:
                                    yield ReasoningToken(parts[0])
                                if parts[1]:
                                    yield AnswerToken(parts[1])
                                continue

                            if in_think_block:
                                yield ReasoningToken(content)
                            else:
                                yield AnswerToken(content)

        except Exception as exc:
            yield AnswerToken(f"\n[Error streaming from LLM server: {exc}]")
