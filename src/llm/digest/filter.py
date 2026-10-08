"""Noise and bot filtering for digest ingestion."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm.settings.models import DigestFilterSettings

# Known common bot signatures
DEFAULT_BOT_PATTERNS = [
    "*bot*",
    "*[bot]*",
    "github-actions*",
    "jenkins*",
    "bors*",
    "canonical-bot*",
    "renovate*",
    "dependabot*",
]


class DigestFilter:
    """Filters messages, authors, channels, and topics based on configured rules."""

    def __init__(self, settings: DigestFilterSettings | None = None) -> None:
        if settings is None:
            from llm.settings.models import DigestFilterSettings

            self.settings = DigestFilterSettings()
        else:
            self.settings = settings

        self.ignored_authors = {a.strip().lower() for a in self.settings.ignored_authors if a.strip()}
        self.ignored_channels = {c.strip().lower() for c in self.settings.ignored_channels if c.strip()}
        self.ignored_topics = {t.strip().lower() for t in self.settings.ignored_topics if t.strip()}

    def should_ignore_author(self, author: str) -> bool:
        """Return True if *author* matches bot patterns or ignored author list."""
        if not author:
            return False
        clean = author.strip().lower()

        if clean in self.ignored_authors:
            return True

        if self.settings.ignore_bots:
            for pat in DEFAULT_BOT_PATTERNS:
                if fnmatch.fnmatch(clean, pat):
                    return True

        return False

    def should_ignore_channel(self, channel: str) -> bool:
        """Return True if *channel* matches any configured ignored channel."""
        if not channel:
            return False
        clean = channel.strip().lower().lstrip("#")
        return any(fnmatch.fnmatch(clean, ignored.lstrip("#")) for ignored in self.ignored_channels)

    def should_ignore_topic(self, topic_or_tag: str) -> bool:
        """Return True if *topic_or_tag* matches any configured ignored topic or tag."""
        if not topic_or_tag:
            return False
        clean = topic_or_tag.strip().lower()
        return any(fnmatch.fnmatch(clean, ignored) for ignored in self.ignored_topics)
