"""Scheduling and lookback window calculations for daily digests.

Enforces weekday-only execution with two slots:
- Slot.MORNING (5:30 AM target):
    - Monday: looks back to Friday 14:00 (covers weekend, ~63.5 hours).
    - Tuesday–Friday: looks back to yesterday 14:00 (15.5 hours).
- Slot.AFTERNOON (2:00 PM target):
    - Monday–Friday: looks back to today 05:30 (8.5 hours).
- Saturday / Sunday: execution is a no-op / skipped.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum


class Slot(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"


def is_weekday(date: dt.date | dt.datetime) -> bool:
    """Return True if *date* is Monday (0) through Friday (4)."""
    return date.weekday() < 5


def get_lookback_window(
    now: dt.datetime | None = None,
    slot: Slot = Slot.MORNING,
) -> tuple[dt.datetime, dt.datetime]:
    """Calculate the [start_time, end_time] lookback window for *slot*.

    Args:
        now: Reference datetime (defaults to current UTC datetime).
        slot: Slot.MORNING or Slot.AFTERNOON.

    Returns:
        tuple[start_time, end_time] in the same timezone (or UTC).

    Raises:
        ValueError: If called on a weekend when weekday-only is enforced.
    """
    if now is None:
        now = dt.datetime.now(dt.UTC)

    tz = now.tzinfo or dt.UTC

    # Weekday check: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    weekday = now.weekday()
    if weekday >= 5:
        raise ValueError(f"Digests only run on weekdays. Current day is {now.strftime('%A')} ({now.date()}).")

    today = now.date()

    if slot == Slot.MORNING:
        # Target end time: today at 05:30:00
        end_time = dt.datetime.combine(today, dt.time(5, 30, 0), tzinfo=tz)
        if weekday == 0:
            # Monday: look back to Friday 14:00:00 (3 days ago)
            friday = today - dt.timedelta(days=3)
            start_time = dt.datetime.combine(friday, dt.time(14, 0, 0), tzinfo=tz)
        else:
            # Tuesday-Friday: look back to yesterday 14:00:00
            yesterday = today - dt.timedelta(days=1)
            start_time = dt.datetime.combine(yesterday, dt.time(14, 0, 0), tzinfo=tz)

    elif slot == Slot.AFTERNOON:
        # Afternoon window: today 05:30:00 to today 14:00:00
        start_time = dt.datetime.combine(today, dt.time(5, 30, 0), tzinfo=tz)
        end_time = dt.datetime.combine(today, dt.time(14, 0, 0), tzinfo=tz)

    else:
        raise ValueError(f"Unknown slot: {slot!r}")

    return start_time, end_time


def get_active_sources_for_slot(slot: Slot) -> list[str]:
    """Return the list of source identifiers included in *slot*."""
    if slot == Slot.MORNING:
        return ["discourse", "github", "mattermost", "matrix", "google", "jira"]
    if slot == Slot.AFTERNOON:
        # Discourse is omitted in the afternoon slot per specification
        return ["github", "mattermost", "matrix", "google", "jira"]
    return []
