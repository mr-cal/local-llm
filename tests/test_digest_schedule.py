"""Unit tests for digest scheduling and time window calculations."""

from __future__ import annotations

import datetime as dt

import pytest

from llm.digest.schedule import Slot, get_lookback_window, is_weekday


def test_is_weekday():
    # 2025-05-12 is Monday
    assert is_weekday(dt.date(2025, 5, 12)) is True
    # 2025-05-16 is Friday
    assert is_weekday(dt.date(2025, 5, 16)) is True
    # 2025-05-17 is Saturday
    assert is_weekday(dt.date(2025, 5, 17)) is False
    # 2025-05-18 is Sunday
    assert is_weekday(dt.date(2025, 5, 18)) is False


def test_weekend_raises_error():
    # Saturday
    saturday = dt.datetime(2025, 5, 17, 10, 0, 0, tzinfo=dt.UTC)
    with pytest.raises(ValueError, match="Digests only run on weekdays"):
        get_lookback_window(now=saturday, slot=Slot.MORNING)

    # Sunday
    sunday = dt.datetime(2025, 5, 18, 10, 0, 0, tzinfo=dt.UTC)
    with pytest.raises(ValueError, match="Digests only run on weekdays"):
        get_lookback_window(now=sunday, slot=Slot.AFTERNOON)


def test_morning_slot_on_monday():
    # Monday 2025-05-12 at 05:30
    monday = dt.datetime(2025, 5, 12, 5, 30, 0, tzinfo=dt.UTC)
    start_time, end_time = get_lookback_window(now=monday, slot=Slot.MORNING)

    # Should start Friday 2025-05-09 at 14:00
    expected_start = dt.datetime(2025, 5, 9, 14, 0, 0, tzinfo=dt.UTC)
    expected_end = dt.datetime(2025, 5, 12, 5, 30, 0, tzinfo=dt.UTC)

    assert start_time == expected_start
    assert end_time == expected_end
    # Total hours: 63.5 hours
    duration = end_time - start_time
    assert duration.total_seconds() == 63.5 * 3600


def test_morning_slot_on_tuesday_through_friday():
    # Wednesday 2025-05-14 at 05:30
    wednesday = dt.datetime(2025, 5, 14, 5, 30, 0, tzinfo=dt.UTC)
    start_time, end_time = get_lookback_window(now=wednesday, slot=Slot.MORNING)

    # Should start Tuesday 2025-05-13 at 14:00
    expected_start = dt.datetime(2025, 5, 13, 14, 0, 0, tzinfo=dt.UTC)
    expected_end = dt.datetime(2025, 5, 14, 5, 30, 0, tzinfo=dt.UTC)

    assert start_time == expected_start
    assert end_time == expected_end
    duration = end_time - start_time
    assert duration.total_seconds() == 15.5 * 3600


def test_afternoon_slot_on_any_weekday():
    # Thursday 2025-05-15 at 14:00
    thursday = dt.datetime(2025, 5, 15, 14, 0, 0, tzinfo=dt.UTC)
    start_time, end_time = get_lookback_window(now=thursday, slot=Slot.AFTERNOON)

    # Should start Thursday 2025-05-15 at 05:30 and end at 14:00
    expected_start = dt.datetime(2025, 5, 15, 5, 30, 0, tzinfo=dt.UTC)
    expected_end = dt.datetime(2025, 5, 15, 14, 0, 0, tzinfo=dt.UTC)

    assert start_time == expected_start
    assert end_time == expected_end
    duration = end_time - start_time
    assert duration.total_seconds() == 8.5 * 3600
