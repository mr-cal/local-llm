"""Unit tests for digest noise and bot filtering."""

from __future__ import annotations

from llm.digest.filter import DigestFilter
from llm.settings.models import DigestFilterSettings


def test_filter_default_bot_detection():
    filt = DigestFilter(DigestFilterSettings(ignore_bots=True))

    assert filt.should_ignore_author("github-actions[bot]") is True
    assert filt.should_ignore_author("jenkins-ci") is True
    assert filt.should_ignore_author("renovate[bot]") is True
    assert filt.should_ignore_author("my-custom-bot") is True
    assert filt.should_ignore_author("alice") is False
    assert filt.should_ignore_author("") is False


def test_filter_disabled_bot_detection():
    filt = DigestFilter(DigestFilterSettings(ignore_bots=False))

    assert filt.should_ignore_author("github-actions[bot]") is False
    assert filt.should_ignore_author("jenkins-ci") is False


def test_filter_custom_ignored_authors():
    filt = DigestFilter(
        DigestFilterSettings(
            ignore_bots=False,
            ignored_authors=["noisy_user", "spammer@example.com"],
        )
    )

    assert filt.should_ignore_author("noisy_user") is True
    assert filt.should_ignore_author("SPAMMER@EXAMPLE.COM") is True
    assert filt.should_ignore_author("alice") is False


def test_filter_ignored_channels():
    filt = DigestFilter(
        DigestFilterSettings(
            ignored_channels=["#ci-builds", "release-*", "random"],
        )
    )

    assert filt.should_ignore_channel("ci-builds") is True
    assert filt.should_ignore_channel("#ci-builds") is True
    assert filt.should_ignore_channel("release-announcements") is True
    assert filt.should_ignore_channel("general") is False
    assert filt.should_ignore_channel("") is False


def test_filter_ignored_topics():
    filt = DigestFilter(
        DigestFilterSettings(
            ignored_topics=["automated-*", "release-notes"],
        )
    )

    assert filt.should_ignore_topic("automated-test-run") is True
    assert filt.should_ignore_topic("Release-Notes") is True
    assert filt.should_ignore_topic("feature-proposal") is False
    assert filt.should_ignore_topic("") is False
