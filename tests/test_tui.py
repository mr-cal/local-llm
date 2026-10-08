"""Unit tests for the Jotter Textual TUI."""

from __future__ import annotations

import pytest

from llm.tui.app import JotterApp


@pytest.mark.anyio
async def test_tui_app_mount():
    app = JotterApp()
    async with app.run_test() as pilot:
        # Check tabs exist
        tabbed_content = pilot.app.query_one("TabbedContent")
        assert tabbed_content is not None

        # Check tabs
        tab_ids = [pane.id for pane in pilot.app.query("TabPane")]
        assert "tab-digests" in tab_ids
        assert "tab-search" in tab_ids
        assert "tab-inspect" in tab_ids
        assert "tab-status" in tab_ids
