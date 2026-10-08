"""Unit tests for craft repository inspection."""

from __future__ import annotations

import subprocess

from llm.digest.sources.craft import CraftInspector


def test_craft_inspector_list_repos(tmp_path):
    # Set up mock repos
    repo1 = tmp_path / "snapcraft"
    repo1.mkdir()
    (repo1 / ".git").mkdir()

    repo2 = tmp_path / "rockcraft"
    repo2.mkdir()
    (repo2 / ".git").mkdir()

    non_repo = tmp_path / "other"
    non_repo.mkdir()

    inspector = CraftInspector(base_dir=tmp_path)
    assert inspector.exists() is True
    repos = inspector.list_repos()
    assert repos == ["rockcraft", "snapcraft"]


def test_craft_inspector_validate_reference(monkeypatch, tmp_path):
    inspector = CraftInspector(base_dir=tmp_path)

    # Mock subprocess.run for rg
    def fake_run(cmd, **kwargs):
        stdout = f"{tmp_path}/snapcraft/parts/info.py:42:snapd_info_key = 'snapd-info'\n"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ref = inspector.validate_reference("snapd-info")
    assert ref["found"] is True
    assert ref["occurrences"] == 1
    assert "snapcraft" in ref["repos"]
    assert ref["matches"][0]["line"] == 42
