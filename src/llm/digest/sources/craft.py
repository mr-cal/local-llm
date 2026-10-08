"""Read-only inspection and reference grounding for craft repositories in ~/dev/craft/."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


class CraftInspector:
    """Read-only code search and reference validator across craft repos."""

    def __init__(self, base_dir: Path | str = "~/dev/craft") -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()

    def exists(self) -> bool:
        return self.base_dir.exists() and self.base_dir.is_dir()

    def list_repos(self) -> list[str]:
        """List subdirectories in craft_dir that are git repositories."""
        if not self.exists():
            return []
        repos = []
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and (entry / ".git").exists():
                repos.append(entry.name)
            elif entry.is_dir():
                # Check nested repo directories like snapcraft/snapcraft-a
                for sub in entry.iterdir():
                    if sub.is_dir() and (sub / ".git").exists():
                        repos.append(f"{entry.name}/{sub.name}")
        return sorted(repos)

    def search_code(self, pattern: str, max_results: int = 20) -> list[dict[str, Any]]:
        """Ripgrep search across all craft repositories (read-only)."""
        if not self.exists() or not pattern.strip():
            return []

        # Run ripgrep or grep
        cmd = [
            "rg",
            "--no-heading",
            "--line-number",
            "--color=never",
            "--max-count",
            "5",
            "--max-filesize",
            "1M",
            pattern,
            str(self.base_dir),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines = res.stdout.splitlines()[:max_results]
        except Exception:
            return []

        matches = []
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) == 3:
                file_path, line_no, content = parts
                rel_path = os.path.relpath(file_path, str(self.base_dir))
                matches.append(
                    {
                        "file": rel_path,
                        "line": int(line_no) if line_no.isdigit() else 0,
                        "content": content.strip(),
                    }
                )
        return matches

    def validate_reference(self, symbol_or_key: str) -> dict[str, Any]:
        """Validate whether a term (e.g. 'snapd-info', function name, error string) exists in craft repos."""
        results = self.search_code(symbol_or_key, max_results=10)
        found = len(results) > 0
        repo_names = sorted({r["file"].split("/")[0] for r in results})
        return {
            "query": symbol_or_key,
            "found": found,
            "occurrences": len(results),
            "repos": repo_names,
            "matches": results,
        }
