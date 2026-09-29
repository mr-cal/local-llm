"""Finding, reading and writing config.toml."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import typer

from llm.core.console import console
from llm.core.files import write_atomic
from llm.settings.models import CONFIG_FILENAME, LxdSettings, Settings

# Template written by `llm config init` - loaded from config_template.toml.
# Every option is a commented example so the file is self-documenting.
CONFIG_TEMPLATE = resources.files("llm").joinpath("config_template.toml").read_text(encoding="utf-8")


def find_config() -> Path:
    """Walk up from CWD to find config.toml."""
    here = Path.cwd()
    for directory in [here, *here.parents]:
        candidate = directory / CONFIG_FILENAME
        if candidate.exists():
            return candidate
        if (directory / "pyproject.toml").exists():
            break  # stop at project root even if config.toml is missing
    return Path.cwd() / CONFIG_FILENAME


def load_config() -> Settings:
    """Load settings from config.toml. Raises SystemExit with a helpful message if not found."""
    config_path = find_config()
    if not config_path.exists():
        console.print(
            f"[red]Config file not found:[/red] {config_path}\n"
            "Run [bold]uv run llm config init[/bold] to create it."
        )
        raise typer.Exit(1)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    return Settings.model_validate(raw)


def try_load_lxd() -> LxdSettings | None:
    """Load only the [lxd] section from config.toml; return None if the file doesn't exist."""
    config_path = find_config()
    if not config_path.exists():
        return None
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    return LxdSettings.model_validate(raw.get("lxd", {}))


def write_config_toml(cfg_dict: dict, path: Path | None = None) -> Path:  # type: ignore[type-arg]
    """Write a config dict as TOML to *path* (default: CWD/config.toml).

    Returns the path written to.
    """
    import tomli_w  # noqa: PLC0415

    if path is None:
        path = Path.cwd() / CONFIG_FILENAME
    write_atomic(path, tomli_w.dumps(cfg_dict))
    return path
