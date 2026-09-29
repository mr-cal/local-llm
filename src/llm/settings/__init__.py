"""Configuration schema and loading.

`llm.settings` is the single import point for configuration: the pydantic
models that define the schema, and the functions that find, load and write
config.toml.
"""

from __future__ import annotations

from llm.settings.loader import (
    CONFIG_TEMPLATE,
    find_config,
    load_config,
    try_load_lxd,
    write_config_toml,
)
from llm.settings.models import (
    BACKEND_FLAGS,
    CONFIG_FILENAME,
    SECRET,
    AuthSettings,
    BuildConfig,
    BuildProfile,
    ClientSettings,
    GitHubSettings,
    HermesSettings,
    LxdSettings,
    ModelCost,
    ModelEntry,
    ModelsSettings,
    MountEntry,
    ProxySettings,
    ServerSettings,
    Settings,
)

__all__ = [
    "BACKEND_FLAGS",
    "CONFIG_FILENAME",
    "CONFIG_TEMPLATE",
    "SECRET",
    "AuthSettings",
    "BuildConfig",
    "BuildProfile",
    "ClientSettings",
    "GitHubSettings",
    "HermesSettings",
    "LxdSettings",
    "ModelCost",
    "ModelEntry",
    "ModelsSettings",
    "MountEntry",
    "ProxySettings",
    "ServerSettings",
    "Settings",
    "find_config",
    "load_config",
    "try_load_lxd",
    "write_config_toml",
]
