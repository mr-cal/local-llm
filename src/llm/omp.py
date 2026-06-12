"""Oh-my-pi (OMP) model configuration helpers.

Provides functions for building and parsing oh-my-pi models.yml files.
Uses PyYAML for robust YAML parsing (yaml.safe_load) and serialization
(yaml.dump) instead of hand-rolled parsing logic.

The models.yml format expected by oh-my-pi::

    providers:
      local-llm:
        baseUrl: https://...
        apiKey: ...
        api: openai-completions
        auth: apiKey
        models:
          - id: local
            name: MyModel
            reasoning: true
            input: [text]
            contextWindow: 4096
            maxTokens: 8192
            cost:
              input: 0.0
              output: 0.0
              cacheRead: 0.0
              cacheWrite: 0.0
"""

from __future__ import annotations

import yaml


def build_omp_yaml(cfg: dict) -> str:  # type: ignore[type-arg]
    """Build the oh-my-pi models.yml string from a config dict.

    Args:
        cfg: Configuration dict with ``providers.local-llm`` structure.

    Returns:
        YAML string suitable for writing to models.yml.
    """
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def parse_omp_yaml(text: str) -> dict:  # type: ignore[type-arg]
    """Parse oh-my-pi models.yml text into a dict.

    Uses yaml.safe_load() for robust parsing of the YAML structure,
    including nested providers and models.

    Args:
        text: YAML text to parse.

    Returns:
        Parsed dict, or empty dict if text is empty/whitespace-only.
    """
    if not text or not text.strip():
        return {}
    result = yaml.safe_load(text)
    if result is None:
        return {}
    if not isinstance(result, dict):
        return {}
    return result
