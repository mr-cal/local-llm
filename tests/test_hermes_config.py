"""Tests for HermesSettings and hermes config template section."""

from __future__ import annotations

import tomllib
from pathlib import Path

from llm.config import HermesSettings, Settings


class TestHermesSettings:
    def test_defaults(self):
        h = HermesSettings()
        assert h.openrouter_key == ""
        assert h.telegram_token == ""
        assert h.telegram_allowed_users == ""
        assert h.github_token == ""

    def test_custom_values(self):
        h = HermesSettings(
            openrouter_key="sk-or-v1-test",
            telegram_token="123:ABC",
            telegram_allowed_users="987654321",
            github_token="ghp_test",
        )
        assert h.openrouter_key == "sk-or-v1-test"
        assert h.telegram_token == "123:ABC"
        assert h.telegram_allowed_users == "987654321"
        assert h.github_token == "ghp_test"

    def test_has_openrouter_false_by_default(self):
        assert HermesSettings().has_openrouter() is False

    def test_has_openrouter_true_with_key(self):
        assert HermesSettings(openrouter_key="sk-or-v1-abc").has_openrouter() is True

    def test_has_openrouter_false_whitespace_only(self):
        assert HermesSettings(openrouter_key="   ").has_openrouter() is False

    def test_has_telegram_false_by_default(self):
        assert HermesSettings().has_telegram() is False

    def test_has_telegram_false_token_only(self):
        assert HermesSettings(telegram_token="123:ABC").has_telegram() is False

    def test_has_telegram_false_users_only(self):
        assert HermesSettings(telegram_allowed_users="987654321").has_telegram() is False

    def test_has_telegram_true_with_both(self):
        h = HermesSettings(telegram_token="123:ABC", telegram_allowed_users="987654321")
        assert h.has_telegram() is True


class TestSettingsHermes:
    def test_hermes_default_on_settings(self):
        s = Settings()
        assert isinstance(s.hermes, HermesSettings)
        assert s.hermes.openrouter_key == ""

    def test_hermes_custom_via_settings(self):
        s = Settings(hermes=HermesSettings(openrouter_key="sk-or-v1-xyz"))
        assert s.hermes.openrouter_key == "sk-or-v1-xyz"


class TestHermesConfigTemplate:
    def test_template_has_hermes_section(self):
        template_path = Path(__file__).parent.parent / "src" / "llm" / "config_template.toml"
        data = tomllib.loads(template_path.read_text())
        assert "hermes" in data, "[hermes] section missing from config_template.toml"

    def test_template_hermes_has_required_keys(self):
        template_path = Path(__file__).parent.parent / "src" / "llm" / "config_template.toml"
        data = tomllib.loads(template_path.read_text())
        hermes = data["hermes"]
        assert "openrouter_key" in hermes
        assert "telegram_token" in hermes
        assert "telegram_allowed_users" in hermes
        assert "github_token" in hermes

    def test_template_hermes_defaults_are_empty(self):
        template_path = Path(__file__).parent.parent / "src" / "llm" / "config_template.toml"
        data = tomllib.loads(template_path.read_text())
        hermes = data["hermes"]
        assert hermes["openrouter_key"] == ""
        assert hermes["telegram_token"] == ""
        assert hermes["telegram_allowed_users"] == ""
        assert hermes["github_token"] == ""
