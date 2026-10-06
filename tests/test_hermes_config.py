"""Tests for HermesSettings and hermes config template section."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from llm.settings import HermesSettings, Settings


class TestHermesSettings:
    def test_defaults(self):
        h = HermesSettings()
        assert h.provider == "local-llm"
        assert h.timezone == ""
        assert h.openrouter_key == ""
        assert h.telegram_token == ""
        assert h.telegram_allowed_users == ""
        assert h.github_token == ""
        assert h.mattermost_url == ""
        assert h.mattermost_token == ""
        assert h.mattermost_team == "canonical"
        assert h.max_concurrent_sessions == 1
        assert h.max_concurrent_children == 1
        assert h.approval_timeout == 3600

    def test_custom_values(self):
        h = HermesSettings(
            timezone="America/New_York",
            provider="openrouter",
            openrouter_key="sk-or-v1-test",
            telegram_token="123:ABC",
            telegram_allowed_users="987654321",
            github_token="ghp_test",
            mattermost_url="https://mm.example.com",
            mattermost_token="mm-token-123",
            mattermost_team="my-team",
            max_concurrent_sessions=2,
            max_concurrent_children=4,
            approval_timeout=1800,
        )
        assert h.timezone == "America/New_York"
        assert h.provider == "openrouter"
        assert h.openrouter_key == "sk-or-v1-test"
        assert h.telegram_token == "123:ABC"
        assert h.telegram_allowed_users == "987654321"
        assert h.github_token == "ghp_test"
        assert h.mattermost_url == "https://mm.example.com"
        assert h.mattermost_token == "mm-token-123"
        assert h.mattermost_team == "my-team"
        assert h.max_concurrent_sessions == 2
        assert h.max_concurrent_children == 4
        assert h.approval_timeout == 1800

    def test_timezone_validation(self):
        h = HermesSettings(timezone="America/New_York")
        assert h.timezone == "America/New_York"
        h_empty = HermesSettings(timezone="")
        assert h_empty.timezone == ""

    def test_unrecognized_timezone_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HermesSettings(timezone="US/chicago")
        with pytest.raises(ValidationError):
            HermesSettings(timezone="Invalid/Timezone")

    def test_effective_timezone_defaults_to_chicago(self):
        h = HermesSettings()
        assert h.effective_timezone() == "America/Chicago"

    def test_effective_timezone_uses_all_cfg_lxd(self):
        from llm.settings import LxdSettings

        h = HermesSettings()
        all_cfg = Settings(lxd=LxdSettings(timezone="America/Denver"))
        assert h.effective_timezone(all_cfg) == "America/Denver"

    def test_effective_timezone_hermes_override_wins(self):
        from llm.settings import LxdSettings

        h = HermesSettings(timezone="America/New_York")
        all_cfg = Settings(lxd=LxdSettings(timezone="America/Denver"))
        assert h.effective_timezone(all_cfg) == "America/New_York"

    def test_has_openrouter_false_by_default(self):
        assert HermesSettings().has_openrouter() is False

    def test_has_openrouter_false_when_local(self):
        assert HermesSettings(provider="local-llm").has_openrouter() is False

    def test_has_openrouter_true_with_key(self):
        assert HermesSettings(provider="openrouter", openrouter_key="sk-or-v1-abc").has_openrouter() is True

    def test_has_openrouter_false_no_key(self):
        assert HermesSettings(provider="openrouter", openrouter_key="").has_openrouter() is False

    def test_has_openrouter_false_whitespace_only(self):
        assert HermesSettings(provider="openrouter", openrouter_key="   ").has_openrouter() is False

    def test_has_telegram_false_by_default(self):
        assert HermesSettings().has_telegram() is False

    def test_has_telegram_false_token_only(self):
        assert HermesSettings(telegram_token="123:ABC").has_telegram() is False

    def test_has_telegram_false_users_only(self):
        assert HermesSettings(telegram_allowed_users="987654321").has_telegram() is False

    def test_has_telegram_true_with_both(self):
        h = HermesSettings(telegram_token="123:ABC", telegram_allowed_users="987654321")
        assert h.has_telegram() is True

    def test_has_github_false_by_default(self):
        assert HermesSettings().has_github() is False

    def test_has_github_true_with_token(self):
        assert HermesSettings(github_token="ghp_test123").has_github() is True

    def test_has_github_false_whitespace_only(self):
        assert HermesSettings(github_token="   ").has_github() is False

    def test_has_mattermost_false_by_default(self):
        assert HermesSettings().has_mattermost() is False

    def test_has_mattermost_true_with_url_and_token(self):
        assert (
            HermesSettings(
                mattermost_url="https://mm.example.com", mattermost_token="mm-tok"
            ).has_mattermost()
            is True
        )

    def test_has_mattermost_false_missing_url(self):
        assert HermesSettings(mattermost_url="", mattermost_token="mm-tok").has_mattermost() is False

    def test_has_mattermost_false_missing_token(self):
        assert (
            HermesSettings(mattermost_url="https://mm.example.com", mattermost_token="").has_mattermost()
            is False
        )

    def test_has_mattermost_false_whitespace_only(self):
        assert HermesSettings(mattermost_url="   ", mattermost_token="   ").has_mattermost() is False

    def test_has_local_llm_true_by_default(self):
        assert HermesSettings().has_local_llm() is True

    def test_has_local_llm_true_explicit(self):
        assert HermesSettings(provider="local-llm").has_local_llm() is True

    def test_has_local_llm_false_when_openrouter(self):
        assert HermesSettings(provider="openrouter").has_local_llm() is False


class TestSettingsHermes:
    def test_hermes_default_on_settings(self):
        s = Settings()
        assert isinstance(s.hermes, HermesSettings)
        assert s.hermes.provider == "local-llm"
        assert s.hermes.openrouter_key == ""

    def test_hermes_custom_via_settings(self):
        s = Settings(hermes=HermesSettings(provider="openrouter", openrouter_key="sk-or-v1-xyz"))
        assert s.hermes.provider == "openrouter"
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
        assert "timezone" in hermes
        assert "provider" in hermes
        assert "openrouter_key" in hermes
        assert "telegram_token" in hermes
        assert "telegram_allowed_users" in hermes
        assert "github_token" in hermes
        assert "mattermost_url" in hermes
        assert "mattermost_token" in hermes
        assert "mattermost_team" in hermes
        assert "max_concurrent_sessions" in hermes
        assert "max_concurrent_children" in hermes
        assert "approval_timeout" in hermes

    def test_template_lxd_has_timezone(self):
        template_path = Path(__file__).parent.parent / "src" / "llm" / "config_template.toml"
        data = tomllib.loads(template_path.read_text())
        assert "timezone" in data["lxd"]
        assert data["lxd"]["timezone"] == "America/Chicago"

    def test_template_hermes_defaults_are_empty(self):
        template_path = Path(__file__).parent.parent / "src" / "llm" / "config_template.toml"
        data = tomllib.loads(template_path.read_text())
        hermes = data["hermes"]
        assert hermes["provider"] == "local-llm"
        assert hermes["openrouter_key"] == ""
        assert hermes["telegram_token"] == ""
        assert hermes["telegram_allowed_users"] == ""
        assert hermes["github_token"] == ""
        assert hermes["mattermost_url"] == ""
        assert hermes["mattermost_token"] == ""
