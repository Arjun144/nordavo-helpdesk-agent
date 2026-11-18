from __future__ import annotations

import pytest

from helpdesk.config import Config, ConfigError


def test_reads_config_from_environment() -> None:
    config = Config.from_env(
        {
            "HELPDESK_API_KEY": "secret",
            "HELPDESK_CANDIDATE_EMAIL": "me@example.com",
            "HELPDESK_CANDIDATE_NAME": "Arjun",
        }
    )

    assert config.api_key == "secret"
    assert config.candidate_email == "me@example.com"
    assert config.candidate_name == "Arjun"
    assert config.max_requests_per_second == 4.0
    assert config.base_url.startswith("https://nakywgkztvzkihdjnwbc.supabase.co/")


def test_candidate_name_is_optional() -> None:
    config = Config.from_env(
        {"HELPDESK_API_KEY": "secret", "HELPDESK_CANDIDATE_EMAIL": "me@example.com"}
    )

    assert config.candidate_name is None


def test_missing_credentials_are_named() -> None:
    with pytest.raises(ConfigError) as excinfo:
        Config.from_env({})

    message = str(excinfo.value)
    assert "HELPDESK_API_KEY" in message
    assert "HELPDESK_CANDIDATE_EMAIL" in message


def test_email_must_look_like_an_email() -> None:
    with pytest.raises(ConfigError, match="applied with"):
        Config.from_env({"HELPDESK_API_KEY": "secret", "HELPDESK_CANDIDATE_EMAIL": "nope"})


def test_rate_limit_cannot_exceed_the_api_ceiling() -> None:
    with pytest.raises(ConfigError, match="at most 5"):
        Config.from_env(
            {
                "HELPDESK_API_KEY": "secret",
                "HELPDESK_CANDIDATE_EMAIL": "me@example.com",
                "HELPDESK_MAX_RPS": "10",
            }
        )


def test_api_key_is_hidden_from_repr() -> None:
    config = Config.from_env(
        {"HELPDESK_API_KEY": "super-secret", "HELPDESK_CANDIDATE_EMAIL": "me@example.com"}
    )

    assert "super-secret" not in repr(config)


def test_dotenv_in_the_working_directory_is_loaded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "HELPDESK_API_KEY=from-dotenv\n"
        "HELPDESK_CANDIDATE_EMAIL=dotenv@example.com\n"
    )
    monkeypatch.chdir(tmp_path)
    for name in ("HELPDESK_API_KEY", "HELPDESK_CANDIDATE_EMAIL", "HELPDESK_CANDIDATE_NAME"):
        monkeypatch.delenv(name, raising=False)

    config = Config.from_env()

    assert config.api_key == "from-dotenv"
    assert config.candidate_email == "dotenv@example.com"


def test_process_environment_beats_the_dotenv_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "HELPDESK_API_KEY=from-dotenv\n"
        "HELPDESK_CANDIDATE_EMAIL=dotenv@example.com\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HELPDESK_API_KEY", "from-shell")

    assert Config.from_env().api_key == "from-shell"
