"""Provider API key values must never appear in CLI error output."""

import pytest

from image_creator_tool import cli
from image_creator_tool.errors import PermanentAPIError
from image_creator_tool.redaction import sanitize_error


def test_sanitize_error_redacts_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret-value")
    out = sanitize_error("HTTP 401: Authorization Bearer or-secret-value")
    assert "or-secret-value" not in out
    assert "REDACTED" in out


def test_sanitize_error_redacts_gemini_default_provider_key(monkeypatch):
    # gemini is the default provider; its key must be covered too.
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    out = sanitize_error("request to https://x/?key=gemini-secret failed")
    assert "gemini-secret" not in out
    assert "REDACTED" in out


def test_sanitize_error_redacts_config_profile_key(monkeypatch, tmp_path):
    # A key present only in a [profile.*] api_key (never exported to env) must
    # still be redacted on the error path.
    for var in (
        "OPENROUTER_API_KEY",
        "DEEPINFRA_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    config = tmp_path / "config.toml"
    config.write_text(
        "[profile.openrouter]\n"
        'provider = "openrouter"\n'
        'api_key = "profile-only-secret"\n'  # pragma: allowlist secret
    )
    monkeypatch.setattr("image_creator_tool.config.CONFIG_FILE", config)
    out = sanitize_error("request failed with api_key=profile-only-secret")
    assert "profile-only-secret" not in out
    assert "REDACTED" in out


def test_sanitize_error_passthrough_when_no_secret(monkeypatch, tmp_path):
    for var in (
        "OPENROUTER_API_KEY",
        "DEEPINFRA_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("image_creator_tool.config.CONFIG_FILE", tmp_path / "absent.toml")
    assert sanitize_error("just a normal message") == "just a normal message"


def test_main_redacts_key_on_known_error(monkeypatch, capsys):
    # A provider key embedded in an ImageCreatorError (e.g. an upstream response
    # body echoing the request URL) must be redacted before reaching stderr.
    monkeypatch.setenv("OPENAI_API_KEY", "leaky-openai-key")

    def boom():
        raise PermanentAPIError(
            "HTTP 401: url https://api.openai.com/v1?api_key=leaky-openai-key"
        )

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == cli.ExitCode.GENERAL_ERROR
    captured = capsys.readouterr()
    assert "leaky-openai-key" not in captured.err
    assert "REDACTED" in captured.err


def test_main_redacts_key_on_unexpected_error(monkeypatch, capsys):
    # The added catch-all must redact too: an unexpected exception's raw string
    # can embed a key (e.g. an Authorization header in a traceback message).
    monkeypatch.setenv("DEEPINFRA_API_KEY", "leaky-di-key")

    def boom():
        raise RuntimeError("connection reset; sent Authorization: Bearer leaky-di-key")

    monkeypatch.setattr(cli, "app", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == cli.ExitCode.GENERAL_ERROR
    captured = capsys.readouterr()
    assert "leaky-di-key" not in captured.err
