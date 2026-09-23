"""Configuration tests.

The main job here is to prove that .env.example and common/config.py agree.
A missing variable in .env.example is the classic "works on my machine" bug in
a Docker Compose project, and it is cheap to rule out.
"""

from __future__ import annotations

import re
from pathlib import Path

from common.config import REPO_ROOT, Settings, get_settings

ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _env_example_keys() -> set[str]:
    keys = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", line.strip())
        if match:
            keys.add(match.group(1).lower())
    return keys


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.exists()


def test_every_setting_is_documented_in_env_example() -> None:
    """Each field in Settings must have a corresponding line in .env.example."""
    documented = _env_example_keys()
    missing = set(Settings.model_fields) - documented

    assert not missing, f"undocumented in .env.example: {sorted(missing)}"


def test_env_example_has_no_unused_keys() -> None:
    """And nothing in .env.example should be dead configuration."""
    # Compose-only variables that are read by docker-compose.yml, not by Python.
    compose_only = {
        "kafka_bootstrap_servers_host",
        "superset_admin",
        "superset_password",
        "superset_secret_key",
        "spark_mem_limit",
    }
    unused = _env_example_keys() - set(Settings.model_fields) - compose_only

    assert not unused, f"in .env.example but not in Settings: {sorted(unused)}"


def test_settings_load_from_env_example(monkeypatch) -> None:
    """The shipped example must produce a valid configuration."""
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.kafka_topic_payments == "payments"
    assert 0.0 <= settings.threshold_review < settings.threshold_block <= 1.0
    assert settings.ollama_model == "llama3.2:3b"


def test_example_salt_is_reported_as_not_configured() -> None:
    """A demo must never believe it has masked PII when it has not."""
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.uses_default_pii_salt is True
    assert Settings(pii_hash_salt="a" * 64).uses_default_pii_salt is False


def test_path_resolves_relative_to_repo_root() -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.path("data/delta/bronze") == REPO_ROOT / "data" / "delta" / "bronze"
    assert settings.path("/tmp/x") == Path("/tmp/x")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
