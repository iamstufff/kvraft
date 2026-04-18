from collections.abc import Iterator
from pathlib import Path

import pytest

from src.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_from_repo_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure tests don't pick up the repo root's `.env` via pydantic-settings.

    Pydantic loads `.env` relative to CWD; chdir'ing to a tmpdir makes that a
    no-op. Tests that need specific env vars use `monkeypatch.setenv` directly.
    """

    monkeypatch.chdir(tmp_path)
