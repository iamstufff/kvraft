from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.cache import embedding


@pytest.fixture(autouse=True)
def _reset_model_cache() -> Iterator[None]:
    embedding._load_model.cache_clear()
    yield
    embedding._load_model.cache_clear()


def test_embed_returns_encoder_output_as_float32(monkeypatch) -> None:
    encoded = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    fake_model = MagicMock()
    fake_model.encode.return_value = encoded
    fake_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(embedding, "SentenceTransformer", fake_ctor)

    result = embedding.embed("hello world")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, encoded.astype(np.float32))
    fake_model.encode.assert_called_once_with(
        "hello world",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_uses_configured_model(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros(4, dtype=np.float32)
    fake_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(embedding, "SentenceTransformer", fake_ctor)

    embedding.embed("anything")

    fake_ctor.assert_called_once_with("custom/model")


def test_model_is_loaded_only_once(monkeypatch) -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros(4, dtype=np.float32)
    fake_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(embedding, "SentenceTransformer", fake_ctor)

    embedding.embed("a")
    embedding.embed("b")
    embedding.embed("c")

    assert fake_ctor.call_count == 1
