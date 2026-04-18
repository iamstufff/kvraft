"""Live checks that the wrapper produces the expected MiniLM output.

First run downloads ``sentence-transformers/all-MiniLM-L6-v2`` (~90MB) into
the HuggingFace cache; subsequent runs read from disk.
"""

from collections.abc import Iterator

import numpy as np
import pytest

from src.cache import embedding

pytestmark = pytest.mark.integration

MINILM_DIM = 384


@pytest.fixture(autouse=True)
def _reset_model_cache() -> Iterator[None]:
    embedding._load_model.cache_clear()
    yield
    embedding._load_model.cache_clear()


def test_embed_returns_minilm_dimensionality() -> None:
    vector = embedding.embed("kvraft is a distributed semantic cache")
    assert vector.shape == (MINILM_DIM,)
    assert vector.dtype == np.float32


def test_embed_is_deterministic() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    first = embedding.embed(text)
    second = embedding.embed(text)
    np.testing.assert_array_equal(first, second)


def test_embeddings_are_unit_norm() -> None:
    vector = embedding.embed("normalization smoke test")
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
