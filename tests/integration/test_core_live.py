"""End-to-end semantic cache checks against the real MiniLM model."""

from collections.abc import Iterator

import pytest

from src.cache import embedding
from src.cache.core import Hit, Miss, SemanticCache

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_model_cache() -> Iterator[None]:
    embedding._load_model.cache_clear()
    yield
    embedding._load_model.cache_clear()


def test_semantically_similar_prompts_hit_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.7")

    cache = SemanticCache()
    provider_response = "Use sorted() or list.sort()."

    miss = cache.get_or_miss("how do I sort a list in Python")
    assert isinstance(miss, Miss)
    cache.put("how do I sort a list in Python", provider_response, embedding=miss.embedding)

    hit = cache.get_or_miss("how can I sort a python list")
    assert isinstance(hit, Hit)
    assert hit.value == provider_response
    assert hit.similarity >= 0.7


def test_semantically_different_prompts_miss_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")

    cache = SemanticCache()
    cache.put("how do I sort a list in Python", "Use sorted() or list.sort().")

    result = cache.get_or_miss("what is the capital of France")
    assert isinstance(result, Miss)
