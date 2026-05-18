import time

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.core import Hit, Miss, SemanticCache
from src.cache.index import SemanticIndex


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def small_index() -> SemanticIndex:
    return SemanticIndex(dim=4, initial_capacity=16)


def test_first_call_returns_miss_with_embedding(
    monkeypatch: pytest.MonkeyPatch, small_index: SemanticIndex
) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr("src.cache.core.embed", lambda _prompt: vec)
    cache = SemanticCache(index=small_index)

    result = cache.get_or_miss("hello")

    assert isinstance(result, Miss)
    assert result.prompt == "hello"
    np.testing.assert_array_equal(result.embedding, vec)
    assert cache.size == 0


def test_put_then_same_prompt_is_hit(
    monkeypatch: pytest.MonkeyPatch, small_index: SemanticIndex
) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr("src.cache.core.embed", lambda _prompt: vec)
    cache = SemanticCache(index=small_index)

    cache.put("question", "answer")
    result = cache.get_or_miss("question")

    assert isinstance(result, Hit)
    assert result.value == "answer"
    assert result.similarity == pytest.approx(1.0, abs=1e-5)


def test_different_prompt_below_threshold_is_miss(
    monkeypatch: pytest.MonkeyPatch, small_index: SemanticIndex
) -> None:
    vectors = {
        "stored": _unit([1.0, 0.0, 0.0, 0.0]),
        "far": _unit([0.0, 1.0, 0.0, 0.0]),
    }
    monkeypatch.setattr("src.cache.core.embed", lambda prompt: vectors[prompt])
    cache = SemanticCache(index=small_index)

    cache.put("stored", "answer")
    result = cache.get_or_miss("far")

    assert isinstance(result, Miss)
    assert result.prompt == "far"


def test_put_accepts_precomputed_embedding(
    monkeypatch: pytest.MonkeyPatch, small_index: SemanticIndex
) -> None:
    call_count = {"n": 0}

    def _counting_embed(_prompt: str) -> NDArray[np.float32]:
        call_count["n"] += 1
        return _unit([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr("src.cache.core.embed", _counting_embed)
    cache = SemanticCache(index=small_index)

    miss = cache.get_or_miss("q")
    assert isinstance(miss, Miss)
    cache.put("q", "v", embedding=miss.embedding)

    assert call_count["n"] == 1


def test_size_reflects_cache_growth(
    monkeypatch: pytest.MonkeyPatch, small_index: SemanticIndex
) -> None:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0, 0.0]),
        "b": _unit([0.0, 1.0, 0.0, 0.0]),
    }
    monkeypatch.setattr("src.cache.core.embed", lambda prompt: vectors[prompt])
    cache = SemanticCache(index=small_index)

    assert cache.size == 0
    cache.put("a", "A")
    cache.put("b", "B")
    assert cache.size == 2


def test_ttl_expired_entry_returns_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0.5")

    cache = SemanticCache()
    cache.put("prompt", "value")
    first = cache.get_or_miss("prompt")
    assert isinstance(first, Hit)

    time.sleep(0.7)
    second = cache.get_or_miss("prompt")
    assert isinstance(second, Miss)


def test_ttl_zero_disables_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")

    cache = SemanticCache()
    cache.put("prompt", "value")
    time.sleep(0.2)
    assert isinstance(cache.get_or_miss("prompt"), Hit)
