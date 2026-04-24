"""LRU eviction behavior on the non-Raft ``SemanticCache``."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.core import Hit, SemanticCache
from src.cache.index import SemanticIndex


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


# Each prompt maps to a near-orthogonal vector so its nearest-neighbor is itself.
VECTORS: dict[str, NDArray[np.float32]] = {
    "a": _unit([1.0, 0.0, 0.0, 0.0, 0.0]),
    "b": _unit([0.0, 1.0, 0.0, 0.0, 0.0]),
    "c": _unit([0.0, 0.0, 1.0, 0.0, 0.0]),
    "d": _unit([0.0, 0.0, 0.0, 1.0, 0.0]),
    "e": _unit([0.0, 0.0, 0.0, 0.0, 1.0]),
}


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> SemanticCache:
    monkeypatch.setenv("MAX_CAPACITY", "3")
    monkeypatch.setenv("REBUILD_THRESHOLD", "0.3")
    monkeypatch.setattr("src.cache.core.embed", lambda prompt: VECTORS[prompt])
    return SemanticCache(index=SemanticIndex(dim=5, initial_capacity=8))


def test_eviction_drops_oldest_when_capacity_reached(cache: SemanticCache) -> None:
    cache.put("a", "A")
    cache.put("b", "B")
    cache.put("c", "C")
    cache.put("d", "D")  # should evict "a"

    assert cache.size == 3
    assert isinstance(cache.get_or_miss("b"), Hit)
    assert isinstance(cache.get_or_miss("c"), Hit)
    assert isinstance(cache.get_or_miss("d"), Hit)
    # "a" is soft-deleted; a lookup on its vector no longer hits.
    assert not isinstance(cache.get_or_miss("a"), Hit)


def test_read_hit_touches_lru_order(cache: SemanticCache) -> None:
    cache.put("a", "A")
    cache.put("b", "B")
    cache.put("c", "C")

    # Touch "a": it should now be most-recent, so "b" becomes the victim.
    assert isinstance(cache.get_or_miss("a"), Hit)
    cache.put("d", "D")

    assert isinstance(cache.get_or_miss("a"), Hit)
    assert isinstance(cache.get_or_miss("c"), Hit)
    assert isinstance(cache.get_or_miss("d"), Hit)
    assert not isinstance(cache.get_or_miss("b"), Hit)


def test_rebuild_keeps_soft_deleted_bounded(cache: SemanticCache) -> None:
    # Fill past capacity repeatedly; rebuilds should keep the soft-delete
    # ratio from running away, and live keys must still equal MAX_CAPACITY.
    for prompt in ("a", "b", "c", "d", "e"):
        cache.put(prompt, prompt.upper())
    assert cache.size == 3
    # With threshold 0.3 and cap 3, each eviction raises ratio to 1/3=0.33 >
    # threshold, so a rebuild fires every time → soft count stays at 0.
    assert cache.soft_deleted_count == 0
    # The three most-recent entries survive.
    assert isinstance(cache.get_or_miss("c"), Hit)
    assert isinstance(cache.get_or_miss("d"), Hit)
    assert isinstance(cache.get_or_miss("e"), Hit)
