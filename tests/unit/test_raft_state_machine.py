"""Unit tests for the pure-Python replicated-cache state machine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.core import Hit
from src.cache.index import SemanticIndex
from src.raft.state_machine import _CacheState


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def _state() -> _CacheState:
    return _CacheState(index=SemanticIndex(dim=4, initial_capacity=8))


@pytest.fixture
def state() -> _CacheState:
    return _state()


def test_apply_put_assigns_monotonic_ids(state: _CacheState) -> None:
    first = state.apply_put("one", _unit([1.0, 0.0, 0.0, 0.0]))
    second = state.apply_put("two", _unit([0.0, 1.0, 0.0, 0.0]))
    third = state.apply_put("three", _unit([0.0, 0.0, 1.0, 0.0]))

    assert (first, second, third) == (0, 1, 2)
    assert state.size == 3


def test_lookup_returns_hit_above_threshold(state: _CacheState) -> None:
    vector = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("answer", vector)

    result = state.lookup(vector, threshold=0.5)

    assert isinstance(result, Hit)
    assert result.value == "answer"
    assert result.similarity == pytest.approx(1.0)


def test_lookup_returns_none_on_empty_state(state: _CacheState) -> None:
    result = state.lookup(_unit([1.0, 0.0, 0.0, 0.0]), threshold=0.5)
    assert result is None


def test_lookup_returns_none_below_threshold(state: _CacheState) -> None:
    state.apply_put("a", _unit([1.0, 0.0, 0.0, 0.0]))
    # near-orthogonal vector: cosine ≈ 0
    result = state.lookup(_unit([0.0, 1.0, 0.0, 0.0]), threshold=0.5)
    assert result is None


def test_deterministic_replay_between_replicas() -> None:
    ops: list[tuple[str, NDArray[np.float32]]] = [
        ("one", _unit([1.0, 0.0, 0.0, 0.0])),
        ("two", _unit([0.0, 1.0, 0.0, 0.0])),
        ("three", _unit([0.0, 0.0, 1.0, 0.0])),
    ]

    leader = _state()
    follower = _state()
    for value, vec in ops:
        assert leader.apply_put(value, vec) == follower.apply_put(value, vec)

    # Each replica, querying the same vector, returns the same value.
    query = _unit([1.0, 0.0, 0.0, 0.0])
    assert leader.lookup(query, threshold=0.5) == follower.lookup(query, threshold=0.5)
