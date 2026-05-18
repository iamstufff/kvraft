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


def test_apply_delete_removes_entry(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    id_ = state.apply_put("doomed", vec)

    assert state.apply_delete(id_) is True
    assert state.size == 0
    assert state.soft_deleted_count == 1
    assert state.lookup(vec, threshold=0.5) is None


def test_apply_delete_returns_false_for_unknown_id(state: _CacheState) -> None:
    assert state.apply_delete(999) is False


def test_lookup_with_id_returns_hit_id(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    id_ = state.apply_put("val", vec)

    found = state.lookup_with_id(vec, threshold=0.5)

    assert found is not None
    hit_id, hit = found
    assert hit_id == id_
    assert hit.value == "val"


def test_delete_then_apply_determinism_between_replicas() -> None:
    vectors = [
        _unit([1.0, 0.0, 0.0, 0.0]),
        _unit([0.0, 1.0, 0.0, 0.0]),
        _unit([0.0, 0.0, 1.0, 0.0]),
    ]

    leader = _state()
    follower = _state()
    ids = []
    for i, vec in enumerate(vectors):
        ids.append(leader.apply_put(f"v{i}", vec))
        follower.apply_put(f"v{i}", vec)

    # Evict the middle entry on both replicas (as Raft would).
    assert leader.apply_delete(ids[1]) is True
    assert follower.apply_delete(ids[1]) is True

    for vec in vectors:
        assert leader.lookup(vec, threshold=0.5) == follower.lookup(vec, threshold=0.5)
    assert leader.size == follower.size == 2


def test_apply_put_stores_expires_at(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=60.0)
    assert state.expires_at(0) == 160.0


def test_apply_put_with_zero_ttl_means_never_expires(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=0.0)
    assert state.expires_at(0) is None


def test_lookup_returns_none_for_expired_entry(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    assert state.lookup(vec, threshold=0.5, now=109.0) is not None
    assert state.lookup(vec, threshold=0.5, now=111.0) is None


def test_lookup_without_now_uses_no_expiry(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    # Backwards-compatible call: no now, no TTL check.
    assert state.lookup(vec, threshold=0.5) is not None


def test_find_expired_returns_ids_due_for_eviction(state: _CacheState) -> None:
    state.apply_put("a", _unit([1.0, 0.0, 0.0, 0.0]), op_time=100.0, ttl_seconds=10.0)
    state.apply_put("b", _unit([0.0, 1.0, 0.0, 0.0]), op_time=100.0, ttl_seconds=100.0)
    expired = state.find_expired(now=120.0)
    assert expired == [0]


def test_apply_delete_clears_expires_at(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    id_ = state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    state.apply_delete(id_)
    assert state.expires_at(id_) is None


def test_rebuild_reclaims_soft_deleted_space(state: _CacheState) -> None:
    vectors = [
        _unit([1.0, 0.0, 0.0, 0.0]),
        _unit([0.0, 1.0, 0.0, 0.0]),
        _unit([0.0, 0.0, 1.0, 0.0]),
    ]
    ids = [state.apply_put(f"v{i}", v) for i, v in enumerate(vectors)]

    state.apply_delete(ids[0])
    state.apply_delete(ids[1])
    assert state.soft_deleted_count == 2
    assert state.should_rebuild(threshold=0.3) is True

    state.rebuild_index()

    assert state.soft_deleted_count == 0
    assert state.total_count == 1
    assert state.size == 1
    # Surviving entry still queryable.
    assert state.lookup(vectors[2], threshold=0.5) is not None
