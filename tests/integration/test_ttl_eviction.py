"""TTL=1s entry must be a Miss after 1.1s on every replica's lookup."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.index import SemanticIndex
from src.raft.state_machine import _CacheState

pytestmark = pytest.mark.integration


def _unit(values: list[float]) -> NDArray[np.float32]:
    arr = np.array(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_ttl_expired_entry_is_miss_on_all_replicas() -> None:
    leader = _CacheState(index=SemanticIndex(dim=4, initial_capacity=8))
    follower = _CacheState(index=SemanticIndex(dim=4, initial_capacity=8))

    op_time = time.time()
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    leader.apply_put("answer", vec, op_time=op_time, ttl_seconds=1.0)
    follower.apply_put("answer", vec, op_time=op_time, ttl_seconds=1.0)

    assert leader.lookup(vec, threshold=0.5, now=op_time + 0.5) is not None
    assert follower.lookup(vec, threshold=0.5, now=op_time + 0.5) is not None

    assert leader.lookup(vec, threshold=0.5, now=op_time + 1.1) is None
    assert follower.lookup(vec, threshold=0.5, now=op_time + 1.1) is None

    assert leader.find_expired(now=op_time + 1.1) == [0]
