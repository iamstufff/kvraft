"""Raft-replicated semantic cache backed by pysyncobj.

The replicated state is the (embedding, value) pairs inserted via ``put`` and
removed via ``_apply_delete``. ``get_or_miss`` is a local read — followers
serve reads from their own replica of the state. Only writes and deletes go
through the Raft log.

Eviction policy:

* LRU bookkeeping is leader-only. Each replica applies ``_apply_delete`` ops
  but only the current leader tracks read/write recency, picks the victim, and
  emits the ``_apply_delete`` through Raft. This trades approximate LRU
  (follower reads are invisible to eviction ordering) for deterministic,
  convergent state.
* ``hnswlib`` soft-deletes accumulate until the fraction crosses
  ``Settings.rebuild_threshold``, at which point each replica deterministically
  rebuilds its own index from the current state.

The state is split into two layers:

* ``_CacheState`` — pure-Python state machine. Applies ops, answers lookups.
  No networking, fully unit-testable.
* ``ReplicatedSemanticCache`` — ``SyncObj`` wrapper. Decorates write/delete
  ops with ``@replicated`` so pysyncobj logs + replicates + commits them
  before applying to local state.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray
from pysyncobj import SyncObj, SyncObjConf, replicated

from src.cache.core import Hit, Miss
from src.cache.embedding import embed
from src.cache.index import SemanticIndex
from src.config import get_settings
from src.metrics import (
    CACHE_EVICTIONS,
    CACHE_LIVE_ENTRIES,
    CACHE_REBUILDS,
    CACHE_SOFT_DELETED_ENTRIES,
)


class _CacheState:
    """In-memory state machine: embedding index + id→value store + id→embedding store.

    Deterministic so every replica arrives at the same state after applying
    the same sequence of ``apply_put`` / ``apply_delete`` calls. Embeddings
    are kept so the index can be rebuilt when soft-deletes pile up.
    """

    def __init__(self, index: SemanticIndex | None = None) -> None:
        self._index = index if index is not None else SemanticIndex()
        self._values: dict[int, str] = {}
        self._embeddings: dict[int, NDArray[np.float32]] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def soft_deleted_count(self) -> int:
        return self._index.soft_deleted_count

    @property
    def total_count(self) -> int:
        return self._index.total_count

    def apply_put(self, value: str, embedding: NDArray[np.float32]) -> int:
        assigned_id = self._next_id
        self._next_id += 1
        self._index.add(embedding, id_=assigned_id)
        self._values[assigned_id] = value
        self._embeddings[assigned_id] = embedding
        return assigned_id

    def apply_delete(self, id_: int) -> bool:
        if id_ not in self._values:
            return False
        self._index.mark_deleted(id_)
        del self._values[id_]
        del self._embeddings[id_]
        return True

    def should_rebuild(self, threshold: float) -> bool:
        total = self._index.total_count
        if total == 0:
            return False
        return self._index.soft_deleted_count / total > threshold

    def rebuild_index(self) -> None:
        self._index.rebuild(self._embeddings.items())

    def lookup(self, embedding: NDArray[np.float32], threshold: float) -> Hit | None:
        found = self.lookup_with_id(embedding, threshold)
        return None if found is None else found[1]

    def lookup_with_id(
        self, embedding: NDArray[np.float32], threshold: float
    ) -> tuple[int, Hit] | None:
        matches = self._index.search(embedding, k=1, threshold=threshold)
        if not matches:
            return None
        top = matches[0]
        value = self._values.get(top.id)
        if value is None:
            return None
        return top.id, Hit(value=value, similarity=top.similarity)


def _publish_cache_gauges(state: _CacheState) -> None:
    CACHE_LIVE_ENTRIES.set(state.size)
    CACHE_SOFT_DELETED_ENTRIES.set(state.soft_deleted_count)


class ReplicatedSemanticCache(SyncObj):  # type: ignore[misc]
    """SyncObj-backed semantic cache. Writes and deletes replicate via Raft.

    LRU tracker lives only on the leader. On leadership change the new leader
    starts with an empty tracker and warms it up from subsequent reads/writes;
    eviction during the warmup window is best-effort.
    """

    def __init__(
        self,
        self_addr: str,
        peer_addrs: list[str],
        *,
        index: SemanticIndex | None = None,
        conf: SyncObjConf | None = None,
    ) -> None:
        # SyncObj.__init__ walks dir(self) to discover @replicated methods, which
        # dereferences every property too — so `_state` must exist before we
        # call super().__init__ or `size` blows up.
        self._state = _CacheState(index)
        self._lru: OrderedDict[int, None] = OrderedDict()
        super().__init__(self_addr, peer_addrs, conf=conf)

    @property
    def size(self) -> int:
        return self._state.size

    def is_leader(self) -> bool:
        """True iff this replica currently believes it is the Raft leader."""
        return bool(self._isLeader())

    def _touch_lru(self, id_: int) -> None:
        self._lru[id_] = None
        self._lru.move_to_end(id_)

    def _maybe_rebuild(self) -> None:
        threshold = get_settings().rebuild_threshold
        if self._state.should_rebuild(threshold):
            self._state.rebuild_index()
            CACHE_REBUILDS.inc()

    @replicated  # type: ignore[untyped-decorator]
    def _apply_put(self, value: str, embedding_bytes: bytes) -> int:
        vector = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
        assigned_id = self._state.apply_put(value, vector)
        _publish_cache_gauges(self._state)
        return assigned_id

    @replicated  # type: ignore[untyped-decorator]
    def _apply_delete(self, id_: int) -> bool:
        deleted = self._state.apply_delete(id_)
        if deleted:
            CACHE_EVICTIONS.inc()
            self._maybe_rebuild()
            _publish_cache_gauges(self._state)
        return deleted

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        settings = get_settings()
        if self.is_leader() and self._state.size >= settings.max_capacity:
            victim = next(iter(self._lru), None)
            if victim is not None:
                self._apply_delete(victim, sync=True)
                self._lru.pop(victim, None)
        # sync=True makes pysyncobj block until the op is committed + applied.
        assigned_id = int(self._apply_put(value, vector.tobytes(), sync=True))
        if self.is_leader():
            self._touch_lru(assigned_id)
        return assigned_id

    def get_or_miss(self, prompt: str) -> Hit | Miss:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        found = self._state.lookup_with_id(vector, threshold)
        if found is not None:
            hit_id, hit = found
            if self.is_leader():
                self._touch_lru(hit_id)
            return hit
        return Miss(prompt=prompt, embedding=vector)
