"""Semantic cache core: embedding + HNSW + similarity threshold.

A ``SemanticCache.get_or_miss`` call returns either a ``Hit`` with the cached
value and its similarity, or a ``Miss`` carrying the prompt's embedding so the
caller can feed it straight back into ``put`` without re-embedding.

The cache is bounded by ``Settings.max_capacity``: when full, the LRU entry
is soft-deleted before a new insert. Soft-deletes are reclaimed via
``SemanticIndex.rebuild`` once they cross ``Settings.rebuild_threshold``.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.cache.embedding import embed
from src.cache.index import SemanticIndex
from src.config import get_settings
from src.metrics import (
    CACHE_EVICTIONS,
    CACHE_LIVE_ENTRIES,
    CACHE_REBUILDS,
    CACHE_SOFT_DELETED_ENTRIES,
)


@dataclass(frozen=True)
class Hit:
    value: str
    similarity: float


@dataclass(frozen=True)
class Miss:
    prompt: str
    embedding: NDArray[np.float32]


CacheResult = Hit | Miss


class SemanticCache:
    def __init__(self, index: SemanticIndex | None = None) -> None:
        self._index = index if index is not None else SemanticIndex()
        self._values: dict[int, str] = {}
        self._embeddings: dict[int, NDArray[np.float32]] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def soft_deleted_count(self) -> int:
        return self._index.soft_deleted_count

    def get_or_miss(self, prompt: str) -> CacheResult:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        matches = self._index.search(vector, k=1, threshold=threshold)
        if matches:
            top = matches[0]
            self._touch(top.id)
            return Hit(value=self._values[top.id], similarity=top.similarity)
        return Miss(prompt=prompt, embedding=vector)

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        settings = get_settings()
        if self._index.size >= settings.max_capacity:
            victim = next(iter(self._lru), None)
            if victim is not None:
                self._evict(victim)
        assigned_id = self._next_id
        self._next_id += 1
        self._index.add(vector, id_=assigned_id)
        self._values[assigned_id] = value
        self._embeddings[assigned_id] = vector
        self._touch(assigned_id)
        self._publish_gauges()
        return assigned_id

    def _touch(self, id_: int) -> None:
        self._lru[id_] = None
        self._lru.move_to_end(id_)

    def _evict(self, id_: int) -> None:
        self._index.mark_deleted(id_)
        self._values.pop(id_, None)
        self._embeddings.pop(id_, None)
        self._lru.pop(id_, None)
        CACHE_EVICTIONS.inc()
        self._maybe_rebuild()
        self._publish_gauges()

    def _maybe_rebuild(self) -> None:
        threshold = get_settings().rebuild_threshold
        total = self._index.total_count
        if total == 0:
            return
        if self._index.soft_deleted_count / total > threshold:
            self._index.rebuild(self._embeddings.items())
            CACHE_REBUILDS.inc()

    def _publish_gauges(self) -> None:
        CACHE_LIVE_ENTRIES.set(self._index.size)
        CACHE_SOFT_DELETED_ENTRIES.set(self._index.soft_deleted_count)
