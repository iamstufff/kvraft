"""Raft-replicated semantic cache backed by pysyncobj.

The replicated state is the (embedding, value) pairs inserted via ``put``.
``get_or_miss`` is a local read — followers serve reads from their own replica
of the state. Only ``put`` goes through the Raft log.

The state is split into two layers:

* ``_CacheState`` — pure-Python state machine. Applies ops, answers lookups.
  No networking, fully unit-testable.
* ``ReplicatedSemanticCache`` — ``SyncObj`` wrapper. Decorates the single
  write op with ``@replicated`` so pysyncobj logs + replicates + commits it
  before applying to local state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pysyncobj import SyncObj, SyncObjConf, replicated

from src.cache.core import Hit, Miss
from src.cache.embedding import embed
from src.cache.index import SemanticIndex
from src.config import get_settings


@dataclass(frozen=True)
class AppliedPut:
    id: int


class _CacheState:
    """In-memory state machine: embedding index + id→value store.

    Deterministic so every replica arrives at the same state after applying
    the same sequence of ``apply_put`` calls.
    """

    def __init__(self, index: SemanticIndex | None = None) -> None:
        self._index = index if index is not None else SemanticIndex()
        self._values: dict[int, str] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.size

    def apply_put(self, value: str, embedding: NDArray[np.float32]) -> int:
        assigned_id = self._next_id
        self._next_id += 1
        self._index.add(embedding, id_=assigned_id)
        self._values[assigned_id] = value
        return assigned_id

    def lookup(self, embedding: NDArray[np.float32], threshold: float) -> Hit | None:
        matches = self._index.search(embedding, k=1, threshold=threshold)
        if not matches:
            return None
        top = matches[0]
        return Hit(value=self._values[top.id], similarity=top.similarity)


class ReplicatedSemanticCache(SyncObj):  # type: ignore[misc]
    """SyncObj-backed semantic cache. Writes replicate via Raft, reads are local."""

    def __init__(
        self,
        self_addr: str,
        peer_addrs: list[str],
        *,
        index: SemanticIndex | None = None,
        conf: SyncObjConf | None = None,
    ) -> None:
        super().__init__(self_addr, peer_addrs, conf=conf)
        self._state = _CacheState(index)

    @property
    def size(self) -> int:
        return self._state.size

    @replicated  # type: ignore[untyped-decorator]
    def _apply_put(self, value: str, embedding_bytes: bytes) -> int:
        vector = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
        return self._state.apply_put(value, vector)

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        # sync=True makes pysyncobj block until the op is committed + applied.
        result = self._apply_put(value, vector.tobytes(), sync=True)
        return int(result)

    def get_or_miss(self, prompt: str) -> Hit | Miss:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        hit = self._state.lookup(vector, threshold)
        if hit is not None:
            return hit
        return Miss(prompt=prompt, embedding=vector)
