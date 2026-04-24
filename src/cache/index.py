"""HNSW index over L2-normalized semantic embeddings.

Uses inner-product distance. ``src.cache.embedding.embed`` guarantees unit-norm
vectors, so inner product is equivalent to cosine similarity with less work.

Eviction uses ``hnswlib.mark_deleted`` — a soft delete: the underlying graph
still holds the vector, but ``knn_query`` skips it. Memory is reclaimed via
``rebuild`` once the soft-deleted fraction crosses
``Settings.rebuild_threshold``.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import hnswlib
import numpy as np
from numpy.typing import NDArray

from src.config import get_settings

EMBEDDING_DIM = 384
INITIAL_CAPACITY = 10_000


@dataclass(frozen=True)
class Match:
    id: int
    similarity: float


class SemanticIndex:
    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        initial_capacity: int = INITIAL_CAPACITY,
    ) -> None:
        self._dim = dim
        self._capacity = initial_capacity
        self._soft_deleted: set[int] = set()
        self._index = self._fresh_index(initial_capacity, dim=dim)

    @staticmethod
    def _fresh_index(capacity: int, dim: int = EMBEDDING_DIM) -> hnswlib.Index:
        settings = get_settings()
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(
            max_elements=capacity,
            ef_construction=settings.hnsw_ef_construction,
            M=settings.hnsw_m,
        )
        index.set_ef(max(settings.hnsw_ef_construction // 2, 50))
        return index

    @property
    def total_count(self) -> int:
        """Entries physically present in the graph, including soft-deleted."""
        return cast(int, self._index.get_current_count())

    @property
    def soft_deleted_count(self) -> int:
        return len(self._soft_deleted)

    @property
    def size(self) -> int:
        """Live (queryable) entries."""
        return self.total_count - self.soft_deleted_count

    def add(self, vector: NDArray[np.float32], id_: int) -> None:
        if self.total_count >= self._capacity:
            self._capacity *= 2
            self._index.resize_index(self._capacity)
        self._index.add_items(
            vector.reshape(1, -1),
            np.array([id_], dtype=np.int64),
        )

    def mark_deleted(self, id_: int) -> None:
        if id_ in self._soft_deleted:
            return
        self._index.mark_deleted(id_)
        self._soft_deleted.add(id_)

    def rebuild(self, items: Iterable[tuple[int, NDArray[np.float32]]]) -> None:
        items_list = list(items)
        new_capacity = max(INITIAL_CAPACITY, len(items_list) * 2 or INITIAL_CAPACITY)
        new_index = self._fresh_index(new_capacity, dim=self._dim)
        if items_list:
            ids = np.array([i for i, _ in items_list], dtype=np.int64)
            vectors = np.stack([v for _, v in items_list])
            new_index.add_items(vectors, ids)
        self._index = new_index
        self._capacity = new_capacity
        self._soft_deleted.clear()

    def search(
        self,
        vector: NDArray[np.float32],
        k: int,
        threshold: float,
    ) -> list[Match]:
        if self.size == 0:
            return []
        effective_k = min(k, self.size)
        labels, distances = self._index.knn_query(vector.reshape(1, -1), k=effective_k)
        matches: list[Match] = []
        for label, distance in zip(labels[0], distances[0], strict=True):
            similarity = 1.0 - float(distance)
            if similarity >= threshold:
                matches.append(Match(id=int(label), similarity=similarity))
        return matches
