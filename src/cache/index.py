"""HNSW index over L2-normalized semantic embeddings.

Uses inner-product distance. ``src.cache.embedding.embed`` guarantees unit-norm
vectors, so inner product is equivalent to cosine similarity with less work.
Capacity doubles when full; there is no shrink path — Raft replicas rebuild
from the replicated log on startup.
"""

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
        settings = get_settings()
        self._dim = dim
        self._capacity = initial_capacity
        self._index = hnswlib.Index(space="ip", dim=dim)
        self._index.init_index(
            max_elements=initial_capacity,
            ef_construction=settings.hnsw_ef_construction,
            M=settings.hnsw_m,
        )
        self._index.set_ef(max(settings.hnsw_ef_construction // 2, 50))

    @property
    def size(self) -> int:
        return cast(int, self._index.get_current_count())

    def add(self, vector: NDArray[np.float32], id_: int) -> None:
        if self.size >= self._capacity:
            self._capacity *= 2
            self._index.resize_index(self._capacity)
        self._index.add_items(
            vector.reshape(1, -1),
            np.array([id_], dtype=np.int64),
        )

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
