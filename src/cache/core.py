"""Semantic cache core: embedding + HNSW + similarity threshold.

A ``SemanticCache.get_or_miss`` call returns either a ``Hit`` with the cached
value and its similarity, or a ``Miss`` carrying the prompt's embedding so the
caller can feed it straight back into ``put`` without re-embedding.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.cache.embedding import embed
from src.cache.index import SemanticIndex
from src.config import get_settings


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
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.size

    def get_or_miss(self, prompt: str) -> CacheResult:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        matches = self._index.search(vector, k=1, threshold=threshold)
        if matches:
            top = matches[0]
            return Hit(value=self._values[top.id], similarity=top.similarity)
        return Miss(prompt=prompt, embedding=vector)

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        assigned_id = self._next_id
        self._next_id += 1
        self._index.add(vector, id_=assigned_id)
        self._values[assigned_id] = value
        return assigned_id
