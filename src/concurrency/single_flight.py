"""Semantic-keyed single-flight: coalesce in-flight upstream calls.

When two paraphrases of the same prompt miss the cache at the same time,
naive code makes two upstream calls and writes the cache twice. ``SingleFlight``
keeps a small list of in-flight requests keyed by their embedding; subsequent
callers with a similar embedding await the same ``asyncio.Future``.

Brute-force cosine is acceptable: ``_inflight`` is bounded by request
concurrency, typically <100 entries, dimension 384. HNSW would be overkill.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.metrics import SINGLEFLIGHT_COALESCED


@dataclass
class _InFlight:
    embedding: NDArray[np.float32]
    future: asyncio.Future[str]
    started_at: float


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: list[_InFlight] = []
        self._lock = asyncio.Lock()

    async def execute(
        self,
        embedding: NDArray[np.float32],
        threshold: float,
        fn: Callable[[], Awaitable[str]],
    ) -> str:
        async with self._lock:
            match = self._best_match(embedding, threshold)
            if match is not None:
                SINGLEFLIGHT_COALESCED.inc()
                return await match.future
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            entry = _InFlight(embedding=embedding, future=future, started_at=time.monotonic())
            self._inflight.append(entry)

        try:
            result = await fn()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            # Mark the exception as retrieved so asyncio doesn't log
            # "Future exception was never retrieved" when no coalescer joined.
            future.exception()
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result
        finally:
            async with self._lock:
                try:
                    self._inflight.remove(entry)
                except ValueError:
                    pass

    def _best_match(self, embedding: NDArray[np.float32], threshold: float) -> _InFlight | None:
        best: _InFlight | None = None
        best_sim = threshold
        for entry in self._inflight:
            sim = float(np.dot(embedding, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best = entry
        return best
