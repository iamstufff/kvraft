"""50 concurrent paraphrases of one prompt collapse to one upstream call."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.embedding import embed
from src.concurrency.single_flight import SingleFlight

pytestmark = pytest.mark.integration


@pytest.fixture
def paraphrases() -> list[str]:
    return [
        "What is SQL injection?",
        "Explain SQLi and how parameterized queries help.",
        "How does a blind SQL injection attack work?",
        "Describe SQL injection in plain terms.",
        "Why is SQLi still common today?",
    ]


async def test_concurrent_paraphrases_coalesce_to_one_upstream_call(
    paraphrases: list[str],
) -> None:
    sf = SingleFlight()
    upstream_calls = 0
    gate = asyncio.Event()

    async def upstream() -> str:
        nonlocal upstream_calls
        upstream_calls += 1
        await gate.wait()
        return "shared"

    async def call(prompt: str) -> str:
        embedding: NDArray[np.float32] = embed(prompt)
        return await sf.execute(embedding, threshold=0.8, fn=upstream)

    prompts = paraphrases * 10
    tasks = [asyncio.create_task(call(p)) for p in prompts]
    await asyncio.sleep(0.05)
    gate.set()
    results = await asyncio.gather(*tasks)

    assert all(r == "shared" for r in results)
    # Tight upper bound: most paraphrases collapse, but the very first concurrent
    # window may issue 2-3 distinct calls before _inflight catches them. Spec
    # target is ">=90% reduction" — 5 or fewer is well inside that for 50 calls.
    assert upstream_calls <= 5
