import asyncio

import numpy as np
from numpy.typing import NDArray

from src.concurrency.single_flight import SingleFlight


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


async def test_single_caller_runs_fn_once() -> None:
    sf = SingleFlight()
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await sf.execute(_unit([1.0, 0.0, 0.0]), threshold=0.8, fn=fn)
    assert result == "ok"
    assert calls == 1


async def test_concurrent_identical_embeddings_collapse_to_one_call() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return "shared"

    vec = _unit([1.0, 0.0, 0.0])

    async def caller() -> str:
        return await sf.execute(vec, threshold=0.8, fn=fn)

    tasks = [asyncio.create_task(caller()) for _ in range(50)]
    await asyncio.sleep(0.01)
    gate.set()
    results = await asyncio.gather(*tasks)

    assert results == ["shared"] * 50
    assert calls == 1


async def test_similar_embeddings_coalesce_when_above_threshold() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return "shared"

    base = _unit([1.0, 0.0, 0.0])
    nearby = _unit([1.0, 0.05, 0.0])

    t_base = asyncio.create_task(sf.execute(base, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    t_near = asyncio.create_task(sf.execute(nearby, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    gate.set()
    r_base, r_near = await asyncio.gather(t_base, t_near)

    assert r_base == r_near == "shared"
    assert calls == 1


async def test_dissimilar_embeddings_each_run_fn() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return f"call-{calls}"

    vec_a = _unit([1.0, 0.0, 0.0])
    vec_b = _unit([0.0, 1.0, 0.0])

    t1 = asyncio.create_task(sf.execute(vec_a, threshold=0.8, fn=fn))
    t2 = asyncio.create_task(sf.execute(vec_b, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(t1, t2)

    assert calls == 2


async def test_exception_propagates_to_all_waiters() -> None:
    sf = SingleFlight()
    gate = asyncio.Event()

    async def fn() -> str:
        await gate.wait()
        raise RuntimeError("boom")

    vec = _unit([1.0, 0.0, 0.0])
    tasks = [asyncio.create_task(sf.execute(vec, threshold=0.8, fn=fn)) for _ in range(5)]
    await asyncio.sleep(0.01)
    gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(results) == 5
    for r in results:
        assert isinstance(r, RuntimeError)
        assert str(r) == "boom"


async def test_late_arrival_after_completion_runs_again() -> None:
    sf = SingleFlight()
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    vec = _unit([1.0, 0.0, 0.0])
    await sf.execute(vec, threshold=0.8, fn=fn)
    await sf.execute(vec, threshold=0.8, fn=fn)
    assert calls == 2
