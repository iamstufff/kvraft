from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
)
from src.proxy.router import ProviderRouter


def _provider(name: str, complete: AsyncMock) -> MagicMock:
    p = MagicMock(spec=Provider)
    p.name = name
    p.complete = complete
    return p


def _router(providers: list[MagicMock]) -> ProviderRouter:
    return ProviderRouter(
        providers=providers,
        failure_threshold=2,
        failure_window_seconds=30.0,
        recovery_seconds=10.0,
    )


def test_router_implements_provider_protocol() -> None:
    p = _provider("p", AsyncMock(return_value="ok"))
    router = _router([p])
    assert isinstance(router, Provider)
    assert router.name == "chain[p]"


async def test_first_provider_succeeds_no_fallback() -> None:
    p1 = _provider("a", AsyncMock(return_value="from-a"))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-a"
    p2.complete.assert_not_called()


async def test_falls_through_to_next_provider_on_timeout() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-b"


async def test_falls_through_on_api_error() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderAPIError("500")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-b"


async def test_chain_exhausted_raises() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(side_effect=ProviderAPIError("500")))
    with pytest.raises(ProviderChainExhaustedError):
        await _router([p1, p2]).complete("hi")


async def test_breaker_opens_after_threshold_and_skips_provider() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    router = _router([p1, p2])

    for _ in range(2):
        assert await router.complete("hi") == "from-b"
    assert p1.complete.await_count == 2

    assert await router.complete("hi") == "from-b"
    assert p1.complete.await_count == 2


def test_empty_chain_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="empty provider chain"):
        _router([])
