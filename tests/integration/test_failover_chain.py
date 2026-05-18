"""Force the first provider into 5x failures inside the window and verify
the router falls through to the next provider on the very next call."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy.base import Provider, ProviderTimeoutError
from src.proxy.router import ProviderRouter

pytestmark = pytest.mark.integration


def _provider(name: str, complete: AsyncMock) -> MagicMock:
    p = MagicMock(spec=Provider)
    p.name = name
    p.complete = complete
    return p


async def test_breaker_trips_after_5_timeouts_and_routes_to_next() -> None:
    flaky = _provider("gemini", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    healthy = _provider("openai", AsyncMock(return_value="ok"))
    router = ProviderRouter(
        providers=[flaky, healthy],
        failure_threshold=5,
        failure_window_seconds=30.0,
        recovery_seconds=15.0,
    )

    for _ in range(5):
        assert await router.complete("prompt") == "ok"

    flaky.complete.reset_mock()
    assert await router.complete("prompt") == "ok"
    flaky.complete.assert_not_called()
