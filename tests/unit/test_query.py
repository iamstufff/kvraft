from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from numpy.typing import NDArray

from src.api import query as query_module
from src.api.query import QueryRequest, query
from src.cache.core import Hit, Miss
from src.proxy.base import ProviderAPIError, ProviderTimeoutError


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    cache = MagicMock()
    monkeypatch.setattr(query_module, "_get_cache", lambda: cache)
    return cache


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock()
    monkeypatch.setattr(query_module, "_get_gemini", lambda: provider)
    return provider


async def test_hit_returns_cached_response_without_calling_provider(
    fake_cache: MagicMock, fake_provider: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Hit(value="cached answer", similarity=0.92)

    result = await query(QueryRequest(prompt="hi"))

    assert result.response == "cached answer"
    assert result.cached is True
    assert result.similarity == pytest.approx(0.92)
    fake_provider.complete.assert_not_called()
    fake_cache.put.assert_not_called()


async def test_miss_calls_provider_and_caches_response(
    fake_cache: MagicMock, fake_provider: MagicMock
) -> None:
    miss_embedding = _unit([1.0, 0.0, 0.0, 0.0])
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=miss_embedding)
    fake_provider.complete.return_value = "fresh answer"

    result = await query(QueryRequest(prompt="hi"))

    assert result.response == "fresh answer"
    assert result.cached is False
    assert result.similarity is None
    fake_provider.complete.assert_awaited_once_with("hi")
    fake_cache.put.assert_called_once_with("hi", "fresh answer", embedding=miss_embedding)


async def test_provider_timeout_maps_to_502(
    fake_cache: MagicMock, fake_provider: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    fake_provider.complete.side_effect = ProviderTimeoutError("slow")

    with pytest.raises(HTTPException) as exc_info:
        await query(QueryRequest(prompt="hi"))

    assert exc_info.value.status_code == 502
    fake_cache.put.assert_not_called()


async def test_provider_api_error_maps_to_502(
    fake_cache: MagicMock, fake_provider: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    fake_provider.complete.side_effect = ProviderAPIError("500")

    with pytest.raises(HTTPException) as exc_info:
        await query(QueryRequest(prompt="hi"))

    assert exc_info.value.status_code == 502


def test_query_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        QueryRequest(prompt="")


def test_query_request_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        QueryRequest(prompt="hi", provider="openai")  # type: ignore[arg-type]
