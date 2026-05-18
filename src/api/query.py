"""POST /query — semantic cache in front of a multi-provider router."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.cache.core import Hit, SemanticCache
from src.concurrency.single_flight import SingleFlight
from src.config import get_settings
from src.metrics import QUERY_LATENCY, QUERY_TOTAL
from src.proxy.anthropic_client import AnthropicClient
from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
    ProxyError,
)
from src.proxy.gemini_client import GeminiClient
from src.proxy.openai_client import OpenAIClient
from src.proxy.router import ProviderRouter
from src.raft.state_machine import ReplicatedSemanticCache

CacheBackend = SemanticCache | ReplicatedSemanticCache

router = APIRouter()


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"


class QueryResponse(BaseModel):
    response: str
    cached: bool
    similarity: float | None = None


@dataclass
class _State:
    cache: CacheBackend | None = field(default=None)
    router: Provider | None = field(default=None)
    single_flight: SingleFlight | None = field(default=None)


_state = _State()


def _build_cache() -> CacheBackend:
    settings = get_settings()
    if settings.raft_enabled:
        return ReplicatedSemanticCache(settings.raft_bind, settings.raft_peers)
    return SemanticCache()


def _get_cache() -> CacheBackend:
    if _state.cache is None:
        _state.cache = _build_cache()
    return _state.cache


def _construct_provider(name: str) -> Provider:
    if name == "gemini":
        return GeminiClient()
    if name == "openai":
        return OpenAIClient()
    if name == "anthropic":
        return AnthropicClient()
    raise ProxyError(f"unknown provider: {name}")


def _build_router() -> Provider:
    settings = get_settings()
    providers: list[Provider] = []
    for name in settings.provider_chain:
        try:
            providers.append(_construct_provider(name))
        except ProxyError:
            continue
    if not providers:
        raise ProxyError(
            "provider_chain empty after dropping unconfigured providers; "
            "set at least one of GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
        )
    return ProviderRouter(
        providers=providers,
        failure_threshold=settings.breaker_failure_threshold,
        failure_window_seconds=settings.breaker_failure_window_seconds,
        recovery_seconds=settings.breaker_recovery_seconds,
    )


def _get_router() -> Provider:
    if _state.router is None:
        _state.router = _build_router()
    return _state.router


def _get_single_flight() -> SingleFlight:
    if _state.single_flight is None:
        _state.single_flight = SingleFlight()
    return _state.single_flight


def _record(outcome: str, start: float) -> None:
    QUERY_TOTAL.labels(outcome=outcome).inc()
    QUERY_LATENCY.labels(outcome=outcome).observe(time.perf_counter() - start)


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    cache = _get_cache()
    provider = _get_router()
    sf = _get_single_flight()
    settings = get_settings()

    result = await asyncio.to_thread(cache.get_or_miss, body.prompt)
    if isinstance(result, Hit):
        _record("hit", started)
        return QueryResponse(
            response=result.value,
            cached=True,
            similarity=result.similarity,
        )

    async def upstream_and_store() -> str:
        text = await provider.complete(body.prompt)
        await asyncio.to_thread(cache.put, body.prompt, text, embedding=result.embedding)
        return text

    try:
        response_text = await sf.execute(
            result.embedding,
            threshold=settings.coalesce_threshold,
            fn=upstream_and_store,
        )
    except ProviderTimeoutError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProviderChainExhaustedError as exc:
        _record("provider_error", started)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(int(settings.breaker_recovery_seconds))},
        ) from exc
    except ProviderAPIError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProxyError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _record("miss", started)
    return QueryResponse(response=response_text, cached=False, similarity=None)
