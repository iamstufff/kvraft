"""POST /query — semantic cache in front of the provider."""

import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.cache.core import Hit, SemanticCache
from src.metrics import PROVIDER_CALLS, QUERY_LATENCY, QUERY_TOTAL
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError, ProxyError
from src.proxy.gemini_client import GeminiClient

router = APIRouter()


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["gemini"] = "gemini"


class QueryResponse(BaseModel):
    response: str
    cached: bool
    similarity: float | None = None


@dataclass
class _State:
    cache: SemanticCache | None = field(default=None)
    gemini: GeminiClient | None = field(default=None)


_state = _State()


def _get_cache() -> SemanticCache:
    if _state.cache is None:
        _state.cache = SemanticCache()
    return _state.cache


def _get_gemini() -> GeminiClient:
    if _state.gemini is None:
        _state.gemini = GeminiClient()
    return _state.gemini


def _record(outcome: str, start: float) -> None:
    QUERY_TOTAL.labels(outcome=outcome).inc()
    QUERY_LATENCY.labels(outcome=outcome).observe(time.perf_counter() - start)


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    cache = _get_cache()
    provider: Provider = _get_gemini()

    result = cache.get_or_miss(body.prompt)
    if isinstance(result, Hit):
        _record("hit", started)
        return QueryResponse(
            response=result.value,
            cached=True,
            similarity=result.similarity,
        )

    try:
        response_text = await provider.complete(body.prompt)
    except ProviderTimeoutError as exc:
        PROVIDER_CALLS.labels(provider=body.provider, result="timeout").inc()
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProviderAPIError as exc:
        PROVIDER_CALLS.labels(provider=body.provider, result="api_error").inc()
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProxyError as exc:
        PROVIDER_CALLS.labels(provider=body.provider, result="api_error").inc()
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    PROVIDER_CALLS.labels(provider=body.provider, result="ok").inc()
    cache.put(body.prompt, response_text, embedding=result.embedding)
    _record("miss", started)
    return QueryResponse(response=response_text, cached=False, similarity=None)
