"""POST /query — semantic cache in front of the provider."""

from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.cache.core import Hit, SemanticCache
from src.proxy.base import Provider, ProxyError
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


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    cache = _get_cache()
    provider: Provider = _get_gemini()

    result = cache.get_or_miss(body.prompt)
    if isinstance(result, Hit):
        return QueryResponse(
            response=result.value,
            cached=True,
            similarity=result.similarity,
        )

    try:
        response_text = await provider.complete(body.prompt)
    except ProxyError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cache.put(body.prompt, response_text, embedding=result.embedding)
    return QueryResponse(response=response_text, cached=False, similarity=None)
