"""FastAPI application bootstrap."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.query import _get_cache, _get_router
from src.api.query import router as query_router
from src.config import get_settings


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Eagerly construct the Raft-backed cache + router at startup so:
    #   1. all replicas bind their Raft ports before the first /query arrives
    #      (otherwise the first write blocks forever waiting for quorum); and
    #   2. an empty/misconfigured provider chain fails loudly at boot.
    if get_settings().raft_enabled:
        _get_cache()
    _get_router()
    yield


app = FastAPI(title="kvraft", lifespan=_lifespan)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(metrics_router)
