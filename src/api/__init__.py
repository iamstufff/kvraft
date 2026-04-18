"""FastAPI application bootstrap."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.query import _get_cache
from src.api.query import router as query_router
from src.config import get_settings


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Eagerly construct the Raft-backed cache at startup so all replicas
    # bind their Raft ports and peer-connect before the first /query arrives;
    # otherwise the first write blocks forever waiting for quorum.
    if get_settings().raft_enabled:
        _get_cache()
    yield


app = FastAPI(title="kvraft", lifespan=_lifespan)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(metrics_router)
