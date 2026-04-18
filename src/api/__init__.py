"""FastAPI application bootstrap."""

from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.query import router as query_router

app = FastAPI(title="kvraft")
app.include_router(health_router)
app.include_router(query_router)
app.include_router(metrics_router)
