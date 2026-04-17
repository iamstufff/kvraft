"""FastAPI application bootstrap."""

from fastapi import FastAPI

from src.api.health import router as health_router

app = FastAPI(title="kvraft")
app.include_router(health_router)
