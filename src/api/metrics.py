"""Prometheus /metrics endpoint."""

from fastapi import APIRouter, Response

from src.metrics import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
