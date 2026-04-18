"""Prometheus /metrics endpoint."""

from fastapi import APIRouter, Response

from src.api.query import _get_cache
from src.metrics import CONTENT_TYPE_LATEST, LEADER_STATE, generate_latest
from src.raft.state_machine import ReplicatedSemanticCache

router = APIRouter()


def _refresh_leader_state() -> None:
    cache = _get_cache()
    if isinstance(cache, ReplicatedSemanticCache):
        LEADER_STATE.set(1.0 if cache.is_leader() else 0.0)
    else:
        LEADER_STATE.set(0.0)


@router.get("/metrics")
def metrics() -> Response:
    _refresh_leader_state()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
