"""Health-check API routes."""

from fastapi import APIRouter

from src.config import get_settings

router = APIRouter()


def get_node_id() -> str:
    """Return the node identifier for the current process."""

    return get_settings().node_id


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "node_id": get_node_id()}
