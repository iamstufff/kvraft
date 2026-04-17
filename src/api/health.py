"""Health-check API routes."""

from os import getenv

from fastapi import APIRouter

DEFAULT_NODE_ID = "node-1"

router = APIRouter()


def get_node_id() -> str:
    """Return the node identifier for the current process."""

    return getenv("NODE_ID", DEFAULT_NODE_ID)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "node_id": get_node_id()}
