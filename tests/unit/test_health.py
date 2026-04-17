from src.api.health import get_node_id, health
from src.config import DEFAULT_NODE_ID


def test_get_node_id_returns_default(monkeypatch) -> None:
    monkeypatch.delenv("NODE_ID", raising=False)

    assert get_node_id() == DEFAULT_NODE_ID


def test_health_uses_node_id_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NODE_ID", "raft-a")

    response = health()

    assert response == {"status": "ok", "node_id": "raft-a"}
