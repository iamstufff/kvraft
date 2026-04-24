from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_M,
    DEFAULT_MAX_CAPACITY,
    DEFAULT_NODE_ID,
    DEFAULT_REBUILD_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    get_settings,
)

TEST_HNSW_EF_CONSTRUCTION = 128
TEST_HNSW_M = 32
TEST_SIMILARITY_THRESHOLD = 0.91
TEST_MAX_CAPACITY = 500
TEST_REBUILD_THRESHOLD = 0.42
TEST_RAFT_PEERS = ["raft-a:4321", "raft-b:4321", "raft-c:4321"]


def test_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("HNSW_EF_CONSTRUCTION", str(TEST_HNSW_EF_CONSTRUCTION))
    monkeypatch.setenv("HNSW_M", str(TEST_HNSW_M))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", str(TEST_SIMILARITY_THRESHOLD))
    monkeypatch.setenv("MAX_CAPACITY", str(TEST_MAX_CAPACITY))
    monkeypatch.setenv("REBUILD_THRESHOLD", str(TEST_REBUILD_THRESHOLD))
    monkeypatch.setenv("NODE_ID", "raft-a")
    monkeypatch.setenv("RAFT_PEERS", '["raft-a:4321", "raft-b:4321", "raft-c:4321"]')

    settings = get_settings()

    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.hnsw_ef_construction == TEST_HNSW_EF_CONSTRUCTION
    assert settings.hnsw_m == TEST_HNSW_M
    assert settings.similarity_threshold == TEST_SIMILARITY_THRESHOLD
    assert settings.max_capacity == TEST_MAX_CAPACITY
    assert settings.rebuild_threshold == TEST_REBUILD_THRESHOLD
    assert settings.node_id == "raft-a"
    assert settings.raft_peers == TEST_RAFT_PEERS


def test_settings_use_defaults_when_environment_is_empty(monkeypatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "EMBEDDING_MODEL",
        "HNSW_EF_CONSTRUCTION",
        "HNSW_M",
        "SIMILARITY_THRESHOLD",
        "MAX_CAPACITY",
        "REBUILD_THRESHOLD",
        "NODE_ID",
        "RAFT_PEERS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.gemini_api_key == ""
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.hnsw_ef_construction == DEFAULT_HNSW_EF_CONSTRUCTION
    assert settings.hnsw_m == DEFAULT_HNSW_M
    assert settings.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert settings.max_capacity == DEFAULT_MAX_CAPACITY
    assert settings.rebuild_threshold == DEFAULT_REBUILD_THRESHOLD
    assert settings.node_id == DEFAULT_NODE_ID
    assert settings.raft_peers == []
