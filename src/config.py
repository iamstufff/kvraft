"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_NODE_ID = "node-1"
DEFAULT_SIMILARITY_THRESHOLD = 0.8


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION
    hnsw_m: int = DEFAULT_HNSW_M
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    node_id: str = DEFAULT_NODE_ID
    raft_bind: str = ""
    raft_peers: list[str] = Field(default_factory=list)

    @property
    def raft_enabled(self) -> bool:
        return bool(self.raft_bind) and bool(self.raft_peers)


@lru_cache
def get_settings() -> Settings:
    return Settings()
