"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_NODE_ID = "node-1"
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MAX_CAPACITY = 10_000
DEFAULT_REBUILD_THRESHOLD = 0.3
DEFAULT_PROVIDER_CHAIN = ["gemini"]
DEFAULT_BREAKER_FAILURE_THRESHOLD = 5
DEFAULT_BREAKER_FAILURE_WINDOW_SECONDS = 30.0
DEFAULT_BREAKER_RECOVERY_SECONDS = 15.0
DEFAULT_COALESCE_THRESHOLD = 0.8
DEFAULT_CACHE_TTL_SECONDS = 3600.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION
    hnsw_m: int = DEFAULT_HNSW_M
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_capacity: int = DEFAULT_MAX_CAPACITY
    rebuild_threshold: float = DEFAULT_REBUILD_THRESHOLD
    node_id: str = DEFAULT_NODE_ID
    raft_bind: str = ""
    raft_peers: list[str] = Field(default_factory=list)

    provider_chain: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_PROVIDER_CHAIN)
    )
    breaker_failure_threshold: int = DEFAULT_BREAKER_FAILURE_THRESHOLD
    breaker_failure_window_seconds: float = DEFAULT_BREAKER_FAILURE_WINDOW_SECONDS
    breaker_recovery_seconds: float = DEFAULT_BREAKER_RECOVERY_SECONDS
    coalesce_threshold: float = DEFAULT_COALESCE_THRESHOLD
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS

    @field_validator("provider_chain", mode="before")
    @classmethod
    def _split_provider_chain(cls, value: object) -> object:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @property
    def raft_enabled(self) -> bool:
        return bool(self.raft_bind) and bool(self.raft_peers)


@lru_cache
def get_settings() -> Settings:
    return Settings()
