"""Prometheus instrumentation for the kvraft node.

Counters and histograms live on the default process-wide REGISTRY. The FastAPI
``/metrics`` endpoint renders them via ``prometheus_client.generate_latest``.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

QUERY_TOTAL = Counter(
    "kvraft_query_total",
    "Total /query requests, labeled by outcome.",
    ["outcome"],  # hit | miss | provider_error
)

QUERY_LATENCY = Histogram(
    "kvraft_query_latency_seconds",
    "End-to-end latency of /query requests in seconds.",
    ["outcome"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PROVIDER_CALLS = Counter(
    "kvraft_provider_calls_total",
    "Calls forwarded to the upstream LLM provider, labeled by provider + result.",
    ["provider", "result"],  # result: ok | timeout | api_error
)

LEADER_STATE = Gauge(
    "kvraft_leader_state",
    "Whether this node currently believes it is the Raft leader (1=leader, 0=follower).",
)

CACHE_EVICTIONS = Counter(
    "kvraft_cache_evictions_total",
    "LRU evictions applied through the Raft log (or locally in non-Raft mode).",
)

CACHE_LIVE_ENTRIES = Gauge(
    "kvraft_cache_live_entries",
    "Entries present in the cache and answerable by search (excludes soft-deleted).",
)

CACHE_SOFT_DELETED_ENTRIES = Gauge(
    "kvraft_cache_soft_deleted_entries",
    "Entries marked deleted in the HNSW graph awaiting rebuild.",
)

CACHE_REBUILDS = Counter(
    "kvraft_cache_rebuilds_total",
    "HNSW index rebuilds executed to reclaim soft-deleted memory.",
)

PROVIDER_CIRCUIT_STATE = Gauge(
    "kvraft_provider_circuit_state",
    "Circuit-breaker state per provider: 0=closed, 1=open, 2=half-open.",
    ["provider"],
)

PROVIDER_FALLBACK = Counter(
    "kvraft_provider_fallback_total",
    "Fallthroughs from one provider to the next, labeled by source, target, and reason.",
    ["from", "to", "reason"],
)

PROVIDER_CHAIN_EXHAUSTED = Counter(
    "kvraft_provider_chain_exhausted_total",
    "Times the entire provider chain failed in a single request (pages oncall).",
)

SINGLEFLIGHT_COALESCED = Counter(
    "kvraft_singleflight_coalesced_total",
    "Concurrent miss requests joined to an existing in-flight upstream call.",
)

CACHE_TTL_EVICTIONS = Counter(
    "kvraft_cache_ttl_evictions_total",
    "Cache entries evicted because their TTL expired (separate from LRU pressure).",
)

__all__ = [
    "CACHE_EVICTIONS",
    "CACHE_LIVE_ENTRIES",
    "CACHE_REBUILDS",
    "CACHE_SOFT_DELETED_ENTRIES",
    "CACHE_TTL_EVICTIONS",
    "CONTENT_TYPE_LATEST",
    "LEADER_STATE",
    "PROVIDER_CALLS",
    "PROVIDER_CHAIN_EXHAUSTED",
    "PROVIDER_CIRCUIT_STATE",
    "PROVIDER_FALLBACK",
    "QUERY_LATENCY",
    "QUERY_TOTAL",
    "SINGLEFLIGHT_COALESCED",
    "generate_latest",
]
