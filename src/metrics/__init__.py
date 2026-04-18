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

__all__ = [
    "CONTENT_TYPE_LATEST",
    "LEADER_STATE",
    "PROVIDER_CALLS",
    "QUERY_LATENCY",
    "QUERY_TOTAL",
    "generate_latest",
]
