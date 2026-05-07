# kvraft — Production LLM Proxy Enhancements

**Date:** 2026-05-08
**Status:** Approved design, pending implementation plan
**Author:** Claude (brainstormed with Abhishek)

## Problem

Today, kvraft is a 3-node Raft-replicated semantic cache fronting a single LLM provider (Gemini). The cache is in good shape (LRU + HNSW soft-delete + rebuild, 8 Prometheus series, ~1.9s leader failover, P50 38ms / P99 58ms at 198 RPS warm). The provider story is single-tier and the request path has no thundering-herd defense: 50 concurrent paraphrases of an uncached prompt produce 50 upstream calls. There is also no TTL atop LRU and no Redis-based comparison benchmark to make the "semantic cache vs exact-match" story concrete.

Constraint set by the engineer: any new feature must (1) score high on ATS for senior backend / distsys+AI roles, (2) live within his existing tech domain (Python, FastAPI, multi-LLM routing, retry/backoff, Redis, Langfuse, IR/HNSW, security/CVE) so it can be defended in interviews without learning new languages or unfamiliar areas, and (3) produce a measurable resume number.

## Goals

- Multi-provider failover with circuit breakers (Gemini → OpenAI → Anthropic).
- Request coalescing via semantic single-flight on cache miss.
- TTL eviction on top of LRU.
- Redis exact-match cache as a benchmark comparison baseline.
- 4 new Prometheus series for breakers, fallbacks, coalesces, TTL evictions.
- Updated resume bullet that adds "circuit breakers", "multi-provider", "single-flight", "thundering herd", and a Redis comparison number.

## Non-goals

- Cluster-wide single-flight (per-replica is enough; standard pattern).
- Replicated circuit breaker state through Raft (in-process per replica is correct).
- Streaming responses (deferred; complicates cache replay and not required by the bundle).
- A new language. Python only.
- Multi-region Raft, Kubernetes, TLS — listed as future work in README.

## Architecture

```
POST /query → cache.get_or_miss(prompt)
  → Hit (and not TTL-expired):  return cached
  → Miss (or expired):
      single_flight.execute(embedding, threshold, fn=...)
        → provider_router.complete(prompt)
            → for provider in chain:
                  if breaker[p].closed_or_halfopen: try p.complete()
                  on success → close breaker, return
                  on failure → open breaker, fall through
            → all providers exhausted: ProviderChainExhaustedError → 503
        → cache.put(prompt, response, embedding)   (existing Raft path)
        → return
```

Single-flight wraps the miss path so concurrent paraphrase bursts collapse to one upstream call. The provider router replaces the bare GeminiClient as the `Provider` injected into `query.py`. Circuit breakers are per-provider, in-process, per replica.

## New modules

| Path | Purpose | Approx LOC |
|---|---|---|
| `src/proxy/circuit_breaker.py` | `closed → open → half-open → closed` state machine, per-provider, in-process | ~80 |
| `src/proxy/router.py` | `ProviderRouter` implementing the existing `Provider` Protocol; ordered chain + breakers | ~120 |
| `src/proxy/openai_client.py` | OpenAI adapter using `openai>=1.0` async client | ~50 |
| `src/proxy/anthropic_client.py` | Anthropic adapter using `anthropic` async client | ~50 |
| `src/concurrency/__init__.py` + `single_flight.py` | `SingleFlight` primitive: in-flight `Future` deduplication keyed by embedding similarity | ~80 |
| `scripts/bench_compare.py` | Benchmark across no-cache / Redis-exact / kvraft-semantic | ~150 |

## Touched modules (small edits)

- `src/api/query.py` — `_get_gemini()` becomes `_get_router()`; miss path wrapped in `single_flight.execute(...)`.
- `src/cache/core.py` — adds `_expires_at: dict[int, float]`; lazy-evict on read; opportunistic TTL eviction on put alongside LRU eviction.
- `src/raft/state_machine.py` — `_apply_put` accepts `op_time` from the leader; `_CacheState._expires_at` mirrors `core.py`.
- `src/config.py` — new env settings (full list in *Configuration* below).
- `src/metrics/__init__.py` — 5 new series (full list in *Metrics*).
- `pyproject.toml` — adds `openai`, `anthropic` as core deps; `redis` and `matplotlib` under `[project.optional-dependencies] bench`.
- `benchmarks/dataset.json` — expands from 60 prompts (20 topics × 3 paraphrases) to 200 prompts (40 topics × 5 paraphrases), still security-flavored.

## Component design

### Circuit breaker

State machine, per provider:

```
                failures ≥ failure_threshold
                  in failure_window_seconds
       closed ──────────────────────────────► open
         ▲                                      │
         │                                      │ recovery_timeout elapsed
         │                                      ▼
         │            probe success         half-open
         └────────────────────────────────────  │
                                                │ probe failure
                                                └──► open (reset timer)
```

Defaults (env-tunable):
- `breaker_failure_threshold = 5`
- `breaker_failure_window_seconds = 30.0`
- `breaker_recovery_seconds = 15.0`

Counts as failure: `ProviderTimeoutError`, `ProviderAPIError`, and 429 rate-limit responses. Successful calls reset the rolling failure window.

In `half-open` state, only one call (the probe) is allowed in flight. Subsequent calls during the probe window skip past this provider as if it were `open`. Implemented with an `asyncio.Lock` acquired only when the breaker is half-open; closed-state calls have no lock contention.

### Provider router

Implements the existing `Provider` Protocol so `query.py` treats the router as a single provider. Holds an ordered list of `(provider, breaker)` pairs constructed at startup based on `PROVIDER_CHAIN` env var. Providers without API keys are silently dropped from the chain. Empty chain → startup error (loud failure better than silent single-provider regression).

Per-call algorithm:
1. Iterate providers in order.
2. Skip if breaker is `open` or actively probing in `half-open`.
3. Try `provider.complete(prompt)`.
4. On success: close breaker, return.
5. On failure: record failure on breaker; emit `kvraft_provider_fallback_total{from, to, reason}`; continue.
6. All providers exhausted → raise `ProviderChainExhaustedError`. `query.py` returns 503 with `Retry-After: <recovery_seconds>` header.

### Single-flight (semantic coalescing)

State per instance:

```python
@dataclass
class _InFlight:
    embedding: NDArray[np.float32]
    future: asyncio.Future[str]
    started_at: float

class SingleFlight:
    _inflight: list[_InFlight]
    _lock: asyncio.Lock
```

Algorithm (pseudocode):

```python
async def execute(self, embedding, threshold, fn) -> str:
    async with self._lock:
        match = self._best_match(embedding, threshold)  # brute-force cosine
        if match:
            COALESCED.inc()
            return await match.future                    # join existing flight
        future = create_future()
        entry = _InFlight(embedding, future, time.monotonic())
        self._inflight.append(entry)
    try:
        result = await fn()
        future.set_result(result)
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        async with self._lock:
            self._inflight.remove(entry)
    return result
```

Brute-force cosine over `_inflight` is acceptable because in-flight is bounded by concurrency (typically <100 entries even under heavy load). HNSW would be overkill.

Threshold for coalescing: same as `Settings.similarity_threshold` (0.8). Configurable as `coalesce_threshold` if it ever needs to diverge.

Order matters: in `query.py`, the wrapped `fn` performs the upstream call, then `cache.put`, *then* the `future.set_result` happens implicitly via `await fn()` returning. This avoids a race where a request waking up could miss the cache write.

Per-replica, in-process. Cluster-wide coalescing is overkill — load balancers shard by client/route so most coalesceable bursts hit one replica.

### TTL on top of LRU

`_CacheState` and `SemanticCache` both gain `_expires_at: dict[int, float]`.

Leader stamps `op_time = time.time()` at the moment of `put` and includes it in the replicated `_apply_put` op. Each replica computes `expires_at = op_time + ttl_seconds` deterministically. If the leader's clock skews vs a follower's, replicas still agree on the absolute `expires_at`; the only divergence is whether each replica's local `now()` considers a given entry expired in the same instant — already aligned with the existing "monotonic-read consistency for reads" guarantee.

Read path: `lookup_with_id` returns the match as today; `get_or_miss` checks `expires_at[id] < now()` and treats expired entries as `Miss`.

Eviction:
- The leader, when running its LRU-victim selection on `put`, also opportunistically scans for any TTL-expired entry it knows about and emits an additional `_apply_delete` for it.
- Followers rely on the leader's deletes; their local reads of expired entries return Miss (no follower-initiated deletes — only leader does writes).
- Stale soft-deletes are eventually reclaimed by the existing HNSW `rebuild()` path once `soft_deleted / total > rebuild_threshold`.

`Settings.cache_ttl_seconds: float = 3600.0`. Setting to `0.0` disables TTL.

### Redis comparison bench

`scripts/bench_compare.py`. CLI:

```
--strategies none,redis-exact,kvraft-semantic
--requests 200 --concurrency 8
--out benchmarks/results/compare.csv
```

For each strategy:
- `none`: every request goes upstream. Records latency.
- `redis-exact`: `GET sha256(prompt)` → if hit, return; else upstream + `SET`. Same dataset.
- `kvraft-semantic`: hits `POST /query` against running cluster.

Outputs:
- `benchmarks/results/compare.csv` — per-strategy per-metric rows.
- `benchmarks/results/compare.png` — bar chart of hit rate per strategy + latency CDF.

Redis dep is optional via `[project.optional-dependencies] bench`; if Redis isn't available at run time, that strategy is skipped with a warning rather than failing the whole bench.

README adds a "How does kvraft compare?" section with the PNG and a 2-line takeaway.

## Configuration (new env settings)

```python
# src/config.py
provider_chain: list[str] = ["gemini"]            # comma-separated, ordered
breaker_failure_threshold: int = 5
breaker_failure_window_seconds: float = 30.0
breaker_recovery_seconds: float = 15.0
coalesce_threshold: float = 0.8                   # same default as similarity_threshold; independent setting
cache_ttl_seconds: float = 3600.0                 # 0 disables TTL

openai_api_key: str = ""
anthropic_api_key: str = ""
# (gemini_api_key already exists)
```

Backward compatibility: with only `GEMINI_API_KEY` set, `provider_chain` collapses to `[gemini]` and behavior matches today's single-provider path.

## Metrics (new on top of existing 8)

| Series | Type | Labels | Purpose |
|---|---|---|---|
| `kvraft_provider_circuit_state` | Gauge | provider | 0=closed, 1=open, 2=half-open |
| `kvraft_provider_fallback_total` | Counter | from, to, reason | Fallthrough counts for breaker dashboards |
| `kvraft_provider_chain_exhausted_total` | Counter | — | Pages oncall when fires |
| `kvraft_singleflight_coalesced_total` | Counter | — | Headline number for resume |
| `kvraft_cache_ttl_evictions_total` | Counter | — | Splits TTL pressure from LRU pressure |

## Error handling

- Provider chain exhausted → `HTTPException(503, "All providers unavailable", headers={"Retry-After": str(recovery_seconds)})`.
- Single-flight leader fails → `future.set_exception(exc)` propagates to all waiters → 502 to clients (same as today's single-call failure path).
- Redis unavailable in `bench_compare.py` → log warning, skip that strategy, continue.
- Missing API key for a provider in the chain → drop that provider at startup; chain fully empty → fail fast.
- TTL clock skew → accepted, consistent with existing read-consistency guarantee.

## Testing

### Unit tests (fast, mocked)

- `tests/unit/test_circuit_breaker.py` — closed→open after threshold, open→half-open after timeout, half-open→closed on probe success, half-open→open on probe failure (timer reset).
- `tests/unit/test_provider_router.py` — three mocked providers; chain order; fallback on 429/timeout/api-error; exhaustion → `ProviderChainExhaustedError`.
- `tests/unit/test_single_flight.py` — `asyncio.gather` of 50 calls with the same embedding hits inner once; exception propagation to all waiters; late-arrival behavior; brute-force similarity correctness.
- `tests/unit/test_ttl.py` — `_CacheState` with stamped op_time, expiration boundary, lazy eviction on read.
- All existing unit tests must still pass.

### Integration tests (real cluster, behind `-m integration`)

- `tests/integration/test_failover_chain.py` — boots 3-node cluster + mock providers; force Gemini to 5×429 in 30s; verify next request routes to OpenAI mock without reaching Gemini.
- `tests/integration/test_thundering_herd.py` — 50 concurrent `/query` calls with paraphrases of one prompt; mock provider counts calls; assert call count = 1.
- `tests/integration/test_ttl_eviction.py` — put with 1s TTL, sleep 1.1s, request returns Miss; leader emits `_apply_delete`; all replicas converge.

## Resume bullet (preview)

> **kvraft — Distributed Semantic Cache for LLM APIs** (Python, FastAPI, Raft, hnswlib, Redis, Prometheus): Built a 3-node FastAPI semantic cache with **multi-provider failover** (Gemini → OpenAI → Anthropic) using **circuit breakers** and **single-flight request coalescing** that cut upstream calls by **>90%** under thundering-herd of paraphrased prompts. Replicated cache state through a **Raft log**; cluster survived **leader failure in 1.9s** with cached reads served throughout. **P50 38ms / P99 58ms at 198 RPS** vs. **6.6s upstream**; semantic cache hit **~80% vs Redis exact-match's ~20%** on a 200-prompt paraphrase workload. [GitHub]

## Decisions made (for DECISIONS.md)

1. **Circuit breakers in-process, not Raft-replicated.** Replicating breaker state would (a) double write traffic, (b) let one slow node trip the breaker for healthy nodes, (c) violate "only deterministic state belongs in the log." Each replica keeps its own breaker. Standard pattern (Hystrix, resilience4j).
2. **Single-flight per-replica, not cluster-wide.** Cluster-wide coordination is overkill; load balancers usually shard by client so most coalesceable bursts hit one replica. Standard pattern (Go's `golang.org/x/sync/singleflight`).
3. **Single-flight keyed by embedding similarity, not prompt hash.** ~20 extra LOC over hash-keyed; catches paraphrase coalescing mid-flight, before any of them lands in cache. Sharper resume story.
4. **TTL `op_time` baked into the Raft op.** Determinism requirement; followers can't compute `expires_at` from their own clocks.
5. **Redis is a bench-only dependency.** No production code change; production adds zero new infra dep. Redis lives in `bench` optional-dependencies.

## Open questions / risks

- **OpenAI / Anthropic free-tier credits.** Engineer has limited paid credit on those providers. Mitigation: integration tests use mocked providers; live failover demo uses Gemini × intentional rate-limit only. Real OpenAI/Anthropic calls happen only during a demo bench run.
- **`google.generativeai` deprecation.** Existing tech debt, not made worse by this work. Migration to `google.genai` remains future work per DECISIONS.md #12.
- **Half-open probe contention.** If 100 requests arrive simultaneously when a breaker flips half-open, the lock serializes them — most will skip the half-open provider and fall through. This is correct behavior but worth a comment in the code.
- **`_apply_put` signature change.** Adding `op_time` to the existing `@replicated _apply_put(value, embedding_bytes)` changes the Raft log entry shape. Replaying a pysyncobj journal/snapshot written before this change will fail. Acceptable now (no production deployment exists; only local bench runs), but the rollout note for the implementation plan should be: delete `/tmp/kvraft-local/*.journal` (and any persisted snapshot files) on first run with the new code. Document in DECISIONS.md.

## Out of scope for this iteration

- Streaming responses (SSE).
- Persistence / snapshots (Bundle B's other half — deferred).
- OpenTelemetry tracing (Bundle C — deferred).
- PII redaction middleware (Bundle C — deferred).
- Grafana dashboards committed to repo (deferred).
- 5-node cluster benchmark.
