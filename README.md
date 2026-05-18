# kvraft

> A distributed semantic cache for LLM API calls. Raft-replicated across 3 nodes, with HNSW-based embedding similarity for cache lookups.

## What it is

`kvraft` sits in front of an LLM provider (Google Gemini today; OpenAI / Anthropic hook into the same `Provider` protocol) and caches responses keyed by the **embedding of the prompt**, not the prompt string. Semantically similar prompts — "explain SQL injection" vs. "what is SQLi?" — hit the same cache entry.

The cache state machine is replicated across three nodes via Raft (`pysyncobj`), so the cluster survives a node failure without losing cached responses or electing two leaders.

## Why semantic caching

Traditional reverse proxies cache on the exact prompt string. LLM workloads rarely hit that: users paraphrase, tools rewrite, agents regenerate. Embedding the prompt with `sentence-transformers/all-MiniLM-L6-v2` and doing an ANN lookup in an `hnswlib` index turns "any near-duplicate of a prior prompt" into a cache hit — which is cheaper and orders of magnitude faster than a provider round-trip.

## Architecture

```
                ┌───────────────────────────────────────────────────────┐
                │                    3-node kvraft cluster              │
                │                                                       │
   Client ───▶  │  FastAPI (8000)                                       │
                │   │                                                   │
                │   ├─ embed(prompt)  ──▶  HNSW index  ──▶  hit?        │
                │   │                                     │             │
                │   │        yes: return cached response  │             │
                │   │                                     │             │
                │   │        no:  ──▶  Gemini provider ──▶│             │
                │   │                                     │             │
                │   └─ cache.put(prompt, response, embed) ▼             │
                │                   │                                   │
                │                   ▼                                   │
                │            pysyncobj Raft log  ◀──▶ replicas 2 & 3    │
                │                                                       │
                │  Prometheus scrapes /metrics on every node            │
                └───────────────────────────────────────────────────────┘
```

- Reads (`get_or_miss`) are local: every replica answers from its own HNSW index.
- Writes (`put`) go through `@replicated`; the leader appends to the Raft log, the commit fires on a majority, and each replica re-applies deterministically from `(value, embedding_bytes, op_time, ttl_seconds)`.
- `LEADER_STATE` is refreshed on every `/metrics` scrape so Prometheus and the `kill-leader.sh` demo script can identify the current leader.

## Production proxy features

| Feature | What it does |
|---|---|
| **Multi-provider failover** | Ordered chain `gemini → openai → anthropic`. The first healthy provider serves the request. Configure via `PROVIDER_CHAIN`. Current defaults: Gemini `gemini-3-flash-preview`, OpenAI `gpt-5.4-mini`. |
| **Circuit breakers** | Per-provider, in-process state machine (`closed → open → half-open → closed`). Trips after 5 failures in 30s; probes the provider every 15s after that. |
| **Semantic single-flight** | A burst of paraphrased prompts on a cold cache collapses to one upstream call. Coalescing key is embedding-cosine, not a string hash. |
| **TTL on top of LRU** | Entries expire after `CACHE_TTL_SECONDS` (default 3600s). The leader stamps `op_time` on every Raft write so all replicas agree on `expires_at` without depending on follower clocks. |
| **Redis comparison bench** | `scripts/bench_compare.py` can run live latency comparisons or an offline hit-rate replay for Redis exact-match vs embedding-similarity cache behavior. |

## Endpoints

| Route | Purpose |
|---|---|
| `POST /query` | `{"prompt": str, "provider": "gemini"\|"openai"\|"anthropic"}` → `{"response": str, "cached": bool, "similarity": float \| null}` |
| `GET /health` | Liveness, returns `{"status":"ok","node_id":...}` |
| `GET /metrics` | Prometheus exposition: `kvraft_query_total`, `kvraft_query_latency_seconds`, `kvraft_provider_calls_total`, `kvraft_leader_state`, `kvraft_cache_evictions_total`, `kvraft_cache_live_entries`, `kvraft_cache_soft_deleted_entries`, `kvraft_cache_rebuilds_total`, `kvraft_provider_circuit_state`, `kvraft_provider_fallback_total`, `kvraft_provider_chain_exhausted_total`, `kvraft_singleflight_coalesced_total`, `kvraft_cache_ttl_evictions_total` |

## Quickstart (single node, local Python)

```bash
uv sync --extra dev
export GEMINI_API_KEY=...        # or put in .env
export OPENAI_API_KEY=...        # optional fallback
export PROVIDER_CHAIN=gemini,openai
./.venv/bin/uvicorn src.api:app --port 8000
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is SQL injection?"}'
# second call returns cached:true
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain SQLi."}'
```

## Quickstart (3-node Raft cluster)

Two paths, depending on whether Docker is available. Both produce the same behavior — three FastAPI servers on 8001/8002/8003 replicating cache writes via pysyncobj Raft on 4321/4322/4323.

### Docker (preferred for CI / demo)

```bash
# .env must contain GEMINI_API_KEY
docker compose up --build

curl -X POST http://127.0.0.1:8001/query -H 'Content-Type: application/json' \
     -d '{"prompt":"What is SSRF?"}'

# Prometheus at http://127.0.0.1:9090
# Trigger a leader-failover demo:
./scripts/kill-leader.sh
```

### Local processes (no Docker required)

```bash
./scripts/run-local-cluster.sh        # spawns 3 uvicorn processes
curl -X POST http://127.0.0.1:8001/query -H 'Content-Type: application/json' \
     -d '{"prompt":"What is SSRF?"}'
./scripts/kill-local-leader.sh        # kill -9 the leader, time re-election
```

PIDs and per-node logs land under `/tmp/kvraft-local/`.

## Benchmarks

The harness at `scripts/bench.py` fires `--requests` parallel `/query` calls against the cluster, samples prompts from `benchmarks/dataset.json` (200 security-question prompts organized as 40 topics × 5 paraphrases), and reports RPS, P50/P95/P99 latency, and cache hit rate.

```bash
uv sync --extra bench
./.venv/bin/python scripts/bench.py \
  --host http://127.0.0.1:8001 \
  --requests 200 --concurrency 8 \
  --out benchmarks/results/run.csv
```

Headline numbers from a live run against `gemini-2.5-flash-lite`, sampled from `benchmarks/dataset.json` before the Gemini 3 Flash default update:

| Metric | Single-node cold (upstream) | Single-node warm | **3-node cluster, warm** |
|---|---:|---:|---:|
| P50 latency | 6.6 s | 21 ms | **38 ms** |
| P95 latency | 17.0 s | 40 ms | **58 ms** |
| P99 latency | — | — | **58 ms** |
| Cache hit rate | 27 % | 100 % | **100 %** |
| Throughput | 0.3 RPS (provider-bound) | 3.2 RPS | **198 RPS** (c=8) |

The cache-hit path is ~170× faster than a Gemini round-trip, and the 3-node cluster serves ~200 req/s on a single endpoint with hit latency still under 60 ms at P99.

**Leader-failover recovery:** ~1.9 s end-to-end on the local 3-process cluster (leader killed with `kill -9`; surviving pair elects a new leader, and reads on the new leader return the previously-replicated cache entry with `cached=true`). Reproduce with `./scripts/run-local-cluster.sh` + `./scripts/kill-local-leader.sh`, or the Docker Compose variants `docker compose up` + `./scripts/kill-leader.sh`.

## How does kvraft compare to Redis?

Two comparison modes are available:

```bash
# No provider calls; validates the hit-rate claim against local embeddings.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/bench_compare.py \
  --offline-hit-rate --requests 200 --similarity-threshold 0.8 \
  --out benchmarks/results/compare_offline_default_threshold.csv \
  --plot benchmarks/results/compare_offline_default_threshold.png

# Full live run; requires a running kvraft cluster, Redis, and provider API budget.
./.venv/bin/python scripts/bench_compare.py \
  --strategies none,redis-exact,kvraft-semantic \
  --requests 200 --concurrency 8
```

Current offline replay over the 200-prompt paraphrase dataset at the production default `SIMILARITY_THRESHOLD=0.8`:

| Strategy | Hit rate | Artifact |
|---|---:|---|
| Redis exact-match | 0.0% | `benchmarks/results/compare_offline_default_threshold.csv` |
| kvraft semantic, threshold 0.8 | 0.5% | `benchmarks/results/compare_offline_default_threshold.csv` |

This result is intentionally kept in the README because it prevents overclaiming: MiniLM at a conservative `0.8` threshold is excellent for near-duplicate prompts but does not catch most broad paraphrases in this dataset. Lower thresholds increase recall but need a precision/false-positive evaluation before they belong in a resume claim.

## Project layout

```
src/
├── api/          # FastAPI routes (/query, /health, /metrics)
├── cache/        # Embedding + HNSW + semantic lookup
├── concurrency/  # Semantic single-flight request coalescing
├── proxy/        # Provider protocol, router, circuit breakers, provider clients
├── raft/         # pysyncobj-backed replicated state machine
├── metrics/      # Prometheus instruments (shared registry)
└── config.py     # Pydantic settings

tests/
├── unit/         # Mocked, fast
└── integration/  # Real MiniLM model, marked `integration`

scripts/
├── bench.py               # Benchmark harness
├── bench_compare.py       # No-cache vs Redis exact-match vs kvraft semantic
├── kill-leader.sh         # Failover demo (Docker variant)
├── run-local-cluster.sh   # 3-process cluster without Docker
└── kill-local-leader.sh   # Failover demo (local-process variant)
```

## Design decisions

- **Embeddings are shipped over Raft as raw `float32` bytes.** The alternative — each replica re-embedding on apply — would double CPU and require the embedding model to be bit-identical across nodes. Bytes are cheaper and deterministic.
- **Reads are local; only writes replicate.** pysyncobj's `@replicated` is sync-committed for writes; `get_or_miss` hits the local HNSW index directly. This is strong-consistency for writes, monotonic-read consistency for reads.
- **LRU eviction is leader-only; deletes replicate through Raft.** The cache is bounded at `MAX_CAPACITY` (default 10k live entries). When the leader fills the bucket it picks its own LRU victim and emits an `_apply_delete` Raft op; followers apply the delete identically. Follower reads don't feed back into eviction ordering — we accept approximate LRU in exchange for strictly convergent state across replicas. `hnswlib` deletes are soft, so each replica deterministically rebuilds its index once the soft-deleted fraction crosses `REBUILD_THRESHOLD` (default 0.3).
- **Circuit breakers and single-flight are local to each replica.** Provider health is intentionally not Raft-replicated, and in-flight request coalescing is process-local. Only deterministic cache state belongs in the Raft log.
- **TTL uses leader-stamped time.** The leader includes `op_time` in every replicated put; each replica derives the same `expires_at` value and treats expired local reads as misses.
- **Single-stage Docker image.** Torch + sentence-transformers are heavy; splitting to multi-stage will come once the layers stabilize (Day 3 stretch).

## Future work

- Multi-region Raft
- TLS between replicas + auth on the public API
- Kubernetes manifests (current deploy target is Docker Compose)
- GPU-backed batched embedding for large concurrent load
