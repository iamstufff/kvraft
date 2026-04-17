# kvraft

> A distributed semantic cache for LLM API calls. Raft-replicated across 3 nodes, with HNSW-based embedding similarity for cache lookups.

**Status:** Under construction — this README will be rewritten on Day 3 with architecture diagram, benchmarks, and quickstart.

## What it is

`kvraft` sits in front of an LLM provider (Google Gemini, with OpenAI and Anthropic as extensions) and caches responses keyed by the *embedding* of the prompt, not the prompt string. Semantically similar prompts hit the same cache entry.

The cache is replicated across 3 nodes via Raft consensus (`pysyncobj`), so the cluster survives a node failure without losing state.

## Why

Traditional LLM proxies cache on exact prompt match — "what is SQL injection?" and "explain SQL injection" miss the cache. `kvraft` embeds the prompt, does an ANN lookup in an HNSW index, and returns the cached response if similarity > threshold.

## Planned benchmarks (populated Day 3)

| Metric | Value |
|---|---|
| Throughput (RPS) | _TBD_ |
| P99 latency | _TBD_ |
| Cache hit rate on test workload | _TBD_ |
| Leader-failover recovery time | _TBD_ |

## Architecture (diagram pending Day 3)

Client → FastAPI node → [embedding → HNSW lookup] → hit? return. miss? → proxy to Gemini → cache response → replicate via Raft.

## Quickstart (pending Day 3)

```bash
# dependencies
uv sync

# run the 3-node cluster
docker compose up

# query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is SQL injection?"}'
```

## Future work

- Kubernetes deployment manifests
- TLS + auth
- Multi-region Raft
- Additional providers (OpenAI, Anthropic)
- GPU-backed batched embedding
