"""Compare three caching strategies on the same dataset.

Strategies:
  * ``none``             — every request hits the upstream provider.
  * ``redis-exact``      — sha256(prompt) lookup against Redis; SET on miss.
  * ``kvraft-semantic``  — POST /query against the running kvraft cluster.

Usage:
    python scripts/bench_compare.py \
        --strategies none,redis-exact,kvraft-semantic \
        --requests 200 --concurrency 8 \
        --dataset benchmarks/dataset.json \
        --kvraft-host http://127.0.0.1:8001 \
        --redis-url redis://127.0.0.1:6379/0 \
        --out benchmarks/results/compare.csv \
        --plot benchmarks/results/compare.png
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
from numpy.typing import NDArray

from src.cache.embedding import embed

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

HTTP_OK = 200


@dataclass
class Result:
    strategy: str
    idx: int
    prompt: str
    cached: bool
    latency_ms: float
    status: int


@dataclass
class CalibrationRow:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    return (
        statistics.median(s),
        s[int(0.95 * (len(s) - 1))],
        s[int(0.99 * (len(s) - 1))],
    )


def _load_dataset(path: str) -> list[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [p for p in raw if isinstance(p, str) and p.strip()]


def _run_offline_hit_rate(prompts: list[str], threshold: float) -> dict[str, list[Result]]:
    """Replay prompts locally to compare exact-match vs semantic-hit behavior.

    This mode does not call any LLM provider or Redis server. It is intended for
    validating the hit-rate claim itself before spending provider calls on a
    full latency benchmark.
    """
    exact_seen: set[str] = set()
    exact_results: list[Result] = []
    semantic_seen: list[NDArray[np.float32]] = []
    semantic_results: list[Result] = []

    for idx, prompt in enumerate(prompts):
        exact_cached = prompt in exact_seen
        exact_results.append(
            Result(
                strategy="redis-exact-offline",
                idx=idx,
                prompt=prompt,
                cached=exact_cached,
                latency_ms=0.0,
                status=HTTP_OK,
            )
        )
        exact_seen.add(prompt)

        started = time.perf_counter()
        embedding = embed(prompt)
        semantic_cached = any(
            float(np.dot(embedding, previous)) >= threshold for previous in semantic_seen
        )
        semantic_results.append(
            Result(
                strategy="kvraft-semantic-offline",
                idx=idx,
                prompt=prompt,
                cached=semantic_cached,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                status=HTTP_OK,
            )
        )
        semantic_seen.append(embedding)

    return {
        "redis-exact-offline": exact_results,
        "kvraft-semantic-offline": semantic_results,
    }


def _calibrate_thresholds(
    prompts: list[str],
    group_size: int,
    thresholds: list[float],
) -> list[CalibrationRow]:
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1")

    embeddings = [embed(prompt) for prompt in prompts]
    labels = [idx // group_size for idx in range(len(prompts))]
    rows: list[CalibrationRow] = []

    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                same_intent = labels[i] == labels[j]
                similar = float(np.dot(embeddings[i], embeddings[j])) >= threshold
                if same_intent and similar:
                    tp += 1
                elif same_intent:
                    fn += 1
                elif similar:
                    fp += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            CalibrationRow(
                threshold=threshold,
                true_positive=tp,
                false_positive=fp,
                true_negative=tn,
                false_negative=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return rows


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def _write_calibration(rows: list[CalibrationRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "threshold",
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
                "precision",
                "recall",
                "f1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.threshold:.3f}",
                    row.true_positive,
                    row.false_positive,
                    row.true_negative,
                    row.false_negative,
                    f"{row.precision:.6f}",
                    f"{row.recall:.6f}",
                    f"{row.f1:.6f}",
                ]
            )


async def _no_cache(
    session: aiohttp.ClientSession,
    host: str,
    idx: int,
    prompt: str,
    sem: asyncio.Semaphore,
) -> Result:
    async with sem:
        started = time.perf_counter()
        async with session.post(
            f"{host}/query",
            json={"prompt": f"NOCACHE-{idx}-{prompt}"},
        ) as resp:
            status = resp.status
            if status == HTTP_OK:
                await resp.json()
        return Result(
            strategy="none",
            idx=idx,
            prompt=prompt,
            cached=False,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
        )


async def _redis_exact(
    redis_client,  # type: ignore[no-untyped-def]
    session: aiohttp.ClientSession,
    host: str,
    idx: int,
    prompt: str,
    sem: asyncio.Semaphore,
) -> Result:
    key = "redis:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    async with sem:
        started = time.perf_counter()
        cached_value = await redis_client.get(key)
        if cached_value is not None:
            latency = (time.perf_counter() - started) * 1000.0
            return Result(
                strategy="redis-exact",
                idx=idx,
                prompt=prompt,
                cached=True,
                latency_ms=latency,
                status=HTTP_OK,
            )
        async with session.post(
            f"{host}/query",
            json={"prompt": f"REDIS-{idx}-{prompt}"},
        ) as resp:
            status = resp.status
            if status == HTTP_OK:
                body = await resp.json()
                await redis_client.set(key, body["response"])
        latency = (time.perf_counter() - started) * 1000.0
        return Result(
            strategy="redis-exact",
            idx=idx,
            prompt=prompt,
            cached=False,
            latency_ms=latency,
            status=status,
        )


async def _kvraft_semantic(
    session: aiohttp.ClientSession,
    host: str,
    idx: int,
    prompt: str,
    sem: asyncio.Semaphore,
) -> Result:
    async with sem:
        started = time.perf_counter()
        async with session.post(f"{host}/query", json={"prompt": prompt}) as resp:
            status = resp.status
            cached = False
            if status == HTTP_OK:
                body = await resp.json()
                cached = bool(body.get("cached", False))
        latency = (time.perf_counter() - started) * 1000.0
        return Result(
            strategy="kvraft-semantic",
            idx=idx,
            prompt=prompt,
            cached=cached,
            latency_ms=latency,
            status=status,
        )


async def _run_strategy(
    strategy: str,
    prompts: list[str],
    concurrency: int,
    kvraft_host: str,
    redis_url: str,
) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        if strategy == "none":
            tasks = [_no_cache(session, kvraft_host, i, p, sem) for i, p in enumerate(prompts)]
        elif strategy == "redis-exact":
            if aioredis is None:
                print("[skip] redis package not installed; skipping redis-exact strategy.")
                return []
            redis_client = aioredis.from_url(redis_url, decode_responses=True)
            try:
                await redis_client.ping()
            except Exception as exc:
                print(f"[skip] redis unreachable at {redis_url}: {exc}; skipping.")
                await redis_client.aclose()
                return []
            await redis_client.flushdb()
            try:
                tasks = [
                    _redis_exact(redis_client, session, kvraft_host, i, p, sem)
                    for i, p in enumerate(prompts)
                ]
                return await asyncio.gather(*tasks)
            finally:
                await redis_client.aclose()
        elif strategy == "kvraft-semantic":
            tasks = [
                _kvraft_semantic(session, kvraft_host, i, p, sem) for i, p in enumerate(prompts)
            ]
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        return await asyncio.gather(*tasks)


def _summarize(results: list[Result]) -> dict[str, float]:
    if not results:
        return {"hits": 0, "total": 0, "hit_rate": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    latencies = [r.latency_ms for r in results if r.status == HTTP_OK]
    hits = sum(1 for r in results if r.cached)
    p50, p95, p99 = _percentiles(latencies)
    return {
        "hits": hits,
        "total": len(results),
        "hit_rate": hits / len(results),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def _maybe_plot(per_strategy: dict[str, list[Result]], out: Path) -> None:
    if plt is None:
        print("[skip] matplotlib not installed; skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    strategies = list(per_strategy.keys())
    hit_rates = [_summarize(per_strategy[s])["hit_rate"] * 100.0 for s in strategies]
    ax1.bar(strategies, hit_rates, color=["#999", "#4C72B0", "#55A868"])
    ax1.set_ylabel("Hit rate (%)")
    ax1.set_title("Cache hit rate by strategy")
    ax1.set_ylim(0, 100)
    for s in strategies:
        latencies = sorted(r.latency_ms for r in per_strategy[s] if r.status == HTTP_OK)
        if not latencies:
            continue
        xs = latencies
        ys = [(i + 1) / len(latencies) for i in range(len(latencies))]
        ax2.plot(xs, ys, label=s)
    ax2.set_xscale("log")
    ax2.set_xlabel("Latency (ms, log scale)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Latency CDF by strategy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"[ok] plot written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        default="none,redis-exact,kvraft-semantic",
        help="Comma-separated list of strategies to run.",
    )
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dataset", default="benchmarks/dataset.json")
    parser.add_argument("--kvraft-host", default="http://127.0.0.1:8001")
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--out", default="benchmarks/results/compare.csv")
    parser.add_argument("--plot", default="benchmarks/results/compare.png")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--offline-hit-rate",
        action="store_true",
        help="Replay dataset locally: Redis exact-match vs embedding-similarity hit rate.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Embedding cosine threshold used by --offline-hit-rate.",
    )
    parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Evaluate precision/recall across thresholds using dataset groups as labels.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Number of adjacent prompts representing one intent for --calibrate-thresholds.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40",
        help="Comma-separated thresholds for --calibrate-thresholds.",
    )
    args = parser.parse_args()

    base = _load_dataset(args.dataset)
    if args.calibrate_thresholds:
        prompts = base[: args.requests]
        thresholds = _parse_thresholds(args.thresholds)
        print(
            f"[run] calibrate-thresholds requests={len(prompts)} "
            f"group_size={args.group_size} thresholds={thresholds}"
        )
        rows = _calibrate_thresholds(prompts, args.group_size, thresholds)
        _write_calibration(rows, Path(args.out))
        print(f"[ok] calibration csv written to {args.out}")
        for row in rows:
            print(
                f"[calibration] threshold={row.threshold:.2f} "
                f"precision={row.precision*100:.1f}% "
                f"recall={row.recall*100:.1f}% "
                f"f1={row.f1*100:.1f}% "
                f"tp={row.true_positive} fp={row.false_positive}"
            )
        return 0

    if args.offline_hit_rate:
        prompts = base[: args.requests]
        print(
            f"[run] offline-hit-rate requests={len(prompts)} "
            f"threshold={args.similarity_threshold}"
        )
        per_strategy = _run_offline_hit_rate(prompts, args.similarity_threshold)
    else:
        rng = random.Random(args.seed)
        prompts = [rng.choice(base) for _ in range(args.requests)]

        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        per_strategy: dict[str, list[Result]] = {}
        for strategy in strategies:
            print(f"[run] strategy={strategy} requests={args.requests} c={args.concurrency}")
            per_strategy[strategy] = asyncio.run(
                _run_strategy(strategy, prompts, args.concurrency, args.kvraft_host, args.redis_url)
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strategy", "idx", "prompt", "cached", "latency_ms", "status"])
        for _strategy, results in per_strategy.items():
            for r in results:
                writer.writerow(
                    [r.strategy, r.idx, r.prompt, r.cached, f"{r.latency_ms:.3f}", r.status]
                )
    print(f"[ok] csv written to {out_path}")

    for strategy, results in per_strategy.items():
        s = _summarize(results)
        print(
            f"[summary] {strategy:18s} hits={s['hits']}/{s['total']} "
            f"hit_rate={s['hit_rate']*100:.1f}% "
            f"P50={s['p50_ms']:.1f}ms P95={s['p95_ms']:.1f}ms P99={s['p99_ms']:.1f}ms"
        )

    _maybe_plot(per_strategy, Path(args.plot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
