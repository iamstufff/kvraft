"""Benchmark harness: fires a stream of /query requests and reports latency + hit rate.

Usage:
    python scripts/bench.py --host http://127.0.0.1:8000 \
        --dataset benchmarks/dataset.json \
        --requests 200 --concurrency 8 \
        --out benchmarks/results/run.csv

Outputs a CSV (one row per request: idx, prompt, cached, latency_ms, status) and a
summary line (RPS, P50/P95/P99 ms, hit rate).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

HTTP_OK = 200


@dataclass
class Result:
    idx: int
    prompt: str
    cached: bool | None
    latency_ms: float
    status: int


async def _one(
    session: aiohttp.ClientSession,
    host: str,
    idx: int,
    prompt: str,
    sem: asyncio.Semaphore,
) -> Result:
    async with sem:
        started = time.perf_counter()
        try:
            async with session.post(f"{host}/query", json={"prompt": prompt}) as resp:
                status = resp.status
                body = await resp.json() if status == HTTP_OK else None
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            return Result(idx=idx, prompt=prompt, cached=None, latency_ms=elapsed, status=-1)
        elapsed = (time.perf_counter() - started) * 1000
        cached = bool(body["cached"]) if isinstance(body, dict) and "cached" in body else None
        return Result(idx=idx, prompt=prompt, cached=cached, latency_ms=elapsed, status=status)


async def run(args: argparse.Namespace) -> int:
    prompts: list[str] = json.loads(Path(args.dataset).read_text())
    rng = random.Random(args.seed)
    stream = [rng.choice(prompts) for _ in range(args.requests)]
    sem = asyncio.Semaphore(args.concurrency)

    timeout = aiohttp.ClientTimeout(total=60)
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_one(session, args.host, i, p, sem) for i, p in enumerate(stream)]
        results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - started

    ok = [r for r in results if r.status == HTTP_OK]
    latencies = sorted(r.latency_ms for r in ok)
    if not latencies:
        print("no successful responses — is the server up?", file=sys.stderr)
        return 1
    hits = sum(1 for r in ok if r.cached)
    hit_rate = hits / len(ok)
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]
    rps = len(ok) / wall

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["idx", "prompt", "cached", "latency_ms", "status"])
            for r in results:
                writer.writerow([r.idx, r.prompt, r.cached, f"{r.latency_ms:.3f}", r.status])

    print(
        f"requests={len(ok)}/{len(results)}  rps={rps:.1f}  "
        f"p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  "
        f"hit_rate={hit_rate:.1%}  wall={wall:.1f}s"
    )
    return 0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="benchmarks/dataset.json")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse())))
