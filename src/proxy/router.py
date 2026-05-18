"""Ordered provider chain protected by per-provider circuit breakers.

Implements the ``Provider`` Protocol so the rest of the system treats it as
a single provider. On miss, walks the chain in order, skipping providers
whose breaker is open, recording failures as they happen, and raising
``ProviderChainExhaustedError`` when no provider remains.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from src.metrics import PROVIDER_CALLS, PROVIDER_CHAIN_EXHAUSTED, PROVIDER_FALLBACK
from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
)
from src.proxy.circuit_breaker import CircuitBreaker


class ProviderRouter:
    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        failure_threshold: int,
        failure_window_seconds: float,
        recovery_seconds: float,
    ) -> None:
        if not providers:
            raise ValueError("empty provider chain")
        self._providers: list[Provider] = list(providers)
        self._breakers: list[CircuitBreaker] = [
            CircuitBreaker(
                name=p.name,
                failure_threshold=failure_threshold,
                failure_window_seconds=failure_window_seconds,
                recovery_seconds=recovery_seconds,
            )
            for p in self._providers
        ]
        self.name = f"chain[{','.join(p.name for p in self._providers)}]"

    async def complete(self, prompt: str) -> str:
        last_attempted: str | None = None
        for provider, breaker in zip(self._providers, self._breakers, strict=True):
            now = time.monotonic()
            if not breaker.allow(now):
                if last_attempted is not None:
                    PROVIDER_FALLBACK.labels(
                        **{"from": last_attempted, "to": provider.name, "reason": "breaker_open"}
                    ).inc()
                last_attempted = provider.name
                continue
            try:
                result = await provider.complete(prompt)
            except ProviderTimeoutError:
                PROVIDER_CALLS.labels(provider=provider.name, result="timeout").inc()
                breaker.record_failure(time.monotonic())
                PROVIDER_FALLBACK.labels(
                    **{"from": provider.name, "to": "next", "reason": "timeout"}
                ).inc()
                last_attempted = provider.name
                continue
            except ProviderAPIError:
                PROVIDER_CALLS.labels(provider=provider.name, result="api_error").inc()
                breaker.record_failure(time.monotonic())
                PROVIDER_FALLBACK.labels(
                    **{"from": provider.name, "to": "next", "reason": "api_error"}
                ).inc()
                last_attempted = provider.name
                continue
            PROVIDER_CALLS.labels(provider=provider.name, result="ok").inc()
            breaker.record_success(time.monotonic())
            return result
        PROVIDER_CHAIN_EXHAUSTED.inc()
        raise ProviderChainExhaustedError("all providers unavailable")
