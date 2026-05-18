"""Per-provider, in-process circuit breaker.

State machine::

    closed ──[>=N failures in window]──> open
       ^                                  |
       |                                  | recovery_seconds elapsed
       |       [probe success]            v
       +────────────────────────────── half-open
                                          |
                                          | [probe failure]
                                          +─> open (timer resets)

All time values are caller-supplied (seconds, monotonic-equivalent). The
breaker holds no clock of its own so it is fully deterministic under test.
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum

from src.metrics import PROVIDER_CIRCUIT_STATE


class CircuitState(IntEnum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int,
        failure_window_seconds: float,
        recovery_seconds: float,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._failure_window = failure_window_seconds
        self._recovery = recovery_seconds
        self._failures: deque[float] = deque()
        self._state: CircuitState = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._probe_in_flight: bool = False
        self._publish()

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow(self, now: float) -> bool:
        """Decide whether a new call is permitted at time ``now``.

        Returns True for the *first* caller per half-open window (the probe);
        subsequent callers get False until the probe resolves.
        """
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None
            if now - self._opened_at < self._recovery:
                return False
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = True
            self._publish()
            return True
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self, now: float) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._probe_in_flight = False
            self._failures.clear()
            self._publish()
            return
        self._failures.clear()

    def record_failure(self, now: float) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._probe_in_flight = False
            self._publish()
            return
        self._failures.append(now)
        cutoff = now - self._failure_window
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        if len(self._failures) >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._publish()

    def _publish(self) -> None:
        PROVIDER_CIRCUIT_STATE.labels(provider=self.name).set(int(self._state))
