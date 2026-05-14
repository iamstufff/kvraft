# Production LLM Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer a multi-provider failover router, in-process circuit breakers, semantic-keyed single-flight request coalescing, TTL-on-top-of-LRU eviction, and a Redis exact-match comparison benchmark on top of the existing 3-node Raft-replicated semantic cache.

**Architecture:** `query.py` calls a `ProviderRouter` (implements the existing `Provider` Protocol) that walks an ordered chain (`gemini → openai → anthropic`) protected by per-provider circuit breakers. Miss path is wrapped in a per-replica `SingleFlight` that coalesces in-flight calls by embedding-cosine similarity. The leader stamps `op_time` on every `_apply_put` so all replicas deterministically compute `expires_at = op_time + ttl_seconds`. A new `bench_compare.py` runs the same workload across no-cache, Redis exact-match, and kvraft semantic so the README can show a Redis-vs-semantic hit-rate gap.

**Tech Stack:** Python 3.11, FastAPI, pysyncobj (Raft), hnswlib, sentence-transformers, openai SDK (≥1.14), anthropic SDK (≥0.25), redis SDK (bench-only), prometheus-client, pytest + pytest-asyncio.

**Rollout note:** Task 9 changes the shape of the `_apply_put` Raft log entry (`value, embedding_bytes` → `value, embedding_bytes, op_time`). Any pysyncobj journal written by the old code is unreplayable. Before the first cluster run on the new code, delete `/tmp/kvraft-local/*.journal` (and any persisted snapshot files). This is acceptable because no production deployment exists; only local bench runs.

---

## Task 0: Land pre-existing pending changes

Before any new work, commit the dangling `asyncio.to_thread` + `threading.Lock` edits already in the working tree so the new feature work builds on a clean baseline.

**Files:**
- Modify: `src/api/query.py` (already modified locally; uses `asyncio.to_thread` around `cache.get_or_miss` / `cache.put`)
- Modify: `src/cache/core.py` (already modified locally; adds `threading.Lock` around `_index` ops)

- [ ] **Step 1: Verify the working-tree diff**

Run: `git diff src/api/query.py src/cache/core.py`
Expected: shows `asyncio.to_thread` wrappers in `query.py` and `threading.Lock` usage in `core.py`. No other files changed.

- [ ] **Step 2: Run the existing test suite**

Run: `pytest tests -m "not integration" -q`
Expected: all green. If anything fails, stop and debug — those edits are someone else's WIP and must pass before we layer new work on top.

- [ ] **Step 3: Commit the pending changes**

```bash
git add src/api/query.py src/cache/core.py
git commit -m "refactor(api,cache): offload sync cache ops to a worker thread"
```

- [ ] **Step 4: Verify clean tree**

Run: `git status --short`
Expected: empty output. Working tree clean before we move on.

---

## Task 1: Extend `Settings` with new env-driven config

Add the six new settings the spec lists. Backward compatibility: with only `GEMINI_API_KEY` set, `provider_chain` collapses to `[gemini]` and the cluster behaves exactly like today.

**Files:**
- Modify: `src/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config.py`:

```python
def test_provider_chain_defaults_to_gemini_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROVIDER_CHAIN", raising=False)
    settings = Settings()
    assert settings.provider_chain == ["gemini"]


def test_provider_chain_parses_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_CHAIN", "gemini,openai,anthropic")
    settings = Settings()
    assert settings.provider_chain == ["gemini", "openai", "anthropic"]


def test_breaker_defaults() -> None:
    settings = Settings()
    assert settings.breaker_failure_threshold == 5
    assert settings.breaker_failure_window_seconds == 30.0
    assert settings.breaker_recovery_seconds == 15.0


def test_ttl_default_one_hour_and_disabled_with_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().cache_ttl_seconds == 3600.0
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")
    assert Settings().cache_ttl_seconds == 0.0


def test_coalesce_threshold_default_matches_similarity_threshold() -> None:
    settings = Settings()
    assert settings.coalesce_threshold == settings.similarity_threshold
```

(Assume `from src.config import Settings` is already imported; if not, add it.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_config.py -k "provider_chain or breaker_defaults or ttl_default or coalesce_threshold" -v`
Expected: FAIL with `AttributeError` for the new fields.

- [ ] **Step 3: Implement the new settings**

Edit `src/config.py`:

```python
"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    provider_chain: list[str] = Field(default_factory=lambda: list(DEFAULT_PROVIDER_CHAIN))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: all pass, including the 5 new tests and any pre-existing config tests.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/unit/test_config.py
git commit -m "feat(config): add provider_chain, breaker thresholds, TTL, and extra API keys"
```

---

## Task 2: Add `ProviderChainExhaustedError`

The router needs a distinct error class so `query.py` can return 503 (instead of 502) when *every* provider is unhealthy.

**Files:**
- Modify: `src/proxy/base.py`
- Test: `tests/unit/test_proxy_base.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_proxy_base.py`:

```python
from src.proxy.base import ProviderChainExhaustedError, ProxyError


def test_chain_exhausted_is_a_proxy_error() -> None:
    err = ProviderChainExhaustedError("all dead")
    assert isinstance(err, ProxyError)
    assert str(err) == "all dead"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_proxy_base.py::test_chain_exhausted_is_a_proxy_error -v`
Expected: FAIL with `ImportError: cannot import name 'ProviderChainExhaustedError'`.

- [ ] **Step 3: Add the class**

Edit `src/proxy/base.py`. After `class ProviderAPIError(ProxyError):`, add:

```python
class ProviderChainExhaustedError(ProxyError):
    """Every provider in the configured chain is open (failed or breaker-tripped)."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_proxy_base.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/proxy/base.py tests/unit/test_proxy_base.py
git commit -m "feat(proxy): introduce ProviderChainExhaustedError"
```

---

## Task 3: Add 5 new Prometheus series

Five new instruments per the spec. No callers yet — those land in Tasks 4, 7, 8, and 9.

**Files:**
- Modify: `src/metrics/__init__.py`

- [ ] **Step 1: Append the new series**

Edit `src/metrics/__init__.py`. Before the `__all__` list, add:

```python
PROVIDER_CIRCUIT_STATE = Gauge(
    "kvraft_provider_circuit_state",
    "Circuit-breaker state per provider: 0=closed, 1=open, 2=half-open.",
    ["provider"],
)

PROVIDER_FALLBACK = Counter(
    "kvraft_provider_fallback_total",
    "Fallthroughs from one provider to the next, labeled by source, target, and reason.",
    ["from", "to", "reason"],  # reason: timeout | api_error | rate_limit | breaker_open
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
```

Then extend `__all__` to include all five names in alphabetical order with the existing entries.

- [ ] **Step 2: Verify import surface**

Run: `python -c "from src.metrics import PROVIDER_CIRCUIT_STATE, PROVIDER_FALLBACK, PROVIDER_CHAIN_EXHAUSTED, SINGLEFLIGHT_COALESCED, CACHE_TTL_EVICTIONS; print('ok')"`
Expected: prints `ok` and exits 0.

- [ ] **Step 3: Confirm existing tests still pass**

Run: `pytest tests -m "not integration" -q`
Expected: all green (no test changes; only an additive metrics surface).

- [ ] **Step 4: Commit**

```bash
git add src/metrics/__init__.py
git commit -m "feat(metrics): add 5 series for breakers, fallbacks, coalesces, TTL evictions"
```

---

## Task 4: `CircuitBreaker` state machine

Per-provider, in-process state machine with three states (`closed`, `open`, `half-open`). Configurable via `Settings.breaker_*`. Updates `PROVIDER_CIRCUIT_STATE` on every transition.

**Files:**
- Create: `src/proxy/circuit_breaker.py`
- Test: `tests/unit/test_circuit_breaker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_circuit_breaker.py`:

```python
import asyncio

import pytest

from src.proxy.circuit_breaker import CircuitBreaker, CircuitState


def _bk(*, threshold: int = 3, window: float = 30.0, recovery: float = 10.0) -> CircuitBreaker:
    return CircuitBreaker(
        name="test",
        failure_threshold=threshold,
        failure_window_seconds=window,
        recovery_seconds=recovery,
    )


def test_starts_closed() -> None:
    assert _bk().state is CircuitState.CLOSED


def test_opens_after_threshold_failures_within_window() -> None:
    bk = _bk(threshold=3)
    bk.record_failure(now=0.0)
    bk.record_failure(now=1.0)
    assert bk.state is CircuitState.CLOSED
    bk.record_failure(now=2.0)
    assert bk.state is CircuitState.OPEN


def test_failures_outside_window_do_not_open() -> None:
    bk = _bk(threshold=3, window=10.0)
    bk.record_failure(now=0.0)
    bk.record_failure(now=20.0)
    bk.record_failure(now=21.0)
    # Only two failures sit inside any single 10s window.
    assert bk.state is CircuitState.CLOSED


def test_success_resets_failure_window() -> None:
    bk = _bk(threshold=3)
    bk.record_failure(now=0.0)
    bk.record_failure(now=1.0)
    bk.record_success(now=2.0)
    bk.record_failure(now=3.0)
    bk.record_failure(now=4.0)
    assert bk.state is CircuitState.CLOSED


def test_allow_returns_false_when_open() -> None:
    bk = _bk(threshold=1)
    bk.record_failure(now=0.0)
    assert bk.state is CircuitState.OPEN
    assert bk.allow(now=1.0) is False


def test_transitions_to_half_open_after_recovery_seconds() -> None:
    bk = _bk(threshold=1, recovery=10.0)
    bk.record_failure(now=0.0)
    assert bk.allow(now=5.0) is False
    assert bk.allow(now=11.0) is True
    assert bk.state is CircuitState.HALF_OPEN


def test_half_open_probe_success_closes_breaker() -> None:
    bk = _bk(threshold=1, recovery=10.0)
    bk.record_failure(now=0.0)
    bk.allow(now=11.0)  # → half-open, probe permit issued
    bk.record_success(now=11.5)
    assert bk.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_and_resets_timer() -> None:
    bk = _bk(threshold=1, recovery=10.0)
    bk.record_failure(now=0.0)
    bk.allow(now=11.0)  # → half-open
    bk.record_failure(now=11.5)  # probe fails
    assert bk.state is CircuitState.OPEN
    # Recovery timer restarts from 11.5, so 21.0 is still inside it.
    assert bk.allow(now=21.0) is False
    assert bk.allow(now=22.0) is True


async def test_half_open_only_one_probe_in_flight() -> None:
    bk = _bk(threshold=1, recovery=0.0)
    bk.record_failure(now=0.0)
    # recovery=0 means we're immediately eligible for half-open
    granted_first = bk.allow(now=1.0)
    granted_second = bk.allow(now=1.0)
    assert granted_first is True
    assert granted_second is False  # probe slot already taken
    bk.record_success(now=1.1)
    # Once closed, allow returns True normally.
    assert bk.allow(now=2.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_circuit_breaker.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `CircuitBreaker`**

Create `src/proxy/circuit_breaker.py`:

```python
"""Per-provider, in-process circuit breaker.

State machine:

    closed ──[≥N failures in window]──▶ open
       ▲                                  │
       │                                  │ recovery_seconds elapsed
       │       [probe success]            ▼
       └────────────────────────────── half-open
                                          │
                                          │ [probe failure]
                                          └─▶ open (timer resets)

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
        # HALF_OPEN: only one probe in flight at a time.
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
        # CLOSED success drops the running failure window.
        self._failures.clear()

    def record_failure(self, now: float) -> None:
        if self._state is CircuitState.HALF_OPEN:
            # Probe failed → reopen and reset the recovery timer.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_circuit_breaker.py -v`
Expected: 9 passing tests.

- [ ] **Step 5: Commit**

```bash
git add src/proxy/circuit_breaker.py tests/unit/test_circuit_breaker.py
git commit -m "feat(proxy): per-provider circuit breaker with closed/open/half-open transitions"
```

---

## Task 5: `OpenAIClient` adapter

Async OpenAI completion adapter implementing the `Provider` protocol. Uses the official `openai>=1.0` async client.

**Files:**
- Create: `src/proxy/openai_client.py`
- Test: `tests/unit/test_openai_client.py`
- Modify: `pyproject.toml` (promote `openai` from `stretch` to core deps in Task 16; for now we add the import-time fail-soft check)

- [ ] **Step 1: Add `openai` to dev dependencies for the test environment**

Run: `pip install "openai>=1.14"`
Expected: install succeeds. (We will promote this to `pyproject.toml` core deps in Task 16 along with `anthropic`.)

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_openai_client.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy import openai_client
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError


@pytest.fixture
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> MagicMock:
    ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(openai_client, "AsyncOpenAI", ctor)
    return ctor


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderAPIError, match="not configured"):
        openai_client.OpenAIClient()


def test_openai_client_satisfies_provider_protocol(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    _install_mock_client(monkeypatch, MagicMock())
    assert isinstance(openai_client.OpenAIClient(), Provider)


async def test_complete_returns_response_text(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    message = MagicMock()
    message.content = "hi back"
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=response)
    _install_mock_client(monkeypatch, fake_client)

    result = await openai_client.OpenAIClient().complete("hi")
    assert result == "hi back"


async def test_complete_maps_timeout_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    import openai as openai_sdk

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=openai_sdk.APITimeoutError(request=MagicMock())
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderTimeoutError):
        await openai_client.OpenAIClient().complete("hi")


async def test_complete_maps_api_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    import openai as openai_sdk

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=openai_sdk.APIError("boom", request=MagicMock(), body=None)
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderAPIError):
        await openai_client.OpenAIClient().complete("hi")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_openai_client.py -v`
Expected: FAIL — module not yet created.

- [ ] **Step 4: Implement `OpenAIClient`**

Create `src/proxy/openai_client.py`:

```python
"""OpenAI provider using the official ``openai`` async client."""

from typing import cast

import openai
from openai import AsyncOpenAI

from src.config import get_settings
from src.proxy.base import ProviderAPIError, ProviderTimeoutError

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = get_settings().openai_api_key
        if not api_key:
            raise ProviderAPIError("openai_api_key is not configured")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError("OpenAI request timed out") from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content
        return cast(str, content or "")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_openai_client.py -v`
Expected: 5 passing.

- [ ] **Step 6: Commit**

```bash
git add src/proxy/openai_client.py tests/unit/test_openai_client.py
git commit -m "feat(proxy): OpenAI provider adapter implementing the Provider protocol"
```

---

## Task 6: `AnthropicClient` adapter

Symmetric to Task 5 but for Anthropic.

**Files:**
- Create: `src/proxy/anthropic_client.py`
- Test: `tests/unit/test_anthropic_client.py`

- [ ] **Step 1: Add `anthropic` to the local environment**

Run: `pip install "anthropic>=0.25"`
Expected: install succeeds.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_anthropic_client.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy import anthropic_client
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError


@pytest.fixture
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> MagicMock:
    ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(anthropic_client, "AsyncAnthropic", ctor)
    return ctor


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderAPIError, match="not configured"):
        anthropic_client.AnthropicClient()


def test_anthropic_client_satisfies_provider_protocol(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    _install_mock_client(monkeypatch, MagicMock())
    assert isinstance(anthropic_client.AnthropicClient(), Provider)


async def test_complete_returns_response_text(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    block = MagicMock()
    block.text = "hi from claude"
    response = MagicMock(content=[block])
    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=response)
    _install_mock_client(monkeypatch, fake_client)

    result = await anthropic_client.AnthropicClient().complete("hi")
    assert result == "hi from claude"


async def test_complete_maps_timeout_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    import anthropic

    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=anthropic.APITimeoutError(request=MagicMock())
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderTimeoutError):
        await anthropic_client.AnthropicClient().complete("hi")


async def test_complete_maps_api_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    import anthropic

    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=anthropic.APIError("boom", request=MagicMock(), body=None)
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderAPIError):
        await anthropic_client.AnthropicClient().complete("hi")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_anthropic_client.py -v`
Expected: FAIL — module not yet created.

- [ ] **Step 4: Implement `AnthropicClient`**

Create `src/proxy/anthropic_client.py`:

```python
"""Anthropic provider using the official ``anthropic`` async client."""

from typing import cast

import anthropic
from anthropic import AsyncAnthropic

from src.config import get_settings
from src.proxy.base import ProviderAPIError, ProviderTimeoutError

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        api_key = get_settings().anthropic_api_key
        if not api_key:
            raise ProviderAPIError("anthropic_api_key is not configured")
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError("Anthropic request timed out") from exc
        except anthropic.APIError as exc:
            raise ProviderAPIError(f"Anthropic request failed: {exc}") from exc
        if not response.content:
            return ""
        first = response.content[0]
        return cast(str, getattr(first, "text", ""))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_anthropic_client.py -v`
Expected: 5 passing.

- [ ] **Step 6: Commit**

```bash
git add src/proxy/anthropic_client.py tests/unit/test_anthropic_client.py
git commit -m "feat(proxy): Anthropic provider adapter implementing the Provider protocol"
```

---

## Task 7: `ProviderRouter` — ordered chain with breakers

The router wraps an ordered list of `(provider, breaker)` pairs and implements the `Provider` Protocol itself so `query.py` can swap it in for `GeminiClient` with no other change.

**Files:**
- Create: `src/proxy/router.py`
- Test: `tests/unit/test_provider_router.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_provider_router.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
)
from src.proxy.router import ProviderRouter


def _provider(name: str, complete: AsyncMock) -> MagicMock:
    p = MagicMock(spec=Provider)
    p.name = name
    p.complete = complete
    return p


def _router(providers: list[MagicMock]) -> ProviderRouter:
    return ProviderRouter(
        providers=providers,
        failure_threshold=2,
        failure_window_seconds=30.0,
        recovery_seconds=10.0,
    )


def test_router_implements_provider_protocol() -> None:
    p = _provider("p", AsyncMock(return_value="ok"))
    router = _router([p])
    assert isinstance(router, Provider)
    assert router.name == "chain[p]"


async def test_first_provider_succeeds_no_fallback() -> None:
    p1 = _provider("a", AsyncMock(return_value="from-a"))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-a"
    p2.complete.assert_not_called()


async def test_falls_through_to_next_provider_on_timeout() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-b"


async def test_falls_through_on_api_error() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderAPIError("500")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    result = await _router([p1, p2]).complete("hi")
    assert result == "from-b"


async def test_chain_exhausted_raises() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(side_effect=ProviderAPIError("500")))
    with pytest.raises(ProviderChainExhaustedError):
        await _router([p1, p2]).complete("hi")


async def test_breaker_opens_after_threshold_and_skips_provider() -> None:
    p1 = _provider("a", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    p2 = _provider("b", AsyncMock(return_value="from-b"))
    router = _router([p1, p2])

    # Two failures trip the breaker on p1 (threshold=2).
    for _ in range(2):
        assert await router.complete("hi") == "from-b"
    assert p1.complete.await_count == 2

    # Third call should skip p1 entirely.
    assert await router.complete("hi") == "from-b"
    assert p1.complete.await_count == 2  # still 2 — no extra call


def test_empty_chain_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="empty provider chain"):
        _router([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_provider_router.py -v`
Expected: FAIL — module not created yet.

- [ ] **Step 3: Implement `ProviderRouter`**

Create `src/proxy/router.py`:

```python
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
                if last_attempted is None:
                    last_attempted = provider.name
                PROVIDER_FALLBACK.labels(
                    **{"from": provider.name, "to": "next", "reason": "timeout"}
                ).inc()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_provider_router.py -v`
Expected: 7 passing.

- [ ] **Step 5: Commit**

```bash
git add src/proxy/router.py tests/unit/test_provider_router.py
git commit -m "feat(proxy): ProviderRouter walks ordered chain with per-provider breakers"
```

---

## Task 8: `SingleFlight` — semantic in-flight deduplication

Per-instance, in-process. Wraps the miss path so a burst of paraphrases collapses to one upstream call. Brute-force cosine over in-flight entries is fine because `len(_inflight) ≤ concurrency` (usually <100).

**Files:**
- Create: `src/concurrency/__init__.py` (empty package marker)
- Create: `src/concurrency/single_flight.py`
- Test: `tests/unit/test_single_flight.py`

- [ ] **Step 1: Create the package marker**

Create `src/concurrency/__init__.py` (one-line module docstring only):

```python
"""Concurrency primitives used by the API layer."""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_single_flight.py`:

```python
import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray

from src.concurrency.single_flight import SingleFlight


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


async def test_single_caller_runs_fn_once() -> None:
    sf = SingleFlight()
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await sf.execute(_unit([1.0, 0.0, 0.0]), threshold=0.8, fn=fn)
    assert result == "ok"
    assert calls == 1


async def test_concurrent_identical_embeddings_collapse_to_one_call() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return "shared"

    vec = _unit([1.0, 0.0, 0.0])

    async def caller() -> str:
        return await sf.execute(vec, threshold=0.8, fn=fn)

    tasks = [asyncio.create_task(caller()) for _ in range(50)]
    await asyncio.sleep(0.01)  # let all 50 reach the lock
    gate.set()
    results = await asyncio.gather(*tasks)

    assert results == ["shared"] * 50
    assert calls == 1


async def test_similar_embeddings_coalesce_when_above_threshold() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return "shared"

    base = _unit([1.0, 0.0, 0.0])
    nearby = _unit([1.0, 0.05, 0.0])  # cosine ≈ 0.998, well above 0.8

    t_base = asyncio.create_task(sf.execute(base, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    t_near = asyncio.create_task(sf.execute(nearby, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    gate.set()
    r_base, r_near = await asyncio.gather(t_base, t_near)

    assert r_base == r_near == "shared"
    assert calls == 1


async def test_dissimilar_embeddings_each_run_fn() -> None:
    sf = SingleFlight()
    calls = 0
    gate = asyncio.Event()

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await gate.wait()
        return f"call-{calls}"

    vec_a = _unit([1.0, 0.0, 0.0])
    vec_b = _unit([0.0, 1.0, 0.0])  # cosine = 0, below threshold

    t1 = asyncio.create_task(sf.execute(vec_a, threshold=0.8, fn=fn))
    t2 = asyncio.create_task(sf.execute(vec_b, threshold=0.8, fn=fn))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(t1, t2)

    assert calls == 2


async def test_exception_propagates_to_all_waiters() -> None:
    sf = SingleFlight()
    gate = asyncio.Event()

    async def fn() -> str:
        await gate.wait()
        raise RuntimeError("boom")

    vec = _unit([1.0, 0.0, 0.0])
    tasks = [asyncio.create_task(sf.execute(vec, threshold=0.8, fn=fn)) for _ in range(5)]
    await asyncio.sleep(0.01)
    gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(results) == 5
    for r in results:
        assert isinstance(r, RuntimeError)
        assert str(r) == "boom"


async def test_late_arrival_after_completion_runs_again() -> None:
    sf = SingleFlight()
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    vec = _unit([1.0, 0.0, 0.0])
    await sf.execute(vec, threshold=0.8, fn=fn)
    await sf.execute(vec, threshold=0.8, fn=fn)
    assert calls == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_single_flight.py -v`
Expected: FAIL — module not created.

- [ ] **Step 4: Implement `SingleFlight`**

Create `src/concurrency/single_flight.py`:

```python
"""Semantic-keyed single-flight: coalesce in-flight upstream calls.

When two paraphrases of the same prompt miss the cache at the same time,
naive code makes two upstream calls and writes the cache twice. ``SingleFlight``
keeps a small list of in-flight requests keyed by their embedding; subsequent
callers with a similar embedding await the same ``asyncio.Future``.

Brute-force cosine is acceptable: ``_inflight`` is bounded by request
concurrency, typically <100 entries, dimension 384. HNSW would be overkill.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.metrics import SINGLEFLIGHT_COALESCED


@dataclass
class _InFlight:
    embedding: NDArray[np.float32]
    future: asyncio.Future[str]
    started_at: float


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: list[_InFlight] = []
        self._lock = asyncio.Lock()

    async def execute(
        self,
        embedding: NDArray[np.float32],
        threshold: float,
        fn: Callable[[], Awaitable[str]],
    ) -> str:
        async with self._lock:
            match = self._best_match(embedding, threshold)
            if match is not None:
                SINGLEFLIGHT_COALESCED.inc()
                return await match.future
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            entry = _InFlight(embedding=embedding, future=future, started_at=time.monotonic())
            self._inflight.append(entry)

        try:
            result = await fn()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result
        finally:
            async with self._lock:
                try:
                    self._inflight.remove(entry)
                except ValueError:
                    pass

    def _best_match(
        self, embedding: NDArray[np.float32], threshold: float
    ) -> _InFlight | None:
        best: _InFlight | None = None
        best_sim = threshold
        for entry in self._inflight:
            sim = float(np.dot(embedding, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best = entry
        return best
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_single_flight.py -v`
Expected: 6 passing.

- [ ] **Step 6: Commit**

```bash
git add src/concurrency/__init__.py src/concurrency/single_flight.py tests/unit/test_single_flight.py
git commit -m "feat(concurrency): semantic single-flight for in-flight miss coalescing"
```

---

## Task 9: TTL in `_CacheState` and `_apply_put` signature

Adds `_expires_at: dict[int, float]` to `_CacheState`. Leader stamps `op_time` on each `_apply_put` so all replicas deterministically compute the same `expires_at`. Read-path treats expired entries as Miss without mutating state — only the leader emits deletes.

**Files:**
- Modify: `src/raft/state_machine.py`
- Test: `tests/unit/test_raft_state_machine.py` (additions only)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_raft_state_machine.py`:

```python
def test_apply_put_stores_expires_at(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=60.0)
    assert state.expires_at(0) == 160.0


def test_apply_put_with_zero_ttl_means_never_expires(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=0.0)
    assert state.expires_at(0) is None


def test_lookup_returns_none_for_expired_entry(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    assert state.lookup(vec, threshold=0.5, now=109.0) is not None
    assert state.lookup(vec, threshold=0.5, now=111.0) is None


def test_lookup_without_now_uses_no_expiry(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    # Backwards-compatible call: no now, no TTL check.
    assert state.lookup(vec, threshold=0.5) is not None


def test_find_expired_returns_ids_due_for_eviction(state: _CacheState) -> None:
    state.apply_put("a", _unit([1.0, 0.0, 0.0, 0.0]), op_time=100.0, ttl_seconds=10.0)
    state.apply_put("b", _unit([0.0, 1.0, 0.0, 0.0]), op_time=100.0, ttl_seconds=100.0)
    expired = state.find_expired(now=120.0)
    assert expired == [0]


def test_apply_delete_clears_expires_at(state: _CacheState) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    id_ = state.apply_put("v", vec, op_time=100.0, ttl_seconds=10.0)
    state.apply_delete(id_)
    assert state.expires_at(id_) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_raft_state_machine.py -v`
Expected: FAIL — `apply_put` doesn't yet accept `op_time` / `ttl_seconds`; `expires_at`, `find_expired`, and `lookup(now=...)` don't exist.

- [ ] **Step 3: Update `_CacheState`**

In `src/raft/state_machine.py`, replace the `_CacheState` class with:

```python
class _CacheState:
    """In-memory state machine: embedding index + id→value + id→embedding + id→expires_at."""

    def __init__(self, index: SemanticIndex | None = None) -> None:
        self._index = index if index is not None else SemanticIndex()
        self._values: dict[int, str] = {}
        self._embeddings: dict[int, NDArray[np.float32]] = {}
        self._expires_at: dict[int, float] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.size

    @property
    def soft_deleted_count(self) -> int:
        return self._index.soft_deleted_count

    @property
    def total_count(self) -> int:
        return self._index.total_count

    def apply_put(
        self,
        value: str,
        embedding: NDArray[np.float32],
        *,
        op_time: float = 0.0,
        ttl_seconds: float = 0.0,
    ) -> int:
        assigned_id = self._next_id
        self._next_id += 1
        self._index.add(embedding, id_=assigned_id)
        self._values[assigned_id] = value
        self._embeddings[assigned_id] = embedding
        if ttl_seconds > 0.0:
            self._expires_at[assigned_id] = op_time + ttl_seconds
        return assigned_id

    def apply_delete(self, id_: int) -> bool:
        if id_ not in self._values:
            return False
        self._index.mark_deleted(id_)
        del self._values[id_]
        del self._embeddings[id_]
        self._expires_at.pop(id_, None)
        return True

    def expires_at(self, id_: int) -> float | None:
        return self._expires_at.get(id_)

    def find_expired(self, now: float) -> list[int]:
        return [id_ for id_, exp in self._expires_at.items() if exp <= now]

    def should_rebuild(self, threshold: float) -> bool:
        total = self._index.total_count
        if total == 0:
            return False
        return self._index.soft_deleted_count / total > threshold

    def rebuild_index(self) -> None:
        self._index.rebuild(self._embeddings.items())

    def lookup(
        self,
        embedding: NDArray[np.float32],
        threshold: float,
        *,
        now: float | None = None,
    ) -> Hit | None:
        found = self.lookup_with_id(embedding, threshold, now=now)
        return None if found is None else found[1]

    def lookup_with_id(
        self,
        embedding: NDArray[np.float32],
        threshold: float,
        *,
        now: float | None = None,
    ) -> tuple[int, Hit] | None:
        matches = self._index.search(embedding, k=1, threshold=threshold)
        if not matches:
            return None
        top = matches[0]
        value = self._values.get(top.id)
        if value is None:
            return None
        if now is not None:
            exp = self._expires_at.get(top.id)
            if exp is not None and exp <= now:
                return None
        return top.id, Hit(value=value, similarity=top.similarity)
```

- [ ] **Step 4: Update `ReplicatedSemanticCache` to stamp `op_time` and pass `ttl_seconds`**

In the same file, replace `_apply_put`, `put`, and `get_or_miss` with:

```python
    @replicated  # type: ignore[untyped-decorator]
    def _apply_put(
        self, value: str, embedding_bytes: bytes, op_time: float, ttl_seconds: float
    ) -> int:
        vector = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
        assigned_id = self._state.apply_put(
            value, vector, op_time=op_time, ttl_seconds=ttl_seconds
        )
        _publish_cache_gauges(self._state)
        return assigned_id

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        settings = get_settings()
        op_time = time.time()
        ttl_seconds = settings.cache_ttl_seconds
        if self.is_leader() and self._state.size >= settings.max_capacity:
            victim = next(iter(self._lru), None)
            if victim is not None:
                self._apply_delete(victim, sync=True)
                self._lru.pop(victim, None)
        # Opportunistic TTL eviction: leader scans for one expired id and
        # piggybacks a delete onto this write. Bounded work per put.
        if self.is_leader() and ttl_seconds > 0.0:
            expired = self._state.find_expired(now=op_time)
            if expired:
                self._apply_delete(expired[0], sync=True)
                self._lru.pop(expired[0], None)
                CACHE_TTL_EVICTIONS.inc()
        assigned_id = int(
            self._apply_put(value, vector.tobytes(), op_time, ttl_seconds, sync=True)
        )
        if self.is_leader():
            self._touch_lru(assigned_id)
        return assigned_id

    def get_or_miss(self, prompt: str) -> Hit | Miss:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        now = time.time()
        found = self._state.lookup_with_id(vector, threshold, now=now)
        if found is not None:
            hit_id, hit = found
            if self.is_leader():
                self._touch_lru(hit_id)
            return hit
        return Miss(prompt=prompt, embedding=vector)
```

Add `import time` and `from src.metrics import CACHE_TTL_EVICTIONS` at the top of the file (extend the existing metrics-import block).

- [ ] **Step 5: Run the targeted tests**

Run: `pytest tests/unit/test_raft_state_machine.py -v`
Expected: all green, including the 6 new TTL tests.

- [ ] **Step 6: Run the full unit suite**

Run: `pytest tests -m "not integration" -q`
Expected: all green. `test_lru.py` and existing tests pass unchanged because old `lookup` calls don't pass `now` and old `apply_put` calls work via the default keyword args.

- [ ] **Step 7: Commit**

```bash
git add src/raft/state_machine.py tests/unit/test_raft_state_machine.py
git commit -m "feat(raft): TTL via leader-stamped op_time on _apply_put"
```

---

## Task 10: TTL in `SemanticCache` (non-Raft path)

Mirror Task 9 on the single-node `SemanticCache` so non-Raft mode behaves identically.

**Files:**
- Modify: `src/cache/core.py`
- Test: `tests/unit/test_core.py` (additions)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_core.py`:

```python
def test_ttl_expired_entry_returns_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0.5")

    cache = SemanticCache()
    cache.put("prompt", "value")
    # First read is a Hit immediately after put.
    first = cache.get_or_miss("prompt")
    assert isinstance(first, Hit)

    time.sleep(0.7)
    second = cache.get_or_miss("prompt")
    assert isinstance(second, Miss)


def test_ttl_zero_disables_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")

    cache = SemanticCache()
    cache.put("prompt", "value")
    time.sleep(0.2)
    assert isinstance(cache.get_or_miss("prompt"), Hit)
```

If `Hit`/`Miss`/`SemanticCache`/`time` aren't imported at the top of the file, add the imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_core.py -k ttl -v`
Expected: FAIL — second read still hits because no TTL.

- [ ] **Step 3: Implement TTL in `SemanticCache`**

In `src/cache/core.py`:

- Add `import time` at the top.
- In `SemanticCache.__init__`, add `self._expires_at: dict[int, float] = {}` next to the other dicts.
- Replace `get_or_miss` and `put` with the versions below; also update `_evict`:

```python
    def get_or_miss(self, prompt: str) -> CacheResult:
        vector = embed(prompt)
        threshold = get_settings().similarity_threshold
        now = time.time()
        with self._lock:
            matches = self._index.search(vector, k=1, threshold=threshold)
            if matches:
                top = matches[0]
                exp = self._expires_at.get(top.id)
                if exp is None or exp > now:
                    self._touch(top.id)
                    return Hit(value=self._values[top.id], similarity=top.similarity)
        return Miss(prompt=prompt, embedding=vector)

    def put(
        self,
        prompt: str,
        value: str,
        embedding: NDArray[np.float32] | None = None,
    ) -> int:
        vector = embedding if embedding is not None else embed(prompt)
        settings = get_settings()
        now = time.time()
        with self._lock:
            if self._index.size >= settings.max_capacity:
                victim = next(iter(self._lru), None)
                if victim is not None:
                    self._evict(victim)
            if settings.cache_ttl_seconds > 0.0:
                for expired_id in [i for i, e in self._expires_at.items() if e <= now]:
                    self._evict(expired_id)
                    CACHE_TTL_EVICTIONS.inc()
            assigned_id = self._next_id
            self._next_id += 1
            self._index.add(vector, id_=assigned_id)
            self._values[assigned_id] = value
            self._embeddings[assigned_id] = vector
            if settings.cache_ttl_seconds > 0.0:
                self._expires_at[assigned_id] = now + settings.cache_ttl_seconds
            self._touch(assigned_id)
            self._publish_gauges()
            return assigned_id

    def _evict(self, id_: int) -> None:
        self._index.mark_deleted(id_)
        self._values.pop(id_, None)
        self._embeddings.pop(id_, None)
        self._expires_at.pop(id_, None)
        self._lru.pop(id_, None)
        CACHE_EVICTIONS.inc()
        self._maybe_rebuild()
        self._publish_gauges()
```

Update the metrics import line at the top of `src/cache/core.py` to include `CACHE_TTL_EVICTIONS`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_core.py -v`
Expected: all green, including the 2 new TTL tests.

- [ ] **Step 5: Run full unit suite**

Run: `pytest tests -m "not integration" -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/cache/core.py tests/unit/test_core.py
git commit -m "feat(cache): TTL eviction on top of LRU in SemanticCache"
```

---

## Task 11: Wire `ProviderRouter` into `query.py`

Replace `_get_gemini()` with `_get_router()` that builds the chain from `Settings.provider_chain`. Drop providers whose API key is missing. Empty chain → startup error. Returns 503 (with `Retry-After`) when the chain is exhausted.

**Files:**
- Modify: `src/api/query.py`
- Test: `tests/unit/test_query.py`

- [ ] **Step 1: Update the failing test fixtures and add new tests**

Replace `tests/unit/test_query.py` with the version below (it keeps the existing tests and adds chain-exhausted handling). Note the fixture rename `fake_provider` → `fake_router`:

```python
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import HTTPException
from numpy.typing import NDArray

from src.api import query as query_module
from src.api.query import QueryRequest, query
from src.cache.core import Hit, Miss
from src.proxy.base import (
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
)


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    cache = MagicMock()
    monkeypatch.setattr(query_module, "_get_cache", lambda: cache)
    return cache


@pytest.fixture
def fake_router(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    router = MagicMock()
    router.complete = AsyncMock()
    router.name = "chain[gemini]"
    monkeypatch.setattr(query_module, "_get_router", lambda: router)
    return router


@pytest.fixture(autouse=True)
def reset_singleflight(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.concurrency.single_flight import SingleFlight

    monkeypatch.setattr(query_module, "_get_single_flight", lambda: SingleFlight())


async def test_hit_returns_cached_response_without_calling_provider(
    fake_cache: MagicMock, fake_router: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Hit(value="cached answer", similarity=0.92)

    result = await query(QueryRequest(prompt="hi"))

    assert result.response == "cached answer"
    assert result.cached is True
    assert result.similarity == pytest.approx(0.92)
    fake_router.complete.assert_not_called()
    fake_cache.put.assert_not_called()


async def test_miss_calls_router_and_caches_response(
    fake_cache: MagicMock, fake_router: MagicMock
) -> None:
    miss_embedding = _unit([1.0, 0.0, 0.0, 0.0])
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=miss_embedding)
    fake_router.complete.return_value = "fresh answer"

    result = await query(QueryRequest(prompt="hi"))

    assert result.response == "fresh answer"
    assert result.cached is False
    assert result.similarity is None
    fake_router.complete.assert_awaited_once_with("hi")
    fake_cache.put.assert_called_once_with("hi", "fresh answer", embedding=miss_embedding)


async def test_provider_timeout_maps_to_502(
    fake_cache: MagicMock, fake_router: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    fake_router.complete.side_effect = ProviderTimeoutError("slow")

    with pytest.raises(HTTPException) as exc_info:
        await query(QueryRequest(prompt="hi"))

    assert exc_info.value.status_code == 502
    fake_cache.put.assert_not_called()


async def test_provider_api_error_maps_to_502(
    fake_cache: MagicMock, fake_router: MagicMock
) -> None:
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    fake_router.complete.side_effect = ProviderAPIError("500")

    with pytest.raises(HTTPException) as exc_info:
        await query(QueryRequest(prompt="hi"))

    assert exc_info.value.status_code == 502


async def test_chain_exhausted_maps_to_503_with_retry_after(
    fake_cache: MagicMock, fake_router: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BREAKER_RECOVERY_SECONDS", "12")
    fake_cache.get_or_miss.return_value = Miss(prompt="hi", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    fake_router.complete.side_effect = ProviderChainExhaustedError("all dead")

    with pytest.raises(HTTPException) as exc_info:
        await query(QueryRequest(prompt="hi"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "12"}


def test_query_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError):
        QueryRequest(prompt="")


def test_query_request_accepts_known_providers() -> None:
    QueryRequest(prompt="hi", provider="gemini")
    QueryRequest(prompt="hi", provider="openai")
    QueryRequest(prompt="hi", provider="anthropic")


def test_query_request_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        QueryRequest(prompt="hi", provider="cohere")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pytest tests/unit/test_query.py -v`
Expected: FAIL — `_get_router` and `_get_single_flight` don't exist; provider literal still only allows `gemini`.

- [ ] **Step 3: Rewrite `src/api/query.py`**

Replace the entire file with:

```python
"""POST /query — semantic cache in front of a multi-provider router."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.cache.core import Hit, SemanticCache
from src.concurrency.single_flight import SingleFlight
from src.config import get_settings
from src.metrics import QUERY_LATENCY, QUERY_TOTAL
from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderChainExhaustedError,
    ProviderTimeoutError,
    ProxyError,
)
from src.proxy.router import ProviderRouter
from src.raft.state_machine import ReplicatedSemanticCache

CacheBackend = SemanticCache | ReplicatedSemanticCache

router = APIRouter()


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1)
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"


class QueryResponse(BaseModel):
    response: str
    cached: bool
    similarity: float | None = None


@dataclass
class _State:
    cache: CacheBackend | None = field(default=None)
    router: Provider | None = field(default=None)
    single_flight: SingleFlight | None = field(default=None)


_state = _State()


def _build_cache() -> CacheBackend:
    settings = get_settings()
    if settings.raft_enabled:
        return ReplicatedSemanticCache(settings.raft_bind, settings.raft_peers)
    return SemanticCache()


def _get_cache() -> CacheBackend:
    if _state.cache is None:
        _state.cache = _build_cache()
    return _state.cache


def _build_router() -> Provider:
    settings = get_settings()
    providers: list[Provider] = []
    for name in settings.provider_chain:
        try:
            providers.append(_construct_provider(name))
        except ProxyError:
            # Missing API key for this provider — drop silently from the chain.
            continue
    if not providers:
        raise ProxyError(
            "provider_chain empty after dropping unconfigured providers; "
            "set at least one of GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
        )
    return ProviderRouter(
        providers=providers,
        failure_threshold=settings.breaker_failure_threshold,
        failure_window_seconds=settings.breaker_failure_window_seconds,
        recovery_seconds=settings.breaker_recovery_seconds,
    )


def _construct_provider(name: str) -> Provider:
    if name == "gemini":
        from src.proxy.gemini_client import GeminiClient

        return GeminiClient()
    if name == "openai":
        from src.proxy.openai_client import OpenAIClient

        return OpenAIClient()
    if name == "anthropic":
        from src.proxy.anthropic_client import AnthropicClient

        return AnthropicClient()
    raise ProxyError(f"unknown provider: {name}")


def _get_router() -> Provider:
    if _state.router is None:
        _state.router = _build_router()
    return _state.router


def _get_single_flight() -> SingleFlight:
    if _state.single_flight is None:
        _state.single_flight = SingleFlight()
    return _state.single_flight


def _record(outcome: str, start: float) -> None:
    QUERY_TOTAL.labels(outcome=outcome).inc()
    QUERY_LATENCY.labels(outcome=outcome).observe(time.perf_counter() - start)


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    cache = _get_cache()
    provider = _get_router()
    sf = _get_single_flight()
    settings = get_settings()

    result = await asyncio.to_thread(cache.get_or_miss, body.prompt)
    if isinstance(result, Hit):
        _record("hit", started)
        return QueryResponse(
            response=result.value,
            cached=True,
            similarity=result.similarity,
        )

    async def upstream_and_store() -> str:
        text = await provider.complete(body.prompt)
        await asyncio.to_thread(cache.put, body.prompt, text, embedding=result.embedding)
        return text

    try:
        response_text = await sf.execute(
            result.embedding,
            threshold=settings.coalesce_threshold,
            fn=upstream_and_store,
        )
    except ProviderTimeoutError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProviderChainExhaustedError as exc:
        _record("provider_error", started)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(int(settings.breaker_recovery_seconds))},
        ) from exc
    except ProviderAPIError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProxyError as exc:
        _record("provider_error", started)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _record("miss", started)
    return QueryResponse(response=response_text, cached=False, similarity=None)
```

- [ ] **Step 4: Update lifespan to eagerly build the router too**

Edit `src/api/__init__.py`. Replace `_lifespan` with:

```python
@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Eagerly construct the Raft-backed cache + router at startup so:
    #   1. all replicas bind their Raft ports before the first /query arrives
    #      (otherwise the first write blocks forever waiting for quorum); and
    #   2. an empty/misconfigured provider chain fails loudly at boot.
    if get_settings().raft_enabled:
        _get_cache()
    from src.api.query import _get_router

    _get_router()
    yield
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_query.py -v`
Expected: all green.

- [ ] **Step 6: Run full unit suite**

Run: `pytest tests -m "not integration" -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/api/query.py src/api/__init__.py tests/unit/test_query.py
git commit -m "feat(api): route /query through ProviderRouter and SingleFlight"
```

---

## Task 12: Expand bench dataset to 200 prompts

Spec target: 40 topics × 5 paraphrases. The current file is 20 × 3 = 60. We extend with new security topics and paraphrases so the bench numbers are statistically meaningful.

**Files:**
- Modify: `benchmarks/dataset.json`

- [ ] **Step 1: Inspect current dataset**

Run: `python -c "import json; d=json.load(open('benchmarks/dataset.json')); print('count:', len(d), 'unique:', len(set(d)))"`
Expected: prints the current count (≈60) and unique count.

- [ ] **Step 2: Rewrite `benchmarks/dataset.json`**

Replace the file with a JSON array of exactly 200 prompts: 40 distinct security topics × 5 paraphrases each. Reuse the existing topics where possible and add 20 new ones (e.g., SSRF, prototype-pollution, race conditions in auth, JWT confusion, IDOR, RCE via template injection, ReDoS, mass-assignment, broken-access-control, supply-chain typosquats, container escape, eBPF risks, Kubernetes RBAC misuse, Spring RCE, log4shell-class issues, ImageMagick exploits, OAuth redirect flaws, SAML XML signature wrapping, S3 bucket misconfig, MFA bypass).

Format (one block per topic, blank lines preserved for readability):

```json
[
  "What is Cross-Site Scripting and how do I prevent it?",
  "Explain XSS and common defenses.",
  "How does stored XSS differ from reflected XSS?",
  "Describe the dangers of DOM-based XSS in a single-page app.",
  "How do Content Security Policy headers mitigate cross-site scripting?",

  ...
]
```

Topics × paraphrases = 40 × 5 = 200.

- [ ] **Step 3: Verify count is exactly 200**

Run: `python -c "import json; d=json.load(open('benchmarks/dataset.json')); print(len(d)); assert len(d)==200"`
Expected: prints `200` and exits 0.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/dataset.json
git commit -m "chore(bench): expand dataset to 200 prompts (40 topics x 5 paraphrases)"
```

---

## Task 13: `bench_compare.py` — no-cache vs Redis-exact vs kvraft-semantic

Strategy-comparison harness. Outputs a CSV row per (strategy, request) and a PNG bar/CDF chart. Redis import is lazy: if the dep is missing, that strategy is skipped with a warning.

**Files:**
- Create: `scripts/bench_compare.py`

- [ ] **Step 1: Install bench-time deps locally**

Run: `pip install "redis>=5.0" "matplotlib>=3.8" "aiohttp>=3.9"`
Expected: install succeeds.

- [ ] **Step 2: Implement `scripts/bench_compare.py`**

Create `scripts/bench_compare.py`:

```python
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

HTTP_OK = 200


@dataclass
class Result:
    strategy: str
    idx: int
    prompt: str
    cached: bool
    latency_ms: float
    status: int


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    return (
        statistics.median(s),
        s[int(0.95 * (len(s) - 1))],
        s[int(0.99 * (len(s) - 1))],
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
            json={"prompt": f"NOCACHE-{idx}-{prompt}"},  # poison the key so cache always misses
        ) as resp:
            status = resp.status
            await resp.json() if status == HTTP_OK else None
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
            json={"prompt": f"REDIS-{idx}-{prompt}"},  # bypass kvraft semantic cache
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
            tasks = [
                _no_cache(session, kvraft_host, i, p, sem) for i, p in enumerate(prompts)
            ]
        elif strategy == "redis-exact":
            try:
                import redis.asyncio as aioredis  # type: ignore[import-untyped]
            except ImportError:
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
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
    args = parser.parse_args()

    raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    base = [p for p in raw if isinstance(p, str) and p.strip()]
    rng = random.Random(args.seed)
    prompts = [rng.choice(base) for _ in range(args.requests)]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    per_strategy: dict[str, list[Result]] = {}
    for strategy in strategies:
        print(f"[run] strategy={strategy} requests={args.requests} c={args.concurrency}")
        per_strategy[strategy] = asyncio.run(
            _run_strategy(
                strategy, prompts, args.concurrency, args.kvraft_host, args.redis_url
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strategy", "idx", "prompt", "cached", "latency_ms", "status"])
        for strategy, results in per_strategy.items():
            for r in results:
                writer.writerow([r.strategy, r.idx, r.prompt, r.cached, f"{r.latency_ms:.3f}", r.status])
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
```

- [ ] **Step 3: Run a syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/bench_compare.py').read())"`
Expected: no output (clean parse).

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_compare.py
git commit -m "feat(bench): bench_compare.py for no-cache vs Redis-exact vs kvraft-semantic"
```

---

## Task 14: Integration tests — failover-chain, thundering-herd, TTL

These run only under `pytest -m integration` and use mocked providers; no real API keys needed.

**Files:**
- Create: `tests/integration/test_failover_chain.py`
- Create: `tests/integration/test_thundering_herd.py`
- Create: `tests/integration/test_ttl_eviction.py`

- [ ] **Step 1: Write the failover-chain test**

Create `tests/integration/test_failover_chain.py`:

```python
"""Force the first provider into 5x failures inside the window and verify
the router falls through to the next provider on the very next call."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.proxy.base import Provider, ProviderTimeoutError
from src.proxy.router import ProviderRouter

pytestmark = pytest.mark.integration


def _provider(name: str, complete: AsyncMock) -> MagicMock:
    p = MagicMock(spec=Provider)
    p.name = name
    p.complete = complete
    return p


async def test_breaker_trips_after_5_timeouts_and_routes_to_next() -> None:
    flaky = _provider("gemini", AsyncMock(side_effect=ProviderTimeoutError("slow")))
    healthy = _provider("openai", AsyncMock(return_value="ok"))
    router = ProviderRouter(
        providers=[flaky, healthy],
        failure_threshold=5,
        failure_window_seconds=30.0,
        recovery_seconds=15.0,
    )

    # Five attempts: each falls through to healthy, but each also records a failure on flaky.
    for _ in range(5):
        assert await router.complete("prompt") == "ok"

    flaky.complete.reset_mock()
    # Sixth attempt should skip flaky entirely.
    assert await router.complete("prompt") == "ok"
    flaky.complete.assert_not_called()
```

- [ ] **Step 2: Write the thundering-herd test**

Create `tests/integration/test_thundering_herd.py`:

```python
"""50 concurrent paraphrases of one prompt collapse to one upstream call."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.embedding import embed
from src.concurrency.single_flight import SingleFlight

pytestmark = pytest.mark.integration


@pytest.fixture
def paraphrases() -> list[str]:
    return [
        "What is SQL injection?",
        "Explain SQLi and how parameterized queries help.",
        "How does a blind SQL injection attack work?",
        "Describe SQL injection in plain terms.",
        "Why is SQLi still common today?",
    ]


async def test_concurrent_paraphrases_coalesce_to_one_upstream_call(
    paraphrases: list[str],
) -> None:
    sf = SingleFlight()
    upstream_calls = 0
    gate = asyncio.Event()

    async def upstream() -> str:
        nonlocal upstream_calls
        upstream_calls += 1
        await gate.wait()
        return "shared"

    async def call(prompt: str) -> str:
        embedding: NDArray[np.float32] = embed(prompt)
        return await sf.execute(embedding, threshold=0.8, fn=upstream)

    # Repeat the 5 paraphrases 10 times → 50 concurrent calls.
    prompts = paraphrases * 10
    tasks = [asyncio.create_task(call(p)) for p in prompts]
    await asyncio.sleep(0.05)
    gate.set()
    results = await asyncio.gather(*tasks)

    assert all(r == "shared" for r in results)
    # Tight upper bound: most paraphrases collapse, but the very first concurrent
    # window may issue 2-3 distinct calls before _inflight catches them. Spec
    # target is "≥90% reduction" — 5 or fewer is well inside that for 50 calls.
    assert upstream_calls <= 5
```

- [ ] **Step 3: Write the TTL-eviction test**

Create `tests/integration/test_ttl_eviction.py`:

```python
"""TTL=1s entry must be a Miss after 1.1s on every replica's lookup."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.index import SemanticIndex
from src.raft.state_machine import _CacheState

pytestmark = pytest.mark.integration


def _unit(values: list[float]) -> NDArray[np.float32]:
    arr = np.array(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_ttl_expired_entry_is_miss_on_all_replicas() -> None:
    leader = _CacheState(index=SemanticIndex(dim=4, initial_capacity=8))
    follower = _CacheState(index=SemanticIndex(dim=4, initial_capacity=8))

    op_time = time.time()
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    leader.apply_put("answer", vec, op_time=op_time, ttl_seconds=1.0)
    follower.apply_put("answer", vec, op_time=op_time, ttl_seconds=1.0)

    # Both replicas see the entry within the TTL window.
    assert leader.lookup(vec, threshold=0.5, now=op_time + 0.5) is not None
    assert follower.lookup(vec, threshold=0.5, now=op_time + 0.5) is not None

    # Both replicas treat it as a Miss after the TTL window.
    assert leader.lookup(vec, threshold=0.5, now=op_time + 1.1) is None
    assert follower.lookup(vec, threshold=0.5, now=op_time + 1.1) is None

    # find_expired returns the id on the leader so it can emit a Raft delete.
    assert leader.find_expired(now=op_time + 1.1) == [0]
```

- [ ] **Step 4: Run the integration tests**

Run: `pytest tests/integration/test_failover_chain.py tests/integration/test_thundering_herd.py tests/integration/test_ttl_eviction.py -v -m integration`
Expected: 3 tests pass.

- [ ] **Step 5: Confirm `-m "not integration"` still excludes them**

Run: `pytest tests -m "not integration" -q`
Expected: integration tests collected and deselected; all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_failover_chain.py tests/integration/test_thundering_herd.py tests/integration/test_ttl_eviction.py
git commit -m "test(integration): failover-chain, thundering-herd, TTL eviction"
```

---

## Task 15: Update `pyproject.toml` deps and run lint/type checks

Promote `openai` and `anthropic` from `stretch` to core; add `redis` and `matplotlib` to `bench`.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `dependencies` and `[project.optional-dependencies]` blocks with:

```toml
dependencies = [
    # Web framework + async
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
    # Embedding model
    "sentence-transformers>=2.7",
    "torch>=2.2",
    "numpy>=1.26",
    # ANN index
    "hnswlib>=0.8",
    # Raft consensus
    "pysyncobj>=0.3.12",
    # LLM providers
    "google-generativeai>=0.5",
    "openai>=1.14",
    "anthropic>=0.25",
    # Config + validation
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    # Observability
    "prometheus-client>=0.20",
    "structlog>=24.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "ruff>=0.4",
    "black>=24.3",
    "mypy>=1.9",
    "pre-commit>=3.7",
]
bench = [
    "aiohttp>=3.9",
    "matplotlib>=3.8",
    "redis>=5.0",
]
```

(Delete the `stretch` block; its members are now in core.)

Also add mypy override blocks for the new SDKs at the bottom of the `[tool.mypy]` section:

```toml
[[tool.mypy.overrides]]
module = ["openai", "openai.*", "anthropic", "anthropic.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Run ruff, black, mypy**

Run: `ruff check src tests && black --check src tests && mypy`
Expected: all green. Fix any small style issues (import order, line length) that surface.

- [ ] **Step 3: Run the entire unit suite once more**

Run: `pytest tests -m "not integration" -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: promote OpenAI + Anthropic to core deps; Redis under bench extras"
```

---

## Task 16: Update README, DECISIONS.md, and resume bullet

Land the docs that turn the implementation into a defensible resume bullet.

**Files:**
- Modify: `README.md`
- Modify: `DECISIONS.md`
- Modify: `E:/Stuff/Resume/RESUME.tex` (Windows path) / `/mnt/e/Stuff/Resume/RESUME.tex` (WSL path)

- [ ] **Step 1: README — extend the endpoint metrics list**

In `README.md`, find the `/metrics` row in the endpoint table and replace it with:

```markdown
| `GET /metrics` | Prometheus exposition: `kvraft_query_total`, `kvraft_query_latency_seconds`, `kvraft_provider_calls_total`, `kvraft_leader_state`, `kvraft_cache_evictions_total`, `kvraft_cache_live_entries`, `kvraft_cache_soft_deleted_entries`, `kvraft_cache_rebuilds_total`, `kvraft_provider_circuit_state`, `kvraft_provider_fallback_total`, `kvraft_provider_chain_exhausted_total`, `kvraft_singleflight_coalesced_total`, `kvraft_cache_ttl_evictions_total` |
```

- [ ] **Step 2: README — add a "Production proxy features" section**

After the existing "Architecture" section, insert:

```markdown
## Production proxy features

| Feature | What it does |
|---|---|
| **Multi-provider failover** | Ordered chain `gemini → openai → anthropic`. The first healthy provider serves the request. Configure via `PROVIDER_CHAIN`. |
| **Circuit breakers** | Per-provider, in-process state machine (`closed → open → half-open → closed`). Trips after 5 failures in 30s; probes the provider every 15s after that. |
| **Semantic single-flight** | A burst of paraphrased prompts on a cold cache collapses to one upstream call. Coalescing key is embedding-cosine, not a string hash. |
| **TTL on top of LRU** | Entries expire after `CACHE_TTL_SECONDS` (default 3600s). The leader stamps `op_time` on every Raft write so all replicas agree on `expires_at` without depending on follower clocks. |
| **Redis comparison bench** | `scripts/bench_compare.py` runs the same 200-prompt paraphrase workload through no-cache / Redis exact-match / kvraft semantic and writes a CSV + PNG to `benchmarks/results/`. |
```

- [ ] **Step 3: README — add a "How does kvraft compare to Redis?" section near the existing benchmarks section**

```markdown
## How does kvraft compare to Redis?

`scripts/bench_compare.py --strategies none,redis-exact,kvraft-semantic --requests 200 --concurrency 8`

| Strategy | Hit rate | P50 | P99 |
|---|---|---|---|
| no cache | 0% | upstream | upstream |
| Redis (exact) | ~20% | low | upstream-on-miss |
| **kvraft (semantic)** | **~80%** | **38 ms** | **58 ms** |

Redis catches exact repeats. kvraft catches paraphrases because lookups are over the prompt's embedding, not the prompt string.

![Hit rate and latency CDF](benchmarks/results/compare.png)
```

(The PNG path is the artifact `bench_compare.py` writes; commit the image once a real run produces it.)

- [ ] **Step 4: DECISIONS.md — append entry**

Append to `DECISIONS.md`:

```markdown
## 16. Production LLM proxy: multi-provider, breakers, single-flight, TTL

2026-05-15. Implemented design from `docs/superpowers/specs/2026-05-08-production-llm-proxy-design.md`.

* **Multi-provider chain** (`gemini → openai → anthropic`) via `src/proxy/router.py`. Providers without API keys are dropped at startup; empty chain fails loud.
* **Circuit breakers** are per-provider and per-replica, in-process. Not Raft-replicated: doing so would let one slow node trip the breaker for healthy nodes and violates the "only deterministic state belongs in the log" invariant. Same pattern as Hystrix / resilience4j.
* **Single-flight** is per-replica, keyed by embedding cosine similarity (not prompt hash). Catches paraphrase coalescing mid-flight before any candidate lands in cache.
* **TTL** is implemented via `op_time` stamped by the leader and included in the Raft `_apply_put` op. All replicas deterministically compute `expires_at = op_time + ttl_seconds`. `_apply_put` signature change invalidates pre-existing pysyncobj journals — required `rm /tmp/kvraft-local/*.journal` on first run.
* **Redis** is a bench-only dep (`[project.optional-dependencies] bench`). No production code path depends on Redis.
```

- [ ] **Step 5: Resume — update the kvraft bullet**

Edit `RESUME.tex`. Find the existing kvraft bullet (the one with "Built a 3-node FastAPI cache in front of Gemini using semantic search over prompt embeddings") and replace it with:

```latex
\item \textbf{kvraft --- Distributed Semantic Cache for LLM APIs} (Python, FastAPI, Raft, hnswlib, Redis, Prometheus): Built a 3-node FastAPI semantic cache with \textbf{multi-provider failover} (Gemini $\rightarrow$ OpenAI $\rightarrow$ Anthropic) using \textbf{circuit breakers} and \textbf{single-flight request coalescing} that cut upstream calls by \textbf{$>$90\%} under thundering-herd of paraphrased prompts. Replicated cache state through a \textbf{Raft log}; cluster survived \textbf{leader failure in 1.9\,s} with cached reads served throughout. \textbf{P50 38\,ms / P99 58\,ms at 198 RPS} vs.\ \textbf{6.6\,s upstream}; semantic cache hit \textbf{$\sim$80\% vs Redis exact-match's $\sim$20\%} on a 200-prompt paraphrase workload. \href{https://github.com/iamstufff/kvraft}{[GitHub]}
```

(Match the LaTeX style — `\textbf{}`, escaped percents, `\,` thin spaces — to whatever the surrounding bullets in the file use.)

- [ ] **Step 6: Commit**

```bash
git add README.md DECISIONS.md
git commit -m "docs: production proxy features, Redis comparison section, resume bullet"
```

(`RESUME.tex` lives outside the repo and is committed separately if its own git repo wants it; otherwise it's just edited in place.)

---

## Task 17: Run end-to-end benchmark and capture numbers

Spin up the cluster, run `bench_compare.py`, capture the artifacts, and confirm the resume-bullet numbers hold up.

**Files:**
- None modified — produces `benchmarks/results/compare.csv` and `benchmarks/results/compare.png`.

- [ ] **Step 1: Wipe stale journals**

Run: `rm -f /tmp/kvraft-local/*.journal /tmp/kvraft-local/*.snapshot`
Expected: no output. (Recall the rollout note — old journals are incompatible with the new `_apply_put` signature.)

- [ ] **Step 2: Start the 3-node local cluster**

Run: `bash scripts/run-local-cluster.sh`
Expected: prints PIDs for nodes 1/2/3 on ports 8001/8002/8003 and Raft on 4321/4322/4323. Logs land under `/tmp/kvraft-local/`.

- [ ] **Step 3: Confirm cluster health**

Run: `curl -s http://127.0.0.1:8001/health && echo && curl -s http://127.0.0.1:8002/health && echo && curl -s http://127.0.0.1:8003/health`
Expected: three `{"status":"ok",...}` lines.

- [ ] **Step 4: Start Redis (if not running)**

Run: `redis-server --daemonize yes`
Expected: starts on localhost:6379. If Redis isn't installed, `bench_compare.py` will skip that strategy with a warning — note that, and proceed.

- [ ] **Step 5: Warm the kvraft cache**

Run: `python scripts/bench.py --host http://127.0.0.1:8001 --dataset benchmarks/dataset.json --requests 200 --concurrency 1 --out benchmarks/results/warm.csv`
Expected: prints summary with 200 successful requests. (Concurrency 1 to stay inside the Gemini free-tier rate limit while seeding the cache.)

- [ ] **Step 6: Run the comparison bench**

Run: `python scripts/bench_compare.py --strategies none,redis-exact,kvraft-semantic --requests 200 --concurrency 8 --kvraft-host http://127.0.0.1:8001 --out benchmarks/results/compare.csv --plot benchmarks/results/compare.png`
Expected: three `[summary]` lines. kvraft-semantic hit rate should be ≥75%; Redis-exact hit rate should be in the 15–25% range; no-cache hit rate is 0. If any number is wildly off, debug before continuing — don't fudge the README.

- [ ] **Step 7: Inspect outputs**

Run: `ls -la benchmarks/results/compare.csv benchmarks/results/compare.png && head -5 benchmarks/results/compare.csv`
Expected: both files exist; CSV header is `strategy,idx,prompt,cached,latency_ms,status`.

- [ ] **Step 8: Replace placeholder numbers in README and resume bullet**

If the live numbers diverge from the spec preview (≥80% / ~20%), update the README's "How does kvraft compare to Redis?" table and the resume bullet to the actual figures. Don't ship a number you didn't measure.

- [ ] **Step 9: Commit the bench artifacts**

```bash
git add benchmarks/results/compare.csv benchmarks/results/compare.png README.md
git commit -m "bench: capture compare.csv + compare.png; reconcile README numbers"
```

(If `RESUME.tex` needed updating, commit it separately in its own location.)

- [ ] **Step 10: Stop the cluster**

Run: `bash scripts/kill-local-leader.sh && pkill -f "uvicorn src.api"`
Expected: leader killed and remaining workers exit. (Or use whatever shutdown script the repo currently has.)

---

## Task 18: Final verification + push

- [ ] **Step 1: Full test suite (unit + integration)**

Run: `pytest tests -q && pytest tests -m integration -q`
Expected: all green in both passes.

- [ ] **Step 2: Lint + type checks**

Run: `ruff check src tests && black --check src tests && mypy`
Expected: all green.

- [ ] **Step 3: Confirm STATUS.md and DECISIONS.md are updated**

Open `STATUS.md` and update:
- `Phase` → "Production LLM proxy enhancements landed"
- Move the in-progress entry to `Completed this session` with today's date
- `Next concrete action` → empty / "spec backlog: Bundle B persistence, Bundle C observability"

Confirm `DECISIONS.md` has entry #16 from Task 16 step 4.

- [ ] **Step 4: Push the branch**

Run: `git push -u origin feat/production-llm-proxy`
Expected: branch pushed.

- [ ] **Step 5: Open PR** (only if user requests it — by default stop here for review)

Don't open the PR automatically; surface the push URL so the user can review before merging.

---

## Spec → Plan coverage map

| Spec section | Task(s) |
|---|---|
| Goals: multi-provider failover | Tasks 5, 6, 7, 11 |
| Goals: request coalescing | Task 8, 11 |
| Goals: TTL on top of LRU | Tasks 9, 10 |
| Goals: Redis comparison baseline | Tasks 12, 13, 17 |
| Goals: 5 new Prometheus series | Task 3 (+ wired in 4, 7, 8, 9, 10) |
| Goals: updated resume bullet | Task 16 |
| New module: `circuit_breaker.py` | Task 4 |
| New module: `router.py` | Task 7 |
| New module: `openai_client.py` | Task 5 |
| New module: `anthropic_client.py` | Task 6 |
| New module: `concurrency/single_flight.py` | Task 8 |
| New module: `scripts/bench_compare.py` | Task 13 |
| Touched: `api/query.py` | Task 11 |
| Touched: `cache/core.py` | Tasks 0, 10 |
| Touched: `raft/state_machine.py` | Task 9 |
| Touched: `config.py` | Task 1 |
| Touched: `metrics/__init__.py` | Task 3 |
| Touched: `pyproject.toml` | Task 15 |
| Touched: `benchmarks/dataset.json` | Task 12 |
| Error: 503 + Retry-After | Task 11 |
| Error: drop unconfigured providers | Task 11 |
| Error: TTL clock skew note | Documented in plan rollout note + Task 16 |
| Risk: `_apply_put` signature break | Rollout note + Task 17 step 1 |
| Resume bullet preview | Task 16 |
| Decisions #1–5 from spec | Task 16 step 4 (single DECISIONS entry) |
