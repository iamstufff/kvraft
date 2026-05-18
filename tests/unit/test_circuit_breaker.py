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
    bk.allow(now=11.0)
    bk.record_success(now=11.5)
    assert bk.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_and_resets_timer() -> None:
    bk = _bk(threshold=1, recovery=10.0)
    bk.record_failure(now=0.0)
    bk.allow(now=11.0)
    bk.record_failure(now=11.5)
    assert bk.state is CircuitState.OPEN
    assert bk.allow(now=21.0) is False
    assert bk.allow(now=22.0) is True


async def test_half_open_only_one_probe_in_flight() -> None:
    bk = _bk(threshold=1, recovery=0.0)
    bk.record_failure(now=0.0)
    granted_first = bk.allow(now=1.0)
    granted_second = bk.allow(now=1.0)
    assert granted_first is True
    assert granted_second is False
    bk.record_success(now=1.1)
    assert bk.allow(now=2.0) is True
