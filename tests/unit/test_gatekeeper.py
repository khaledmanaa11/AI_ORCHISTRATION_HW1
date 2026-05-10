"""Unit tests for shared/gatekeeper.py — ApiGatekeeper."""

import pytest

from signal_dataset.shared.gatekeeper import (
    ApiGatekeeper,
    GatekeeperMaxRetriesError,
    GatekeeperQueueFullError,
    RateLimitConfig,
)


@pytest.fixture
def minimal_config():
    """RateLimitConfig with very permissive settings for fast tests."""
    return RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        concurrent_max=5,
        retry_after_seconds=0,
        max_retries=3,
        queue_max_depth=10,
    )


@pytest.fixture
def gatekeeper(minimal_config):
    """Default ApiGatekeeper instance."""
    return ApiGatekeeper(minimal_config)


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_gatekeeper_instantiates_with_valid_config(minimal_config):
    gk = ApiGatekeeper(minimal_config)
    assert gk is not None


# ── execute() ─────────────────────────────────────────────────────────────────

def test_gatekeeper_execute_calls_function(gatekeeper):
    called = []
    gatekeeper.execute(lambda: called.append(1))
    assert called == [1]


def test_gatekeeper_execute_returns_function_result(gatekeeper):
    result = gatekeeper.execute(lambda: 42)
    assert result == 42


def test_gatekeeper_execute_logs_call(gatekeeper):
    gatekeeper.execute(lambda: None)
    assert len(gatekeeper.get_call_log()) == 1


def test_gatekeeper_execute_single_callable_no_error(gatekeeper):
    gatekeeper.execute(lambda: "ok")  # must not raise


# ── Queue status ──────────────────────────────────────────────────────────────

def test_gatekeeper_queue_status_returns_queue_status_object(gatekeeper):
    from signal_dataset.shared.gatekeeper import QueueStatus
    status = gatekeeper.get_queue_status()
    assert isinstance(status, QueueStatus)


def test_gatekeeper_queue_status_depth_starts_at_0(gatekeeper):
    assert gatekeeper.get_queue_status().depth == 0


def test_gatekeeper_queue_status_processed_starts_at_0(gatekeeper):
    assert gatekeeper.get_queue_status().processed == 0


def test_gatekeeper_queue_status_failed_starts_at_0(gatekeeper):
    assert gatekeeper.get_queue_status().failed == 0


# ── Call log ──────────────────────────────────────────────────────────────────

def test_gatekeeper_get_call_log_returns_list(gatekeeper):
    assert isinstance(gatekeeper.get_call_log(), list)


def test_gatekeeper_get_call_log_is_empty_initially(gatekeeper):
    assert gatekeeper.get_call_log() == []


def test_gatekeeper_get_call_log_grows_after_execute(gatekeeper):
    gatekeeper.execute(lambda: None)
    gatekeeper.execute(lambda: None)
    assert len(gatekeeper.get_call_log()) == 2


def test_gatekeeper_call_log_entry_has_timestamp(gatekeeper):
    gatekeeper.execute(lambda: None)
    entry = gatekeeper.get_call_log()[0]
    assert entry.timestamp is not None


def test_gatekeeper_call_log_entry_has_function_name(gatekeeper):
    def my_func():
        return 1

    gatekeeper.execute(my_func)
    entry = gatekeeper.get_call_log()[0]
    assert "my_func" in entry.function_name


def test_gatekeeper_call_log_entry_has_status_success(gatekeeper):
    gatekeeper.execute(lambda: None)
    entry = gatekeeper.get_call_log()[0]
    assert entry.status == "success"


def test_gatekeeper_call_log_entry_has_status_failure_on_exception(gatekeeper):
    def bad():
        raise ValueError("oops")

    with pytest.raises(GatekeeperMaxRetriesError):
        gatekeeper.execute(bad)
    entry = gatekeeper.get_call_log()[-1]
    assert entry.status == "failure"


# ── Retry behaviour ───────────────────────────────────────────────────────────

def test_gatekeeper_execute_retries_on_exception(minimal_config):
    cfg = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        concurrent_max=5,
        retry_after_seconds=0,
        max_retries=3,
        queue_max_depth=10,
    )
    gk = ApiGatekeeper(cfg)
    counter = {"n": 0}

    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = gk.execute(flaky)
    assert result == "ok"
    assert counter["n"] == 3


def test_gatekeeper_execute_raises_after_max_retries(gatekeeper):
    def always_fails():
        raise RuntimeError("permanent")

    with pytest.raises(GatekeeperMaxRetriesError):
        gatekeeper.execute(always_fails)


def test_gatekeeper_execute_respects_max_retries_config():
    cfg = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        concurrent_max=5,
        retry_after_seconds=0,
        max_retries=2,
        queue_max_depth=10,
    )
    gk = ApiGatekeeper(cfg)
    counter = {"n": 0}

    def always_fails():
        counter["n"] += 1
        raise RuntimeError("permanent")

    with pytest.raises(GatekeeperMaxRetriesError):
        gk.execute(always_fails)
    assert counter["n"] == 2


# ── Queue depth / backpressure ────────────────────────────────────────────────

def test_gatekeeper_queue_depth_not_exceeded(minimal_config):
    gk = ApiGatekeeper(minimal_config)
    for _ in range(5):
        gk.execute(lambda: None)
    assert gk.get_queue_status().depth == 0


def test_gatekeeper_raises_when_queue_full():
    cfg = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        concurrent_max=5,
        retry_after_seconds=0,
        max_retries=1,
        queue_max_depth=0,  # zero depth → always full
    )
    gk = ApiGatekeeper(cfg)
    with pytest.raises(GatekeeperQueueFullError):
        gk.execute(lambda: None)
