"""Tests for shared/gatekeeper.py — ApiGatekeeper (neural_signal)."""

from __future__ import annotations

import pytest

from neural_signal.shared.gatekeeper import (
    ApiGatekeeper,
    GatekeeperMaxRetriesError,
    GatekeeperQueueFullError,
    RateLimitConfig,
)


@pytest.fixture()
def gk() -> ApiGatekeeper:
    cfg = RateLimitConfig(max_retries=2, queue_max_depth=5, retry_after_seconds=0)
    return ApiGatekeeper(config=cfg)


def test_gatekeeper_instantiates(gk):
    assert gk is not None


def test_gatekeeper_execute_calls_function(gk):
    calls = []
    gk.execute(calls.append, 1)
    assert calls == [1]


def test_gatekeeper_execute_returns_result(gk):
    assert gk.execute(lambda: 42) == 42


def test_gatekeeper_queue_status_returns_object(gk):
    status = gk.get_queue_status()
    assert status is not None


def test_gatekeeper_queue_depth_starts_at_0(gk):
    assert gk.get_queue_status().depth == 0


def test_gatekeeper_processed_count_grows_after_execute(gk):
    gk.execute(lambda: None)
    assert gk.get_queue_status().processed == 1


def test_gatekeeper_failed_count_grows_on_exception(gk):
    with pytest.raises(GatekeeperMaxRetriesError):
        gk.execute(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert gk.get_queue_status().failed == 1


def test_gatekeeper_get_call_log_returns_list(gk):
    gk.execute(lambda: None)
    assert isinstance(gk.get_call_log(), list)


def test_gatekeeper_call_log_entry_has_timestamp(gk):
    gk.execute(lambda: None)
    entry = gk.get_call_log()[0]
    assert entry.timestamp > 0


def test_gatekeeper_call_log_entry_has_function_name(gk):
    def my_func():
        return 1
    gk.execute(my_func)
    entry = gk.get_call_log()[0]
    assert "my_func" in entry.function_name


def test_gatekeeper_call_log_entry_has_status_success(gk):
    gk.execute(lambda: None)
    assert gk.get_call_log()[0].status == "success"


def test_gatekeeper_call_log_entry_has_status_failure(gk):
    with pytest.raises(GatekeeperMaxRetriesError):
        gk.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert gk.get_call_log()[-1].status == "failure"


def test_gatekeeper_retries_on_exception():
    attempts = []
    cfg = RateLimitConfig(max_retries=2, retry_after_seconds=0)
    gk = ApiGatekeeper(config=cfg)

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("fail")
        return "ok"

    result = gk.execute(flaky)
    assert result == "ok"
    assert len(attempts) == 3


def test_gatekeeper_raises_after_max_retries(gk):
    with pytest.raises(GatekeeperMaxRetriesError):
        gk.execute(lambda: (_ for _ in ()).throw(ValueError("always fails")))


def test_gatekeeper_raises_when_queue_full():
    cfg = RateLimitConfig(max_retries=0, queue_max_depth=0)
    gk = ApiGatekeeper(config=cfg)
    with pytest.raises(GatekeeperQueueFullError):
        gk.execute(lambda: None)
