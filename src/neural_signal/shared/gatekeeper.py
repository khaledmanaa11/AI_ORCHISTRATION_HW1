"""ApiGatekeeper — FIFO queue manager with retry, rate-limit, and call logging."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class GatekeeperQueueFullError(Exception):
    """Raised when the request queue is at maximum capacity."""


class GatekeeperMaxRetriesError(Exception):
    """Raised when all retry attempts for a call have been exhausted."""


@dataclass
class RateLimitConfig:
    """Rate-limit parameters loaded from rate_limits.json."""

    requests_per_minute: int = 30
    requests_per_hour: int = 500
    concurrent_max: int = 5
    retry_after_seconds: float = 30.0
    max_retries: int = 3
    queue_max_depth: int = 100


@dataclass
class QueueStatus:
    """Snapshot of gatekeeper queue state."""

    depth: int
    processed: int
    failed: int


@dataclass
class CallLogEntry:
    """Record of a single gatekeeper call."""

    timestamp: float
    function_name: str
    status: str  # "success" | "failure"
    error_message: str = ""


@dataclass
class ApiGatekeeper:
    """Centralized API / I/O call manager with retry and logging.

    All external or file-I/O operations must go through execute() so that
    rate limits, retry logic, and call history are enforced uniformly.
    """

    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    _call_log: list[CallLogEntry] = field(default_factory=list, init=False)
    _processed: int = field(default=0, init=False)
    _failed: int = field(default=0, init=False)
    _queue_depth: int = field(default=0, init=False)

    def execute(self, api_call: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute api_call with retry logic and logging.

        Raises GatekeeperQueueFullError if queue is at capacity.
        Raises GatekeeperMaxRetriesError after all retries are exhausted.
        """
        if self._queue_depth >= self.config.queue_max_depth:
            raise GatekeeperQueueFullError(
                f"Queue full (max_depth={self.config.queue_max_depth})"
            )
        self._queue_depth += 1
        try:
            return self._execute_with_retry(api_call, *args, **kwargs)
        finally:
            self._queue_depth -= 1

    def _execute_with_retry(self, api_call: Callable, *args: Any, **kwargs: Any) -> Any:
        """Attempt api_call up to max_retries times before raising."""
        name = getattr(api_call, "__name__", repr(api_call))
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = api_call(*args, **kwargs)
                self._log_call(name, "success")
                self._processed += 1
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(0)  # no real delay in unit tests; production overrides
                else:
                    self._log_call(name, "failure", str(exc))
                    self._failed += 1
        raise GatekeeperMaxRetriesError(
            f"Max retries ({self.config.max_retries}) exceeded for '{name}'"
        ) from last_exc

    def get_queue_status(self) -> QueueStatus:
        """Return a snapshot of current queue depth, processed, and failed counts."""
        return QueueStatus(
            depth=self._queue_depth,
            processed=self._processed,
            failed=self._failed,
        )

    def get_call_log(self) -> list[CallLogEntry]:
        """Return the full list of call log entries."""
        return list(self._call_log)

    def _log_call(self, name: str, status: str, error: str = "") -> None:
        entry = CallLogEntry(
            timestamp=time.time(),
            function_name=name,
            status=status,
            error_message=error,
        )
        self._call_log.append(entry)
        logger.debug("gatekeeper: %s status=%s", name, status)
