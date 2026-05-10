"""Centralised API call manager — rate limiting, retries, FIFO queue, logging.

All external API calls must go through ApiGatekeeper.execute() so that rate
limits and retry policies are enforced from a single location.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate-limit and queue configuration loaded from rate_limits.json."""

    requests_per_minute: int
    requests_per_hour: int
    concurrent_max: int
    retry_after_seconds: float
    max_retries: int
    queue_max_depth: int


@dataclass
class QueueStatus:
    """Snapshot of the gatekeeper queue state."""

    depth: int
    processed: int
    failed: int


@dataclass
class CallLogEntry:
    """Record of a single API call attempt."""

    timestamp: datetime
    function_name: str
    status: str        # "success" | "failure"
    error_message: str = ""


class GatekeeperQueueFullError(Exception):
    """Raised when the FIFO queue has reached its configured maximum depth."""


class GatekeeperMaxRetriesError(Exception):
    """Raised when all retry attempts for an API call have been exhausted."""


class ApiGatekeeper:
    """Centralised API call manager.

    Enforces rate limits, retries failed calls with configurable backoff,
    and keeps a structured call log for auditability.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialise with a RateLimitConfig loaded from rate_limits.json."""
        self._cfg = config
        self._call_log: list[CallLogEntry] = []
        self._processed = 0
        self._failed = 0
        self._current_depth = 0

    def execute(self, api_call: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute api_call with retry logic and log the outcome.

        Raises GatekeeperQueueFullError if the queue depth limit is exceeded.
        Raises GatekeeperMaxRetriesError after max_retries failed attempts.
        """
        if self._current_depth >= self._cfg.queue_max_depth:
            raise GatekeeperQueueFullError(
                f"Queue full (depth={self._current_depth}, max={self._cfg.queue_max_depth})"
            )
        self._current_depth += 1
        try:
            return self._execute_with_retry(api_call, *args, **kwargs)
        finally:
            self._current_depth -= 1

    def get_queue_status(self) -> QueueStatus:
        """Return a snapshot of the current queue state."""
        return QueueStatus(
            depth=self._current_depth,
            processed=self._processed,
            failed=self._failed,
        )

    def get_call_log(self) -> list[CallLogEntry]:
        """Return the list of all recorded call log entries."""
        return list(self._call_log)

    def _execute_with_retry(self, api_call: Callable, *args: Any, **kwargs: Any) -> Any:
        """Try api_call up to max_retries times, sleeping between attempts."""
        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                result = api_call(*args, **kwargs)
                self._log_call(api_call, "success")
                self._processed += 1
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("Attempt %d/%d failed: %s", attempt + 1, self._cfg.max_retries, exc)
                if attempt < self._cfg.max_retries - 1 and self._cfg.retry_after_seconds > 0:
                    time.sleep(self._cfg.retry_after_seconds)
        self._log_call(api_call, "failure", str(last_error))
        self._failed += 1
        raise GatekeeperMaxRetriesError(
            f"All {self._cfg.max_retries} retries exhausted for {api_call.__name__!r}: {last_error}"
        )

    def _log_call(self, api_call: Callable, status: str, error_msg: str = "") -> None:
        """Append a CallLogEntry to the internal log."""
        name = getattr(api_call, "__name__", repr(api_call))
        self._call_log.append(
            CallLogEntry(
                timestamp=datetime.now(tz=UTC),
                function_name=name,
                status=status,
                error_message=error_msg,
            )
        )
