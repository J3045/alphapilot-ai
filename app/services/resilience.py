# pyrefly: ignore[missing-import]
"""Shared resilience and retry utility for AlphaPilot AI.

Provides:
- Synchronous retry execution with exponential backoff and bounded jitter.
- Precise error classification (transient vs terminal errors).
- Clean propagation of unhandled/exhausted errors without swallowing.
- Privacy-safe diagnostic logging (never logs prompts, responses, or API keys).
"""

import logging
import random
import time
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from app.config import settings

logger = logging.getLogger("app.services.resilience")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Deadline and Budget Primitives
# ---------------------------------------------------------------------------

class WorkflowDeadlineExceededError(TimeoutError):
    """Raised when an operation or workflow exceeds its configured execution budget."""
    pass


class WorkflowDeadline:
    """Monotonic execution budget tracker.

    Tracks remaining time and deadline expiration using monotonic clocks (time.perf_counter).
    Provides deterministic testing via injectable time_fn.
    """

    def __init__(
        self,
        timeout_seconds: float | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.time_fn = time_fn if time_fn is not None else time.perf_counter
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.workflow_timeout_seconds
        )
        self.start_time = self.time_fn()
        self.deadline = self.start_time + self.timeout_seconds

    def elapsed_seconds(self) -> float:
        """Elapsed time since budget initialization in seconds."""
        return max(0.0, self.time_fn() - self.start_time)

    def remaining_seconds(self) -> float:
        """Remaining time before deadline in seconds (floored at 0.0)."""
        return max(0.0, self.deadline - self.time_fn())

    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return (self.deadline - self.time_fn()) <= 0.0

    def check_or_raise(self) -> None:
        """Raise WorkflowDeadlineExceededError if deadline has passed."""
        if self.is_expired():
            raise WorkflowDeadlineExceededError(
                f"Workflow execution budget of {self.timeout_seconds:.1f}s exceeded."
            )


# ---------------------------------------------------------------------------
# Error Classification
# ---------------------------------------------------------------------------

def is_retryable_error(exc: Exception) -> bool:
    """Classify whether an exception is transient and safe to retry.

    Retryable:
    - Rate limiting (HTTP 429, RateLimitError)
    - Transient server errors (HTTP 500, 502, 503, 504, InternalServerError)
    - Connection / Network failures (ConnectionError, APIConnectionError, NetworkError)
    - Request timeouts (TimeoutError, APITimeoutError, TimeoutException)

    Non-Retryable (Terminal):
    - Workflow deadline exceeded (WorkflowDeadlineExceededError)
    - Authentication errors (HTTP 401, AuthenticationError)
    - Permission / Authorization errors (HTTP 403, PermissionDeniedError)
    - Invalid requests / Client errors (HTTP 400, BadRequestError, 404, 422)
    - Context length exceeded
    - Validation errors (pydantic.ValidationError, ValueError)
    - Programming errors (TypeError, KeyError, AttributeError, IndexError, AssertionError)
    - Unknown errors (Must NOT automatically be treated as transient)
    """
    if exc is None:
        return False

    # Budget exhaustion is non-retryable (no budget remaining)
    if isinstance(exc, WorkflowDeadlineExceededError):
        return False

    # 1. Non-retryable programming & validation errors
    if isinstance(
        exc,
        (
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            IndexError,
            AssertionError,
            NotImplementedError,
        ),
    ):
        return False

    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    # 2. Check HTTP status code if present (e.g. from OpenAI APIStatusError or httpx)
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        # Explicit non-retryable client status codes
        if status_code in (400, 401, 403, 404, 422):
            return False
        # Explicit retryable status codes
        if status_code in (408, 429, 500, 502, 503, 504):
            return True

    # 3. Non-retryable OpenAI / Auth / Request patterns
    if (
        "authentication" in exc_type
        or "permissiondenied" in exc_type
        or "badrequest" in exc_type
        or "notfound" in exc_type
        or "unprocessable" in exc_type
        or "auth" in exc_type
        or "unauthorized" in exc_msg
        or "api_key" in exc_msg
        or "invalid api key" in exc_msg
    ):
        return False

    # 4. Context length / prompt size exceeded
    if (
        "context_length" in exc_type
        or "maximum context" in exc_msg
        or "context window" in exc_msg
        or "context length" in exc_msg
    ):
        return False

    # 5. Retryable: Timeouts
    if (
        isinstance(exc, TimeoutError)
        or "timeout" in exc_type
        or "timed out" in exc_msg
        or "timeout" in exc_msg
    ):
        return True

    # 6. Retryable: Connection & Network failures
    if (
        isinstance(exc, ConnectionError)
        or "connection" in exc_type
        or "network" in exc_type
        or "connection reset" in exc_msg
        or "connection refused" in exc_msg
        or "failed to establish a new connection" in exc_msg
    ):
        return True

    # 7. Retryable: Rate limits (HTTP 429)
    if "ratelimit" in exc_type or "rate_limit" in exc_type or "429" in exc_msg:
        return True

    # 8. Retryable: Server errors (5xx)
    if (
        "internalserver" in exc_type
        or "serviceunavailable" in exc_type
        or "badgateway" in exc_type
        or "gatewaytimeout" in exc_type
        or "500" in exc_msg
        or "502" in exc_msg
        or "503" in exc_msg
        or "504" in exc_msg
    ):
        return True

    # 9. Conservative fallback: Unknown errors are NOT treated as transient
    return False


# ---------------------------------------------------------------------------
# Backoff Calculation
# ---------------------------------------------------------------------------

def calculate_backoff(
    attempt: int,
    initial_backoff: float,
    max_backoff: float,
    jitter_factor: float = 0.25,
    rng: Callable[[float, float], float] | None = None,
) -> float:
    """Calculate exponential backoff delay with bounded jitter.

    Args:
        attempt: The 1-indexed attempt number (1 for first retry, 2 for second, etc.).
        initial_backoff: Starting backoff delay in seconds.
        max_backoff: Maximum backoff ceiling in seconds.
        jitter_factor: Proportion of base delay applied as lower-bound jitter (default 0.25).
        rng: Optional uniform RNG function (defaults to random.uniform for production).

    Formula:
        base = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
        low = base * (1.0 - jitter_factor)
        high = base
        jittered = rng(low, high)
    """
    if attempt < 1:
        attempt = 1

    base_delay = min(max_backoff, initial_backoff * (2.0 ** (attempt - 1)))
    low_bound = base_delay * (1.0 - max(0.0, min(1.0, jitter_factor)))
    high_bound = base_delay

    uniform_fn = rng if rng is not None else random.uniform
    jittered_delay = uniform_fn(low_bound, high_bound)

    return max(0.0, min(max_backoff, jittered_delay))


# ---------------------------------------------------------------------------
# Retry Executor
# ---------------------------------------------------------------------------

def execute_with_retry(
    fn: Callable[[], T],
    max_retries: int | None = None,
    initial_backoff_seconds: float | None = None,
    max_backoff_seconds: float | None = None,
    retryable_predicate: Callable[[Exception], bool] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    rng: Callable[[float, float], float] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    deadline: WorkflowDeadline | None = None,
) -> T:
    """Execute a callable with synchronous exponential backoff and bounded jitter.

    Semantics:
    - `max_retries` specifies the number of retry attempts AFTER an initial failure.
      Total potential attempts = 1 + max_retries (e.g. max_retries=2 means at most 3 total attempts).
    - If `deadline` is provided, checks remaining budget before every attempt and backoff sleep.
      Does not start attempts or sleep if the budget has expired or is exceeded by the backoff.
    - If the callable succeeds on attempt 1, sleep is never called and execution returns immediately.
    - If an error is non-retryable (terminal or unknown), it is raised immediately without sleeping.
    - If retries are exhausted, the last exception is re-raised without being swallowed.

    Privacy & Telemetry:
    - Logs concise diagnostic attempt events via `app.services.resilience`.
    - Never logs prompts, model outputs, API keys, or private portfolio details.
    """
    retries = max_retries if max_retries is not None else settings.max_retries
    initial_backoff = (
        initial_backoff_seconds
        if initial_backoff_seconds is not None
        else settings.retry_initial_backoff_seconds
    )
    max_backoff = (
        max_backoff_seconds
        if max_backoff_seconds is not None
        else settings.retry_max_backoff_seconds
    )
    predicate = retryable_predicate if retryable_predicate is not None else is_retryable_error
    sleep = sleep_fn if sleep_fn is not None else time.sleep

    total_attempts = 1 + max(0, retries)

    for attempt in range(1, total_attempts + 1):
        if deadline is not None:
            deadline.check_or_raise()

        try:
            return fn()
        except Exception as exc:
            # If on the final attempt or error is non-retryable, re-raise immediately
            if attempt >= total_attempts or not predicate(exc):
                raise

            if deadline is not None:
                if deadline.is_expired():
                    raise WorkflowDeadlineExceededError(
                        f"Workflow execution budget of {deadline.timeout_seconds:.1f}s exceeded before scheduling retry."
                    ) from exc

                rem = deadline.remaining_seconds()

            delay = calculate_backoff(
                attempt=attempt,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                rng=rng,
            )

            if deadline is not None:
                # Do not sleep for a backoff interval that exceeds the remaining budget
                # Check whether sufficient budget remains before scheduling a retry
                if delay >= rem:
                    raise WorkflowDeadlineExceededError(
                        f"Workflow execution budget exhausted: remaining time ({rem:.2f}s) is less than required backoff ({delay:.2f}s)."
                    ) from exc

            logger.warning(
                "transient_error_retry_scheduled",
                extra={
                    "attempt": attempt,
                    "max_attempts": total_attempts,
                    "retry_delay_seconds": round(delay, 4),
                    "error_type": type(exc).__name__,
                },
            )

            if on_retry is not None:
                on_retry(attempt, exc, delay)

            sleep(delay)

    # Fallback return (unreachable under normal exception propagation)
    raise RuntimeError("Unexpected execution state in execute_with_retry")
