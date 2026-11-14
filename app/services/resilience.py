# pyrefly: ignore[missing-import]
"""Shared resilience, retry, and bounded concurrency utility for AlphaPilot AI.

Provides:
- Synchronous and asynchronous retry execution with exponential backoff and bounded jitter.
- Precise error classification (transient vs terminal errors).
- Application-level bounded concurrency (asyncio.Semaphore) integrated with workflow deadlines.
- Clean propagation of unhandled/exhausted errors without swallowing.
- Privacy-safe diagnostic logging (never logs prompts, responses, or API keys).
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
import inspect
import logging
import random
import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, TypeVar
import weakref

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


_active_workflow_deadline: ContextVar["WorkflowDeadline | None"] = ContextVar(
    "alphapilot_active_workflow_deadline",
    default=None,
)


def get_active_workflow_deadline() -> "WorkflowDeadline | None":
    """Return the active WorkflowDeadline bound to the current async execution context, if any."""
    return _active_workflow_deadline.get()


@contextmanager
def workflow_deadline_scope(deadline: "WorkflowDeadline | None") -> Iterator["WorkflowDeadline | None"]:
    """Bind a WorkflowDeadline to the current execution context cleanly."""
    token = _active_workflow_deadline.set(deadline)
    try:
        yield deadline
    finally:
        _active_workflow_deadline.reset(token)


class ConcurrencyLimitExceededError(RuntimeError):
    """Raised when an operation cannot obtain an LLM concurrency slot within its wait budget."""
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

    # Budget exhaustion and local backpressure saturation are non-retryable
    if isinstance(exc, (WorkflowDeadlineExceededError, ConcurrencyLimitExceededError)):
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
# Retry Executors (Sync & Async)
# ---------------------------------------------------------------------------

def _resolve_retry_config(
    max_retries: int | None,
    initial_backoff_seconds: float | None,
    max_backoff_seconds: float | None,
    retryable_predicate: Callable[[Exception], bool] | None,
) -> tuple[int, float, float, Callable[[Exception], bool]]:
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
    total_attempts = 1 + max(0, retries)
    return total_attempts, initial_backoff, max_backoff, predicate


def _prepare_retry_delay(
    attempt: int,
    total_attempts: int,
    exc: Exception,
    predicate: Callable[[Exception], bool],
    deadline: WorkflowDeadline | None,
    initial_backoff: float,
    max_backoff: float,
    rng: Callable[[float, float], float] | None,
    on_retry: Callable[[int, Exception, float], None] | None,
) -> float:
    """Validate retry eligibility and workflow budget, returning the backoff delay."""
    if attempt >= total_attempts or not predicate(exc):
        raise exc

    rem: float | None = None
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

    retry_after = getattr(exc, "retry_after_seconds", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        if float(retry_after) > max_backoff:
            raise exc
        delay = max(delay, float(retry_after))

    if deadline is not None and rem is not None and delay >= rem:
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

    return delay


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
    total_attempts, initial_backoff, max_backoff, predicate = _resolve_retry_config(
        max_retries, initial_backoff_seconds, max_backoff_seconds, retryable_predicate
    )
    sleep = sleep_fn if sleep_fn is not None else time.sleep

    for attempt in range(1, total_attempts + 1):
        if deadline is not None:
            deadline.check_or_raise()

        try:
            return fn()
        except Exception as exc:
            delay = _prepare_retry_delay(
                attempt=attempt,
                total_attempts=total_attempts,
                exc=exc,
                predicate=predicate,
                deadline=deadline,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                rng=rng,
                on_retry=on_retry,
            )
            sleep(delay)

    raise RuntimeError("Unexpected execution state in execute_with_retry")


async def execute_with_retry_async(
    fn: Callable[[], Awaitable[T]],
    max_retries: int | None = None,
    initial_backoff_seconds: float | None = None,
    max_backoff_seconds: float | None = None,
    retryable_predicate: Callable[[Exception], bool] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    rng: Callable[[float, float], float] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    deadline: WorkflowDeadline | None = None,
) -> T:
    """Execute an async callable with non-blocking exponential backoff and bounded jitter."""
    total_attempts, initial_backoff, max_backoff, predicate = _resolve_retry_config(
        max_retries, initial_backoff_seconds, max_backoff_seconds, retryable_predicate
    )
    sleep = sleep_fn if sleep_fn is not None else asyncio.sleep

    for attempt in range(1, total_attempts + 1):
        if deadline is not None:
            deadline.check_or_raise()

        try:
            return await fn()
        except Exception as exc:
            delay = _prepare_retry_delay(
                attempt=attempt,
                total_attempts=total_attempts,
                exc=exc,
                predicate=predicate,
                deadline=deadline,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                rng=rng,
                on_retry=on_retry,
            )
            await sleep(delay)

    raise RuntimeError("Unexpected execution state in execute_with_retry_async")


# ---------------------------------------------------------------------------
# Application-Level Bounded Concurrency & Backpressure
# ---------------------------------------------------------------------------

class LLMConcurrencyLimiter:
    """Bounds concurrent in-flight LLM operations per worker event loop.

    Uses per-event-loop `asyncio.Semaphore` instances stored in a `WeakKeyDictionary`
    so each worker loop enforces bounded concurrency without leaking waiters across
    closed test loops. Waiting is bounded by `min(queue_timeout, deadline.remaining_seconds())`
    to prevent unbounded request queues and retry amplification under load.
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        queue_timeout_seconds: float | None = None,
    ) -> None:
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}.")
        if queue_timeout_seconds is not None and queue_timeout_seconds <= 0.0:
            raise ValueError(f"queue_timeout_seconds must be > 0, got {queue_timeout_seconds}.")

        self._max_concurrent = max_concurrent
        self._queue_timeout_seconds = queue_timeout_seconds
        self._lock = threading.Lock()
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def max_concurrent(self) -> int:
        return (
            self._max_concurrent
            if self._max_concurrent is not None
            else settings.max_concurrent_llm_requests
        )

    @property
    def queue_timeout_seconds(self) -> float:
        return (
            self._queue_timeout_seconds
            if self._queue_timeout_seconds is not None
            else settings.llm_queue_timeout_seconds
        )

    def _get_loop_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            sem = self._semaphores.get(loop)
            if sem is None:
                sem = asyncio.Semaphore(self.max_concurrent)
                self._semaphores[loop] = sem
            return sem

    def available_slots(self) -> int:
        """Return currently available slots on the running event loop (diagnostic)."""
        sem = self._get_loop_semaphore()
        return max(0, getattr(sem, "_value", 0))

    @asynccontextmanager
    async def acquire(self, deadline: WorkflowDeadline | None = None) -> AsyncIterator[None]:
        """Acquire a concurrency slot with bounded wait integrated with WorkflowDeadline."""
        if deadline is not None:
            deadline.check_or_raise()

        semaphore = self._get_loop_semaphore()
        wait_budget = self.queue_timeout_seconds
        if deadline is not None:
            wait_budget = min(wait_budget, deadline.remaining_seconds())
            if wait_budget <= 0.0:
                raise WorkflowDeadlineExceededError(
                    f"Workflow execution budget of {deadline.timeout_seconds:.1f}s exceeded while waiting for capacity."
                )

        acquired = False
        try:
            if not semaphore.locked():
                await semaphore.acquire()
                acquired = True
            else:
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=wait_budget)
                    acquired = True
                except TimeoutError as exc:
                    if deadline is not None and deadline.is_expired():
                        raise WorkflowDeadlineExceededError(
                            f"Workflow execution budget of {deadline.timeout_seconds:.1f}s exceeded while waiting for LLM capacity."
                        ) from exc
                    logger.warning(
                        "llm_concurrency_limit_exceeded",
                        extra={
                            "max_concurrent": self.max_concurrent,
                            "wait_budget_seconds": round(wait_budget, 4),
                        },
                    )
                    raise ConcurrencyLimitExceededError(
                        f"Model execution capacity saturated ({self.max_concurrent} concurrent requests)."
                    ) from exc
            yield
        finally:
            if acquired:
                semaphore.release()


_DEFAULT_LIMITER_LOCK = threading.Lock()
_DEFAULT_LLM_LIMITER: LLMConcurrencyLimiter | None = None


def get_default_llm_limiter() -> LLMConcurrencyLimiter:
    """Return the process-local shared LLM concurrency limiter."""
    global _DEFAULT_LLM_LIMITER
    if _DEFAULT_LLM_LIMITER is None:
        with _DEFAULT_LIMITER_LOCK:
            if _DEFAULT_LLM_LIMITER is None:
                _DEFAULT_LLM_LIMITER = LLMConcurrencyLimiter()
    return _DEFAULT_LLM_LIMITER


def reset_default_llm_limiter() -> None:
    """Reset the process-local LLM concurrency limiter (for tests or shutdown)."""
    global _DEFAULT_LLM_LIMITER
    with _DEFAULT_LIMITER_LOCK:
        _DEFAULT_LLM_LIMITER = None


def supports_async_method(obj: Any, method_name: str) -> bool:
    """Return True only when `obj.<method_name>` is an actual coroutine function.

    Guards against unittest.mock.MagicMock auto-generated attributes, which return
    MagicMock instances rather than coroutine functions.
    """
    method = getattr(obj, method_name, None)
    if method is None:
        return False
    return inspect.iscoroutinefunction(method)


def sanitize_provider_error(exc: Exception) -> str:
    """Safely describe provider, vector storage, deadline, and backpressure errors without leaking secrets."""
    if isinstance(exc, WorkflowDeadlineExceededError):
        return "Workflow execution deadline exceeded."
    if isinstance(exc, ConcurrencyLimitExceededError):
        return "Model execution capacity temporarily saturated. Please try again shortly."

    exc_type = type(exc).__name__.lower()
    exc_module = (getattr(type(exc), "__module__", "") or "").lower()
    exc_msg = str(exc).lower()

    if "dimensionmismatch" in exc_type or "embeddingdimension" in exc_type:
        return "Vector storage embedding dimension or model mismatch."
    if "vectorstoreconnection" in exc_type:
        return "Vector storage database connection failed."
    if "vectorstore" in exc_type or any(
        pkg in exc_module for pkg in ("sqlalchemy", "asyncpg", "pgvector", "psycopg", "alembic")
    ):
        if "timeout" in exc_type or "timed out" in exc_msg:
            return "Vector storage database request timed out."
        if "authentication" in exc_type or "password" in exc_msg:
            return "Vector storage authentication failure."
        if "connection" in exc_type or "connect" in exc_msg or "operational" in exc_type:
            return "Vector storage database connection failed."
        if "dimension" in exc_msg:
            return "Vector storage embedding dimension or model mismatch."
        return f"Vector storage operation failed ({type(exc).__name__})."

    if "deadline" in exc_type or "deadline" in exc_msg or "budget" in exc_msg:
        return "Workflow execution deadline exceeded."
    if "concurrency" in exc_type or "capacity saturated" in exc_msg:
        return "Model execution capacity temporarily saturated. Please try again shortly."
    if "broker" in exc_type or "brokertoken" in exc_type:
        if "expired" in exc_type:
            return "Broker session expired. Please reconnect your brokerage account."
        if "revoked" in exc_type:
            return "Broker connection has been revoked. Please reconnect your brokerage account."
        if "permission" in exc_type:
            return "Broker API permission denied."
        if "ratelimit" in exc_type:
            return "Broker API rate limit exceeded. Please retry later."
        if "timeout" in exc_type:
            return "Request to brokerage provider timed out."
        if "validation" in exc_type:
            return "Brokerage provider returned an invalid or unexpected response."
        if "configuration" in exc_type:
            return "Brokerage integration is not properly configured."
        return f"Brokerage provider operation failed ({type(exc).__name__})."
    if "authentication" in exc_type or "auth" in exc_type or "authentication" in exc_msg or "auth" in exc_msg:
        return "Authentication failure with model provider."
    if "timeout" in exc_type or "timed out" in exc_msg:
        return "Request to model provider timed out."
    if "connection" in exc_type or "connect" in exc_msg:
        return "Connection failure with model provider."
    return f"Model provider request failed ({type(exc).__name__})."


async def invoke_llm_with_resilience_async(
    client: Any,
    messages: Any,
    deadline: WorkflowDeadline | None = None,
    concurrency_limiter: LLMConcurrencyLimiter | None = None,
) -> Any:
    """Invoke an LLM client asynchronously with bounded concurrency, retries, and deadline checks.

    Uses native `ainvoke` when available and falls back to `asyncio.to_thread(client.invoke, ...)`
    for synchronous-only clients/mocks. Acquires a concurrency slot only around each single
    provider call so backoff sleeps never hold scarce execution slots.
    """
    limiter = concurrency_limiter if concurrency_limiter is not None else get_default_llm_limiter()
    use_async = supports_async_method(client, "ainvoke")

    async def _call_once() -> Any:
        async with limiter.acquire(deadline=deadline):
            if use_async:
                return await client.ainvoke(messages)
            return await asyncio.to_thread(client.invoke, messages)

    return await execute_with_retry_async(_call_once, deadline=deadline)
