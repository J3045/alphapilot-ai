# pyrefly: ignore[missing-import]
"""Deterministic tests for AlphaPilot AI shared resilience layer."""

from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.resilience import (
    calculate_backoff,
    execute_with_retry,
    is_retryable_error,
)


# Synthetic exception classes matching provider hierarchies without live dependencies
class SyntheticOpenAIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SyntheticRateLimitError(SyntheticOpenAIError):
    def __init__(self, message: str = "Rate limit reached") -> None:
        super().__init__(message, status_code=429)


class SyntheticAuthenticationError(SyntheticOpenAIError):
    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(message, status_code=401)


class SyntheticBadRequestError(SyntheticOpenAIError):
    def __init__(self, message: str = "Invalid request format") -> None:
        super().__init__(message, status_code=400)


# ---------------------------------------------------------------------------
# 1. Immediate Success Without Sleeping
# ---------------------------------------------------------------------------

def test_immediate_success_without_sleeping():
    """Verify that a successful operation returns immediately and never invokes sleep."""
    fn = MagicMock(return_value="result_ok")
    mock_sleep = MagicMock()

    result = execute_with_retry(
        fn=fn,
        max_retries=2,
        sleep_fn=mock_sleep,
    )

    assert result == "result_ok"
    assert fn.call_count == 1
    assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 2. Transient Error Followed by Success
# ---------------------------------------------------------------------------

def test_transient_error_followed_by_success():
    """Verify transient rate limit recovers on subsequent attempt after backoff sleep."""
    fn = MagicMock(
        side_effect=[
            SyntheticRateLimitError("Too Many Requests"),
            "recovered_value",
        ]
    )
    mock_sleep = MagicMock()
    mock_rng = MagicMock(side_effect=lambda low, high: high)

    result = execute_with_retry(
        fn=fn,
        max_retries=2,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=2.0,
        sleep_fn=mock_sleep,
        rng=mock_rng,
    )

    assert result == "recovered_value"
    assert fn.call_count == 2
    assert mock_sleep.call_count == 1
    # First retry attempt delay: min(2.0, 0.5 * 2^0) = 0.5
    assert mock_sleep.call_args[0][0] == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Exhaustion After Configured Retries
# ---------------------------------------------------------------------------

def test_exhaustion_after_configured_retries():
    """Verify retry semantics: max_retries=2 means exactly 3 total attempts before failing."""
    error = TimeoutError("Request to model provider timed out")
    fn = MagicMock(side_effect=error)
    mock_sleep = MagicMock()

    with pytest.raises(TimeoutError) as exc_info:
        execute_with_retry(
            fn=fn,
            max_retries=2,
            sleep_fn=mock_sleep,
        )

    assert exc_info.value is error
    # 1 initial attempt + 2 retries = 3 total attempts
    assert fn.call_count == 3
    # 2 backoff sleeps between the 3 attempts
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# 4. No Retry for Authentication Failures
# ---------------------------------------------------------------------------

def test_no_retry_for_authentication_failures():
    """Verify terminal authentication errors fail immediately without retry."""
    fn = MagicMock(side_effect=SyntheticAuthenticationError("Authentication failure"))
    mock_sleep = MagicMock()

    with pytest.raises(SyntheticAuthenticationError):
        execute_with_retry(
            fn=fn,
            max_retries=3,
            sleep_fn=mock_sleep,
        )

    assert fn.call_count == 1
    assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 5. No Retry for Invalid Request and Validation Errors
# ---------------------------------------------------------------------------

def test_no_retry_for_invalid_request():
    """Verify invalid request errors (HTTP 400) fail immediately without retry."""
    fn = MagicMock(side_effect=SyntheticBadRequestError("Malformed parameters"))
    mock_sleep = MagicMock()

    with pytest.raises(SyntheticBadRequestError):
        execute_with_retry(
            fn=fn,
            max_retries=2,
            sleep_fn=mock_sleep,
        )

    assert fn.call_count == 1
    assert mock_sleep.call_count == 0


def test_no_retry_for_validation_errors():
    """Verify ValueError and domain validation errors fail immediately without retry."""
    fn = MagicMock(side_effect=ValueError("Invalid portfolio holding quantity"))
    mock_sleep = MagicMock()

    with pytest.raises(ValueError):
        execute_with_retry(
            fn=fn,
            max_retries=2,
            sleep_fn=mock_sleep,
        )

    assert fn.call_count == 1
    assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 6. Unknown Exceptions Fail Without Retry
# ---------------------------------------------------------------------------

def test_unknown_exceptions_fail_without_retry():
    """Verify unknown unclassified errors are NOT automatically treated as transient."""
    fn = MagicMock(side_effect=RuntimeError("Unexpected unclassified internal fault"))
    mock_sleep = MagicMock()

    with pytest.raises(RuntimeError):
        execute_with_retry(
            fn=fn,
            max_retries=2,
            sleep_fn=mock_sleep,
        )

    assert fn.call_count == 1
    assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# 7. Exponential Backoff Bounds and Jitter
# ---------------------------------------------------------------------------

def test_exponential_backoff_bounds_and_jitter():
    """Verify backoff calculation follows exponential scaling, ceiling capping, and jitter."""
    # Deterministic ceiling via mock RNG returning upper bound
    high_rng = lambda low, high: high

    # Attempt 1: min(2.0, 0.5 * 2^0) = 0.5
    d1 = calculate_backoff(1, initial_backoff=0.5, max_backoff=2.0, rng=high_rng)
    assert d1 == pytest.approx(0.5, abs=1e-4)

    # Attempt 2: min(2.0, 0.5 * 2^1) = 1.0
    d2 = calculate_backoff(2, initial_backoff=0.5, max_backoff=2.0, rng=high_rng)
    assert d2 == pytest.approx(1.0, abs=1e-4)

    # Attempt 3: min(2.0, 0.5 * 2^2) = 2.0 (reaches max_backoff)
    d3 = calculate_backoff(3, initial_backoff=0.5, max_backoff=2.0, rng=high_rng)
    assert d3 == pytest.approx(2.0, abs=1e-4)

    # Attempt 4: capped at max_backoff
    d4 = calculate_backoff(4, initial_backoff=0.5, max_backoff=2.0, rng=high_rng)
    assert d4 == pytest.approx(2.0, abs=1e-4)

    # Lower jitter bound: base * (1 - 0.25) = 0.5 * 0.75 = 0.375
    low_rng = lambda low, high: low
    d1_low = calculate_backoff(1, initial_backoff=0.5, max_backoff=2.0, jitter_factor=0.25, rng=low_rng)
    assert d1_low == pytest.approx(0.375, abs=1e-4)


# ---------------------------------------------------------------------------
# 8. Error Classification Unit Tests
# ---------------------------------------------------------------------------

def test_is_retryable_error_classification():
    """Verify classification logic across error categories."""
    # Retryable:
    assert is_retryable_error(TimeoutError("Connection timed out")) is True
    assert is_retryable_error(ConnectionError("Connection reset by peer")) is True
    assert is_retryable_error(SyntheticRateLimitError()) is True
    assert is_retryable_error(SyntheticOpenAIError("Internal server error", status_code=500)) is True
    assert is_retryable_error(SyntheticOpenAIError("Gateway timeout", status_code=504)) is True

    # Terminal / Non-retryable:
    assert is_retryable_error(SyntheticAuthenticationError()) is False
    assert is_retryable_error(SyntheticBadRequestError()) is False
    assert is_retryable_error(SyntheticOpenAIError("Context length exceeded", status_code=400)) is False
    assert is_retryable_error(ValueError("Bad value")) is False
    assert is_retryable_error(TypeError("Type error")) is False
    assert is_retryable_error(RuntimeError("Unknown random error")) is False


# ---------------------------------------------------------------------------
# 9. Invalid Configuration Values Rejected
# ---------------------------------------------------------------------------

def test_invalid_configuration_values_rejected():
    """Verify Settings rejects invalid retry settings."""
    # Negative max_retries
    with pytest.raises(ValidationError):
        Settings(max_retries=-1)

    # Zero or negative initial backoff
    with pytest.raises(ValidationError):
        Settings(retry_initial_backoff_seconds=0.0)

    # Max backoff smaller than initial backoff
    with pytest.raises(ValidationError):
        Settings(
            retry_initial_backoff_seconds=5.0,
            retry_max_backoff_seconds=2.0,
        )
