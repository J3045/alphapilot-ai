"""Sanitized exception hierarchy for brokerage integrations."""

from typing import Optional

from app.logging_config import redact_sensitive_text


class BrokerError(Exception):
    """Base exception for broker adapter and ingestion errors with scrubbed messages."""

    def __init__(
        self,
        message: str,
        *,
        broker_name: Optional[str] = None,
        error_type: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        cleaned = redact_sensitive_text(message)
        super().__init__(cleaned)
        self.message = cleaned
        self.broker_name = broker_name
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class BrokerConfigurationError(BrokerError):
    """Raised when required broker credentials, redirect URI, or keys are missing/invalid."""


class BrokerAuthenticationError(BrokerError):
    """Raised when OAuth handshake, checksum verification, or login exchange fails."""


class BrokerTokenExpiredError(BrokerAuthenticationError):
    """Raised when a broker session access token is expired or invalid (e.g. Kite TokenException)."""


class BrokerConnectionRevokedError(BrokerAuthenticationError):
    """Raised when the user's broker connection has been revoked and cannot be queried."""


class BrokerPermissionError(BrokerError):
    """Raised when the broker denies access due to missing permissions or blocked endpoint."""


class BrokerRateLimitError(BrokerError):
    """Raised when broker API rate limits are exceeded (HTTP 429 / TooManyRequests)."""


class BrokerTimeoutError(BrokerError):
    """Raised when a broker HTTP request exceeds its configured timeout budget."""


class BrokerUnavailableError(BrokerError):
    """Raised when a broker API or network gateway returns 5xx or is unreachable."""


class BrokerResponseValidationError(BrokerError):
    """Raised when a broker returns malformed, non-JSON, or structurally invalid data."""


class BrokerUnsupportedOperationError(BrokerError):
    """Raised when an unsupported broker name or prohibited operation (e.g. trading) is requested."""

