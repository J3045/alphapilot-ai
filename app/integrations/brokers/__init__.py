"""Broker integration interfaces, exception hierarchy, and typed domain models."""

from app.integrations.brokers.base import BrokerAdapter, BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConfigurationError,
    BrokerConnectionRevokedError,
    BrokerError,
    BrokerPermissionError,
    BrokerRateLimitError,
    BrokerResponseValidationError,
    BrokerTimeoutError,
    BrokerTokenExpiredError,
    BrokerUnavailableError,
    BrokerUnsupportedOperationError,
)
from app.integrations.brokers.models import (
    BrokerAuthorizationRequest,
    BrokerBalance,
    BrokerConnectionStatus,
    BrokerConnectionSummary,
    BrokerHoldingItem,
    BrokerPositionItem,
    BrokerProfile,
    BrokerSessionToken,
    NormalizedHolding,
    NormalizedPortfolioResult,
    SnapshotStatus,
    SupportedBroker,
)

__all__ = [
    "BrokerAdapter",
    "BrokerRegistry",
    "BrokerAuthorizationRequest",
    "BrokerBalance",
    "BrokerConnectionStatus",
    "BrokerConnectionSummary",
    "BrokerHoldingItem",
    "BrokerPositionItem",
    "BrokerProfile",
    "BrokerSessionToken",
    "NormalizedHolding",
    "NormalizedPortfolioResult",
    "SnapshotStatus",
    "SupportedBroker",
    "BrokerError",
    "BrokerConfigurationError",
    "BrokerAuthenticationError",
    "BrokerTokenExpiredError",
    "BrokerConnectionRevokedError",
    "BrokerPermissionError",
    "BrokerRateLimitError",
    "BrokerResponseValidationError",
    "BrokerTimeoutError",
    "BrokerUnavailableError",
    "BrokerUnsupportedOperationError",
]

