"""Broker-agnostic async adapter contract and registry."""

from abc import ABC, abstractmethod

from app.integrations.brokers.exceptions import BrokerUnsupportedOperationError
from app.integrations.brokers.models import (
    BrokerBalance,
    BrokerHoldingItem,
    BrokerHoldingsBatch,
    BrokerPositionItem,
    BrokerPositionsBatch,
    BrokerProfile,
    BrokerSessionToken,
    SupportedBroker,
)
from app.services.resilience import WorkflowDeadline


class BrokerAdapter(ABC):
    """Abstract asynchronous read-only brokerage adapter.

    Implementations handle broker-specific authentication protocols, HTTP headers,
    rate-limit error translation, and response schema parsing while returning
    standardized AlphaPilot broker domain models.
    """

    @property
    @abstractmethod
    def broker_name(self) -> SupportedBroker:
        """Canonical lowercase identifier for this brokerage provider."""

    @abstractmethod
    def build_authorization_url(self, state: str, redirect_uri: str | None = None) -> str:
        """Construct official user consent redirect URL binding signed CSRF state."""

    @abstractmethod
    async def exchange_request_token(self, request_token: str) -> BrokerSessionToken:
        """Exchange one-time authorization request token for a broker session token."""

    @abstractmethod
    async def revoke_session(self, access_token: str) -> None:
        """Invalidate the active session token with the broker provider when supported."""

    @abstractmethod
    async def get_profile(self, access_token: str) -> BrokerProfile:
        """Fetch authenticated account profile details (read-only)."""

    @abstractmethod
    async def get_holdings(self, access_token: str) -> list[BrokerHoldingItem]:
        """Fetch settled and T+1 equity holdings (read-only)."""

    @abstractmethod
    async def get_positions(self, access_token: str) -> list[BrokerPositionItem]:
        """Fetch current trading positions across day/net books (read-only)."""

    @abstractmethod
    async def get_balance(self, access_token: str) -> BrokerBalance:
        """Fetch available cash and margin balances (read-only)."""

    async def get_holdings_batch(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerHoldingsBatch:
        """Fetch equity holdings separated into valid rows and sanitized malformed diagnostics."""
        _ = deadline
        items = await self.get_holdings(access_token)
        return BrokerHoldingsBatch(items=items, malformed_records=[])

    async def get_positions_batch(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerPositionsBatch:
        """Fetch trading positions separated into valid rows and sanitized malformed diagnostics."""
        _ = deadline
        items = await self.get_positions(access_token)
        return BrokerPositionsBatch(items=items, malformed_records=[])

    async def aclose(self) -> None:
        """Release any underlying HTTP client resources cleanly."""


class BrokerRegistry:
    """Thread-safe registry mapping canonical broker names to read-only adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BrokerAdapter] = {}

    def register(self, adapter: BrokerAdapter) -> None:
        self._adapters[adapter.broker_name] = adapter

    def get(self, broker_name: str) -> BrokerAdapter:
        key = broker_name.strip().lower()
        adapter = self._adapters.get(key)
        if adapter is None:
            raise BrokerUnsupportedOperationError(
                f"Broker '{broker_name}' is not registered or supported.",
                broker_name=key,
            )
        return adapter

    def list_brokers(self) -> list[str]:
        return sorted(self._adapters.keys())

    async def aclose_all(self) -> None:
        for adapter in self._adapters.values():
            await adapter.aclose()

