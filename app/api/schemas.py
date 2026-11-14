"""Pydantic API request/response schemas for auth, brokers, portfolios, and snapshots."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.integrations.brokers.models import SnapshotStatus, SupportedBroker


class ZerodhaConnectRequest(BaseModel):
    """Optional request parameters for initiating Zerodha Kite Connect login."""

    model_config = ConfigDict(extra="forbid")

    redirect_uri: str | None = Field(default=None, max_length=512)


class PortfolioCreateRequest(BaseModel):
    """Create a user-owned logical portfolio connected to a brokerage provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    broker_name: SupportedBroker = "zerodha"
    broker_connection_id: str | None = Field(default=None, max_length=64)
    currency: str = Field(default="INR", min_length=3, max_length=8)


class PortfolioRefreshRequest(BaseModel):
    """Manual portfolio refresh parameters (no polling/background execution)."""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    partial_policy: Literal["fail", "mark_partial"] = "mark_partial"


class PortfolioSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_id: str
    name: str
    broker_name: str
    broker_connection_id: str | None = None
    currency: str
    created_at: datetime
    updated_at: datetime


class SnapshotHoldingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holding_id: str
    symbol: str
    exchange: str
    instrument_identifier: str
    isin: str | None = None
    quantity: Decimal
    average_purchase_price: Decimal
    last_price: Decimal | None = None
    currency: str
    price_source: str | None = None
    valuation_available: bool


class SnapshotPositionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_id: str
    symbol: str
    exchange: str
    instrument_identifier: str
    product: str
    quantity: Decimal
    average_price: Decimal
    last_price: Decimal | None = None
    pnl: Decimal | None = None
    unrealised: Decimal | None = None
    realised: Decimal | None = None
    currency: str


class PortfolioSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    portfolio_id: str
    broker_name: str
    broker_account_id: str | None = None
    fetched_at: datetime
    status: SnapshotStatus
    idempotency_key: str | None = None
    cash_balance: Decimal | None = None
    currency: str
    valuation_available: bool
    holdings_count: int
    positions_count: int = 0
    limitations: list[str] = Field(default_factory=list)
    malformed_records: list[dict[str, Any]] = Field(default_factory=list)
    error_detail: str | None = None
    created_at: datetime
    holdings: list[SnapshotHoldingResponse] = Field(default_factory=list)
    positions: list[SnapshotPositionResponse] = Field(default_factory=list)


class LatestPortfolioResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: PortfolioSummaryResponse
    latest_snapshot: PortfolioSnapshotResponse | None = None


class PortfolioRiskSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: dict[str, Any]
    risk_assessment: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)


class PortfolioAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)


class PortfolioAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    provenance: dict[str, Any]
    risk_assessment: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)
    final_response: str | None = None
    error: str | None = None

