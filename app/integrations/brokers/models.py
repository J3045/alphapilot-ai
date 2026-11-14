"""Typed domain models for broker connections, normalized holdings, and portfolio snapshots."""

from datetime import datetime
from decimal import Decimal
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.config import settings
from app.models.portfolio import Holding, Portfolio, is_valid_ticker_symbol

SupportedBroker = Literal["zerodha"]
BrokerConnectionStatus = Literal["pending", "active", "expired", "revoked", "failed"]
SnapshotStatus = Literal["completed", "partial", "failed"]
PriceSource = Literal["broker_ltp", "unavailable"]

_ISIN_REGEX = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")
POSITIONS_EXCLUSION_LIMITATION = (
    "Open/intraday positions fetched in this snapshot are stored separately and excluded from "
    "long-only demat holdings concentration metrics."
)


def _validate_finite_decimal(val: Any, field_name: str, *, allow_zero: bool = True) -> Decimal:
    """Parse finite numeric input into Decimal and reject NaN/Inf/negative values."""
    if isinstance(val, bool) or val is None:
        raise ValueError(f"{field_name} must be a finite numeric value.")
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError(f"{field_name} must be a finite number.")
    try:
        dec = Decimal(str(val))
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid decimal number.") from exc
    if not dec.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    if allow_zero:
        if dec < Decimal("0"):
            raise ValueError(f"{field_name} must be greater than or equal to 0.")
    else:
        if dec <= Decimal("0"):
            raise ValueError(f"{field_name} must be strictly greater than 0.")
    return dec


def _validate_signed_finite_decimal(val: Any, field_name: str) -> Decimal:
    """Parse finite numeric input into Decimal allowing negative values (e.g. P&L or net short)."""
    if isinstance(val, bool) or val is None:
        raise ValueError(f"{field_name} must be a finite numeric value.")
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError(f"{field_name} must be a finite number.")
    try:
        dec = Decimal(str(val))
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid decimal number.") from exc
    if not dec.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    return dec


class BrokerAuthorizationRequest(BaseModel):
    """Response returned when initiating a broker OAuth/login flow."""

    model_config = ConfigDict(extra="forbid")

    broker_name: SupportedBroker
    authorization_url: str
    state_expires_at: datetime


class BrokerSessionToken(BaseModel):
    """Internal broker session token bundle returned by session exchange.

    `access_token` uses SecretStr so repr/logs never emit plaintext credentials.
    """

    model_config = ConfigDict(extra="forbid")

    broker_name: SupportedBroker
    access_token: SecretStr
    broker_account_id: str = Field(min_length=1, max_length=128)
    broker_user_name: str | None = None
    email: str | None = None
    exchanges: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    login_time: datetime | None = None
    expires_at: datetime | None = None


class BrokerProfile(BaseModel):
    """Sanitized broker user profile metadata."""

    model_config = ConfigDict(extra="forbid")

    broker_name: SupportedBroker
    broker_account_id: str = Field(min_length=1, max_length=128)
    user_name: str | None = None
    email: str | None = None
    user_type: str | None = None
    exchanges: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)


class BrokerHoldingItem(BaseModel):
    """Validated holding item returned by a broker adapter."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    exchange: str = Field(default="UNKNOWN", min_length=1, max_length=32)
    instrument_identifier: str | None = Field(default=None, max_length=128)
    isin: str | None = Field(default=None, max_length=32)
    product: str | None = Field(default=None, max_length=32)
    quantity: Decimal
    t1_quantity: Decimal = Field(default=Decimal("0"))
    average_price: Decimal
    last_price: Decimal | None = None
    pnl: Decimal | None = None
    currency: str = Field(default="INR", min_length=3, max_length=8)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol", "exchange")
    @classmethod
    def _clean_upper(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Identifier field cannot be empty.")
        return cleaned

    @field_validator("quantity", "t1_quantity", "average_price", mode="before")
    @classmethod
    def _non_negative_decimal(cls, value: Any, info) -> Decimal:
        return _validate_finite_decimal(value, info.field_name, allow_zero=True)

    @field_validator("last_price", mode="before")
    @classmethod
    def _optional_non_negative_price(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return _validate_finite_decimal(value, "last_price", allow_zero=True)

    @field_validator("pnl", mode="before")
    @classmethod
    def _optional_signed_pnl(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return _validate_signed_finite_decimal(value, "pnl")


class BrokerPositionItem(BaseModel):
    """Validated position item returned by a broker adapter."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    exchange: str = Field(default="UNKNOWN", min_length=1, max_length=32)
    instrument_identifier: str | None = Field(default=None, max_length=128)
    product: str | None = Field(default=None, max_length=32)
    quantity: Decimal
    average_price: Decimal
    last_price: Decimal | None = None
    pnl: Decimal | None = None
    unrealised: Decimal | None = None
    realised: Decimal | None = None
    currency: str = Field(default="INR", min_length=3, max_length=8)

    @field_validator("symbol", "exchange")
    @classmethod
    def _clean_upper(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Identifier field cannot be empty.")
        return cleaned

    @field_validator("quantity", "pnl", "unrealised", "realised", mode="before")
    @classmethod
    def _signed_optional_decimal(cls, value: Any, info) -> Decimal | None:
        if value is None:
            return None
        return _validate_signed_finite_decimal(value, info.field_name)

    @field_validator("average_price", mode="before")
    @classmethod
    def _non_negative_avg(cls, value: Any) -> Decimal:
        return _validate_finite_decimal(value, "average_price", allow_zero=True)

    @field_validator("last_price", mode="before")
    @classmethod
    def _optional_price(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return _validate_finite_decimal(value, "last_price", allow_zero=True)


class BrokerBalance(BaseModel):
    """Validated margin / cash balance returned by a broker adapter."""

    model_config = ConfigDict(extra="forbid")

    available_cash: Decimal
    net_balance: Decimal | None = None
    collateral: Decimal = Field(default=Decimal("0"))
    utilised_debits: Decimal = Field(default=Decimal("0"))
    currency: str = Field(default="INR", min_length=3, max_length=8)
    segment: str = Field(default="equity", max_length=32)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("available_cash", "collateral", "utilised_debits", mode="before")
    @classmethod
    def _non_negative_balance(cls, value: Any, info) -> Decimal:
        return _validate_finite_decimal(value, info.field_name, allow_zero=True)

    @field_validator("net_balance", mode="before")
    @classmethod
    def _optional_signed_net(cls, value: Any) -> Decimal | None:
        if value is None:
            return None
        return _validate_signed_finite_decimal(value, "net_balance")


class MalformedBrokerRecord(BaseModel):
    """Sanitized diagnostic for an individual broker record that failed validation."""

    model_config = ConfigDict(extra="forbid")

    record_type: Literal["holding", "position"] = "holding"
    index: int = Field(ge=0)
    symbol: str = Field(default="UNKNOWN", max_length=64)
    exchange: str | None = Field(default=None, max_length=32)
    reason: str = Field(max_length=512)


class BrokerHoldingsBatch(BaseModel):
    """Typed batch returned by broker adapters separating valid rows from malformed rows."""

    model_config = ConfigDict(extra="forbid")

    items: list[BrokerHoldingItem] = Field(default_factory=list)
    malformed_records: list[MalformedBrokerRecord] = Field(default_factory=list)


class BrokerPositionsBatch(BaseModel):
    """Typed batch returned by broker adapters separating valid position rows from malformed rows."""

    model_config = ConfigDict(extra="forbid")

    items: list[BrokerPositionItem] = Field(default_factory=list)
    malformed_records: list[MalformedBrokerRecord] = Field(default_factory=list)


class BrokerConnectionSummary(BaseModel):
    """Public/API-safe metadata view of a user's brokerage connection (zero secret fields)."""

    model_config = ConfigDict(extra="forbid")

    connection_id: str
    broker_name: SupportedBroker
    broker_account_id: str | None = None
    status: BrokerConnectionStatus
    token_expires_at: datetime | None = None
    last_authenticated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class NormalizedHolding(BaseModel):
    """Canonical normalized holding used for persistence and portfolio analytics."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=64)
    exchange: str = Field(min_length=1, max_length=32)
    instrument_identifier: str | None = Field(default=None, max_length=128)
    isin: str | None = Field(default=None, max_length=32)
    quantity: Decimal
    settled_quantity: Decimal
    t1_quantity: Decimal = Field(default=Decimal("0"))
    average_purchase_price: Decimal
    last_price: Decimal | None = None
    currency: str = Field(default="INR", min_length=3, max_length=8)
    price_source: PriceSource
    valuation_available: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedPortfolioResult(BaseModel):
    """Complete output of broker portfolio ingestion and canonical normalization."""

    model_config = ConfigDict(extra="forbid")

    broker_name: SupportedBroker
    broker_account_id: str
    currency: str = "INR"
    cash_balance: Decimal | None = None
    holdings: list[NormalizedHolding] = Field(default_factory=list)
    positions: list[BrokerPositionItem] = Field(default_factory=list)
    status: SnapshotStatus
    valuation_available: bool
    malformed_records: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    fetched_at: datetime
    profile: BrokerProfile | None = None

    def to_domain_portfolio(self, portfolio_id: str) -> tuple[Portfolio | None, list[str]]:
        """Convert normalized portfolio to AlphaPilot core Portfolio when full valuation is valid.

        Never substitutes `average_purchase_price` for market `last_price`, and never drops
        individual holdings silently (which would distort single-stock concentration metrics).
        Aggregates multi-exchange listings sharing a verified canonical ISIN; refuses to merge
        ambiguous multi-exchange entries lacking ISIN identity and emits an explicit limitation.
        Returns `(Portfolio, limitations)` or `(None, blocking_reasons)`.
        """
        reasons: list[str] = list(self.limitations)

        if self.positions and POSITIONS_EXCLUSION_LIMITATION not in reasons:
            reasons.append(POSITIONS_EXCLUSION_LIMITATION)

        if self.status != "completed":
            reasons.append(
                f"Snapshot status is '{self.status}', so portfolio concentration risk cannot be "
                "computed on incomplete or partial broker data."
            )
            return None, reasons

        if self.cash_balance is None:
            reasons.append("Broker cash balance is unavailable; cannot compute cash allocation percentage.")
            return None, reasons

        positive_holdings: list[tuple[int, NormalizedHolding, str | None]] = []
        symbol_isins: dict[str, set[str]] = {}
        symbol_unverified_count: dict[str, int] = {}

        for idx, item in enumerate(self.holdings):
            if item.quantity <= Decimal("0"):
                continue
            if not item.valuation_available or item.last_price is None or item.last_price <= Decimal("0"):
                reasons.append(
                    f"Holding '{item.symbol}' ({item.exchange}) lacks a positive current market price "
                    "(average purchase cost is intentionally not treated as market value)."
                )
                return None, reasons

            max_allowed_len = min(settings.max_symbol_length, 32)
            if len(item.symbol.strip()) > settings.max_symbol_length or not is_valid_ticker_symbol(
                item.symbol, max_length=max_allowed_len
            ):
                reasons.append(
                    f"Holding symbol '{item.symbol}' exceeds core domain ticker validation bounds "
                    f"(max {max_allowed_len} chars); refusing partial concentration calculation."
                )
                return None, reasons

            raw_isin = (item.isin or "").strip().upper()
            valid_isin = raw_isin if _ISIN_REGEX.match(raw_isin) else None
            clean_symbol = item.symbol.strip().upper()

            if valid_isin:
                symbol_isins.setdefault(clean_symbol, set()).add(valid_isin)
            else:
                symbol_unverified_count[clean_symbol] = symbol_unverified_count.get(clean_symbol, 0) + 1

            positive_holdings.append((idx, item, valid_isin))

        for sym, unverified_cnt in symbol_unverified_count.items():
            if unverified_cnt > 1 or (unverified_cnt >= 1 and len(symbol_isins.get(sym, set())) >= 1):
                reasons.append(
                    f"Holding '{sym}' appears across multiple entries/exchanges without verifiable ISIN "
                    "identity; rows were not merged for concentration calculation."
                )

        isin_buckets: dict[str, list[NormalizedHolding]] = {}
        ordered_keys: list[tuple[str, Any]] = []
        seen_isins: set[str] = set()

        for idx, item, valid_isin in positive_holdings:
            if valid_isin is not None:
                isin_buckets.setdefault(valid_isin, []).append(item)
                if valid_isin not in seen_isins:
                    seen_isins.add(valid_isin)
                    ordered_keys.append(("isin", valid_isin))
            else:
                ordered_keys.append(("unverified", (idx, item)))

        domain_holdings: list[Holding] = []
        for kind, payload in ordered_keys:
            try:
                if kind == "isin":
                    valid_isin = payload
                    rows = isin_buckets[valid_isin]
                    total_qty = sum((r.quantity for r in rows), Decimal("0"))
                    total_val = sum((r.quantity * r.last_price for r in rows if r.last_price is not None), Decimal("0"))
                    weighted_price = total_val / total_qty
                    preferred = next((r for r in rows if r.exchange.upper() == "NSE"), rows[0])
                    domain_holdings.append(
                        Holding(
                            symbol=preferred.symbol,
                            quantity=float(total_qty),
                            price=float(weighted_price),
                            instrument_id=f"isin:{valid_isin}",
                        )
                    )
                else:
                    idx, item = payload
                    ident = (item.instrument_identifier or item.symbol).strip().upper()
                    domain_holdings.append(
                        Holding(
                            symbol=item.symbol,
                            quantity=float(item.quantity),
                            price=float(item.last_price),
                            instrument_id=f"unverified:{item.exchange.upper()}:{ident}:{idx}",
                        )
                    )
            except ValidationError:
                sym = payload if isinstance(payload, str) else payload[1].symbol
                reasons.append(
                    f"Holding '{sym}' failed strict core Portfolio validation rules."
                )
                return None, reasons

        try:
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                cash=float(self.cash_balance),
                holdings=domain_holdings,
            )
        except ValidationError:
            reasons.append("Normalized portfolio exceeds core Portfolio schema boundaries.")
            return None, reasons

        return portfolio, reasons

