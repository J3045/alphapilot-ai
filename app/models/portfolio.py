# pyrefly: ignore[missing-import]
import math
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator


TICKER_SYMBOL_PATTERN = re.compile(
    r"^[A-Z0-9][A-Z0-9.\-_]{0,31}(?:&[A-Z0-9.\-_]{1,31})*$"
)


def is_valid_ticker_symbol(symbol: str, max_length: int = 32) -> bool:
    """Validate an uppercase ticker symbol supporting legitimate Indian/global symbols (e.g., M&M, NIFTY100BEES)."""
    stripped = symbol.strip().upper()
    if not stripped or len(stripped) > max_length:
        return False
    return bool(TICKER_SYMBOL_PATTERN.match(stripped))


class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(
        min_length=1,
        max_length=32,
        description=(
            "Ticker symbol for the asset (1-32 uppercase ticker characters, supporting ., -, _, &)."
        ),
    )
    quantity: float = Field(
        gt=0,
        le=1e9,
        allow_inf_nan=False,
        description="Positive finite quantity of shares or units held.",
    )
    price: float = Field(
        gt=0,
        le=1e9,
        allow_inf_nan=False,
        description="Positive finite unit price in quote currency.",
    )
    instrument_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional canonical instrument identity (e.g., 'isin:INE002A01018') for aggregating duplicate exchange listings."
        ),
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        stripped = value.strip().upper()
        if not stripped:
            raise ValueError("symbol cannot be empty or whitespace-only.")
        if not is_valid_ticker_symbol(stripped, max_length=32):
            raise ValueError(
                f"Invalid symbol format: '{value}'. Must be 1-32 valid ticker characters."
            )
        return stripped

    @field_validator("instrument_id")
    @classmethod
    def validate_instrument_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("quantity", "price")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Financial numeric values must be finite numbers (not NaN or Inf).")
        return value

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_id: str = Field(
        min_length=1,
        max_length=64,
        description="Unique identifier for the investment portfolio.",
    )
    cash: float = Field(
        ge=0,
        le=1e12,
        allow_inf_nan=False,
        description="Available uninvested cash balance (finite, non-negative).",
    )
    holdings: list[Holding] = Field(
        default_factory=list,
        max_length=500,
        description="List of individual asset holdings within the portfolio (max 500).",
    )

    @field_validator("portfolio_id")
    @classmethod
    def validate_portfolio_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("portfolio_id cannot be empty or whitespace-only.")
        return stripped

    @field_validator("cash")
    @classmethod
    def validate_finite_cash(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Cash value must be a finite non-negative number (not NaN or Inf).")
        return value

    @property
    def holdings_value(self) -> float:
        return sum(holding.market_value for holding in self.holdings)

    @property
    def total_value(self) -> float:
        return self.cash + self.holdings_value
