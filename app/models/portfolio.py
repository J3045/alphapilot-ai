# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field, ConfigDict


class Holding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=10)
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_id: str = Field(min_length=1)
    cash: float = Field(ge=0)
    holdings: list[Holding]

    @property
    def holdings_value(self) -> float:
        return sum(holding.market_value for holding in self.holdings)

    @property
    def total_value(self) -> float:
        return self.cash + self.holdings_value
