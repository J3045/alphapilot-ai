import pytest
# pyrefly: ignore[missing-import]
from pydantic import ValidationError

from app.models.portfolio import Holding, Portfolio
from app.services.risk_engine import assess_portfolio_risk


def sample_portfolio() -> Portfolio:
    return Portfolio(
        portfolio_id="DEMO-001",
        cash=2000,
        holdings=[
            Holding(symbol="ALFA", quantity=10, price=100),
            Holding(symbol="BETA", quantity=5, price=200),
        ],
    )


def test_portfolio_total_value():
    portfolio = sample_portfolio()

    assert portfolio.holdings_value == 2000
    assert portfolio.total_value == 4000


def test_risk_engine_detects_concentration():
    portfolio = Portfolio(
        portfolio_id="DEMO-002",
        cash=100,
        holdings=[
            Holding(symbol="ALFA", quantity=9, price=100),
            Holding(symbol="BETA", quantity=1, price=100),
        ],
    )

    result = assess_portfolio_risk(portfolio)

    assert result["passed"] is False
    assert any("ALFA" in item for item in result["violations"])


def test_holding_rejects_negative_quantity():
    with pytest.raises(ValidationError):
        Holding(symbol="ALFA", quantity=-1, price=100)
