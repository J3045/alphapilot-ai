import json
import pytest
from pydantic import ValidationError

from app.models.portfolio import Holding, Portfolio
from app.tools.portfolio_tools import (
    PortfolioAssessmentInput,
    PortfolioAssessmentResult,
    assess_portfolio_risk_tool,
    execute_tool,
)


def sample_portfolio_data() -> dict:
    return {
        "portfolio_id": "DEMO-001",
        "cash": 2000.0,
        "holdings": [
            {"symbol": "ALFA", "quantity": 10.0, "price": 100.0},
            {"symbol": "BETA", "quantity": 5.0, "price": 200.0},
        ],
    }


def test_tool_valid_portfolio_returns_structured_result():
    payload = {"portfolio": sample_portfolio_data()}
    result = assess_portfolio_risk_tool(payload)

    assert isinstance(result, PortfolioAssessmentResult)
    assert result.portfolio_id == "DEMO-001"
    assert result.total_value == 4000.0
    assert result.cash_pct == 50.0
    assert result.passed is True
    assert result.violations == []
    assert len(result.concentrations) == 2


def test_tool_detects_overconcentration_violation():
    payload = {
        "portfolio": {
            "portfolio_id": "DEMO-002",
            "cash": 100.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 9.0, "price": 100.0},
                {"symbol": "BETA", "quantity": 1.0, "price": 100.0},
            ],
        }
    }
    result = assess_portfolio_risk_tool(payload)

    assert result.passed is False
    assert any("ALFA exceeds the 40.0% single-holding limit" in v for v in result.violations)


def test_tool_rejects_invalid_holding_quantity():
    payload = {
        "portfolio": {
            "portfolio_id": "DEMO-001",
            "cash": 1000.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": -5.0, "price": 100.0}
            ],
        }
    }
    with pytest.raises(ValidationError):
        assess_portfolio_risk_tool(payload)


def test_tool_rejects_negative_cash():
    payload = {
        "portfolio": {
            "portfolio_id": "DEMO-001",
            "cash": -50.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 5.0, "price": 100.0}
            ],
        }
    }
    with pytest.raises(ValidationError):
        assess_portfolio_risk_tool(payload)


def test_tool_rejects_unexpected_fields():
    payload = {
        "portfolio": sample_portfolio_data(),
        "unauthorized_field": "arbitrary_data",
    }
    with pytest.raises(ValidationError):
        assess_portfolio_risk_tool(payload)


def test_tool_output_is_json_serializable():
    payload = {"portfolio": sample_portfolio_data()}
    result = assess_portfolio_risk_tool(payload)

    raw_dict = result.model_dump()
    serialized = json.dumps(raw_dict)
    deserialized = json.loads(serialized)

    assert deserialized["portfolio_id"] == "DEMO-001"
    assert deserialized["total_value"] == 4000.0
    assert deserialized["passed"] is True
    assert isinstance(deserialized["concentrations"], list)


def test_execute_tool_dispatcher_success():
    payload = {"portfolio": sample_portfolio_data()}
    output = execute_tool("assess_portfolio_risk", payload)

    assert isinstance(output, dict)
    assert output["portfolio_id"] == "DEMO-001"
    assert output["passed"] is True


def test_execute_tool_rejects_arbitrary_unauthorized_functions():
    with pytest.raises(ValueError, match="is not permitted or does not exist"):
        execute_tool("os.system", {"command": "dir"})

    with pytest.raises(ValueError, match="is not permitted or does not exist"):
        execute_tool("eval", {"expression": "1+1"})
