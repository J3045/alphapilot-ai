import pytest
from app.agents.portfolio_graph import (
    build_portfolio_graph,
    run_portfolio_workflow,
)


def sample_valid_portfolio_payload() -> dict:
    return {
        "portfolio": {
            "portfolio_id": "DEMO-001",
            "cash": 2000.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 10.0, "price": 100.0},
                {"symbol": "BETA", "quantity": 5.0, "price": 200.0},
            ],
        }
    }


def sample_overconcentrated_portfolio_payload() -> dict:
    return {
        "portfolio": {
            "portfolio_id": "DEMO-002",
            "cash": 100.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 9.0, "price": 100.0},
                {"symbol": "BETA", "quantity": 1.0, "price": 100.0},
            ],
        }
    }


def test_valid_portfolio_reaches_completed_state():
    payload = sample_valid_portfolio_payload()
    result = run_portfolio_workflow(payload)

    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["assessment_result"] is not None
    assert result["assessment_result"]["portfolio_id"] == "DEMO-001"
    assert result["assessment_result"]["total_value"] == 4000.0
    assert result["assessment_result"]["passed"] is True
    assert result["assessment_result"]["violations"] == []


def test_overconcentrated_portfolio_completes_with_failed_risk_assessment():
    payload = sample_overconcentrated_portfolio_payload()
    result = run_portfolio_workflow(payload)

    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["assessment_result"] is not None
    assert result["assessment_result"]["portfolio_id"] == "DEMO-002"
    assert result["assessment_result"]["passed"] is False
    assert len(result["assessment_result"]["violations"]) > 0
    assert any("ALFA exceeds the 40.0% single-holding limit" in v for v in result["assessment_result"]["violations"])


def test_invalid_input_follows_failed_error_path():
    invalid_payload = {
        "portfolio": {
            "portfolio_id": "DEMO-003",
            "cash": -50.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 10.0, "price": 100.0}
            ],
        }
    }
    result = run_portfolio_workflow(invalid_payload)

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] is not None
    assert "Portfolio assessment rejected" in result["error"]


def test_unexpected_fields_follow_failed_error_path():
    payload_with_extra = {
        "portfolio": {
            "portfolio_id": "DEMO-004",
            "cash": 1000.0,
            "holdings": [],
        },
        "malicious_extra_field": "exploit_attempt",
    }
    result = run_portfolio_workflow(payload_with_extra)

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] is not None
    assert "Portfolio assessment rejected" in result["error"]


def test_missing_input_payload_fails_safely():
    graph = build_portfolio_graph()
    result = graph.invoke({})

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] == "Missing 'portfolio_input' in workflow state."


def test_graph_terminates_without_unbounded_loop():
    graph = build_portfolio_graph()
    payload = sample_valid_portfolio_payload()

    # LangGraph recursion limit test to verify single-step deterministic termination
    result = graph.invoke(
        {"portfolio_input": payload, "assessment_result": None, "error": None, "status": "pending"},
        config={"recursion_limit": 5},
    )

    assert result["status"] == "completed"
    assert result["assessment_result"] is not None
