from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage

from app.agents.portfolio_graph import run_llm_portfolio_workflow


def sample_valid_tool_args() -> dict:
    return {
        "portfolio": {
            "portfolio_id": "DEMO-001",
            "cash": 2000.0,
            "holdings": [
                {"symbol": "ALFA", "quantity": 10.0, "price": 100.0},
                {"symbol": "BETA", "quantity": 5.0, "price": 200.0},
            ],
        },
        "max_single_holding_pct": 40.0,
        "min_cash_pct": 5.0,
    }


def test_approved_tool_call_executes_and_synthesizes_response():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "assess_portfolio_risk",
                    "args": sample_valid_tool_args(),
                    "id": "call_abc123",
                }
            ],
        ),
        AIMessage(content="Portfolio DEMO-001 passed risk assessment with a total value of $4,000."),
    ]

    result = run_llm_portfolio_workflow(
        user_query="Please assess risk for DEMO-001",
        llm_client=mock_llm,
    )

    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["assessment_result"] is not None
    assert result["assessment_result"]["portfolio_id"] == "DEMO-001"
    assert result["assessment_result"]["total_value"] == 4000.0
    assert result["assessment_result"]["passed"] is True
    assert result["final_response"] == "Portfolio DEMO-001 passed risk assessment with a total value of $4,000."
    assert mock_llm.invoke.call_count == 2


def test_malformed_tool_arguments_rejected_with_failed_status():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assess_portfolio_risk",
                "args": {
                    "portfolio": {
                        "portfolio_id": "DEMO-INVALID",
                        "cash": -500.0,  # Negative cash is invalid
                        "holdings": [],
                    }
                },
                "id": "call_malformed",
            }
        ],
    )

    result = run_llm_portfolio_workflow(
        user_query="Assess invalid portfolio",
        llm_client=mock_llm,
    )

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] is not None
    assert "Controlled tool execution rejected: malformed tool arguments" in result["error"]
    assert mock_llm.invoke.call_count == 1


def test_unapproved_tool_name_rejected_with_failed_status():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "execute_arbitrary_shell",
                "args": {"cmd": "dir"},
                "id": "call_unapproved",
            }
        ],
    )

    result = run_llm_portfolio_workflow(
        user_query="Run system command",
        llm_client=mock_llm,
    )

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] == "Controlled tool execution rejected: unapproved tool 'execute_arbitrary_shell'."
    assert mock_llm.invoke.call_count == 1


def test_plain_text_response_without_tool_call_completes():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="I am AlphaPilot AI. How can I assist you with your investment portfolio today?",
    )

    result = run_llm_portfolio_workflow(
        user_query="Hello who are you?",
        llm_client=mock_llm,
    )

    assert result["status"] == "completed"
    assert result["assessment_result"] is None
    assert result["error"] is None
    assert result["final_response"] == "I am AlphaPilot AI. How can I assist you with your investment portfolio today?"
    assert mock_llm.invoke.call_count == 1


def test_provider_authentication_error_handled_safely():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("AuthenticationError: Invalid API key sk-secret-123456789")

    result = run_llm_portfolio_workflow(
        user_query="Assess portfolio",
        llm_client=mock_llm,
    )

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] is not None
    # Must NOT leak secret key in error output
    assert "sk-secret-123456789" not in result["error"]
    assert "Authentication failure with model provider" in result["error"]


def test_provider_timeout_error_handled_safely():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = TimeoutError("Connection to api.openai.com timed out after 30s")

    result = run_llm_portfolio_workflow(
        user_query="Assess portfolio",
        llm_client=mock_llm,
    )

    assert result["status"] == "failed"
    assert result["assessment_result"] is None
    assert result["error"] == "LLM provider error: Request to model provider timed out."


def test_iteration_limit_enforced_on_repeated_tool_requests():
    mock_llm = MagicMock()
    # The model repeatedly outputs tool calls without ever reaching text synthesis
    mock_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "assess_portfolio_risk",
                "args": sample_valid_tool_args(),
                "id": "call_loop",
            }
        ],
    )

    result = run_llm_portfolio_workflow(
        user_query="Assess portfolio in a loop",
        llm_client=mock_llm,
        max_iterations=2,
    )

    assert result["status"] == "failed"
    assert "Execution limit reached: model reached maximum invocations limit (2)" in result["error"]
    # Model should have been called exactly 2 times before the boundary terminated execution
    assert mock_llm.invoke.call_count == 2
