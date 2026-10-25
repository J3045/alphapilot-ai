# pyrefly: ignore[missing-import]
"""Deterministic integration tests for shared resilience with RAG, agents, and telemetry."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.agents.portfolio_graph import make_call_model_node, run_llm_portfolio_workflow
from app.config import settings
from app.observability import clear_recorded_events, get_recorded_events
from app.services.rag_service import RagService
from app.services.retrieval_service import FakeEmbeddingClient
from tests.test_resilience import (
    SyntheticAuthenticationError,
    SyntheticBadRequestError,
    SyntheticRateLimitError,
)


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Ensure in-memory event sink is clean before and after every test."""
    clear_recorded_events()
    yield
    clear_recorded_events()


# ---------------------------------------------------------------------------
# 1. RAG Resilience Integration
# ---------------------------------------------------------------------------

def test_rag_transient_failure_followed_by_success():
    """Verify RAG answer generation retries a transient timeout and succeeds with single telemetry event."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        TimeoutError("Model provider request timed out"),
        AIMessage(
            content="According to [SYNTH-DOC-002], maintain a 5.0% cash reserve.",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        ),
    ]

    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
    )

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        result = service.answer_question("What is the cash reserve requirement?")

    assert result.status == "answered"
    assert "SYNTH-DOC-002" in result.answer
    assert mock_llm.invoke.call_count == 2
    assert mock_sleep.call_count == 1

    # Exactly ONE final telemetry event must be emitted for the completed operation
    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.operation == "rag_answer"
    assert event.outcome == "success"
    assert event.input_tokens == 120
    assert event.output_tokens == 30
    assert event.total_tokens == 150
    assert event.estimated_cost is not None


def test_rag_retry_exhaustion_returns_existing_failure_result():
    """Verify RAG retries until exhaustion (3 total attempts) and preserves error output and telemetry."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = TimeoutError("Connection timed out repeatedly")

    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
    )

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        result = service.answer_question("What is the cash reserve requirement?")

    assert result.status == "failed"
    assert result.error == "Request to model provider timed out."
    # 1 initial attempt + 2 retries = 3 total attempts
    assert mock_llm.invoke.call_count == 3
    assert mock_sleep.call_count == 2

    # Exactly ONE final failure telemetry event
    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.operation == "rag_answer"
    assert event.outcome == "timeout"
    assert event.error_category == "timeout"
    assert event.input_tokens is None


def test_rag_terminal_failure_without_retries():
    """Verify non-retryable authentication failure fails immediately without retry attempts."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = SyntheticAuthenticationError("Invalid API key")

    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
    )

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        result = service.answer_question("What is the cash reserve requirement?")

    assert result.status == "failed"
    assert "Authentication failure" in result.error
    assert mock_llm.invoke.call_count == 1
    assert mock_sleep.call_count == 0

    events = get_recorded_events()
    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].error_category == "authentication"


# ---------------------------------------------------------------------------
# 2. Portfolio Agent Resilience Integration
# ---------------------------------------------------------------------------

def test_agent_transient_failure_followed_by_success():
    """Verify portfolio agent recovers from transient connection drop without inflating graph iteration count."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        ConnectionError("Connection reset by peer"),
        AIMessage(content="Portfolio risk assessment completed successfully."),
    ]

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        state = run_llm_portfolio_workflow(
            user_query="Assess portfolio risk",
            llm_client=mock_llm,
        )

    assert state["status"] == "completed"
    assert state["final_response"] == "Portfolio risk assessment completed successfully."
    assert mock_llm.invoke.call_count == 2
    assert mock_sleep.call_count == 1
    # Critical: Graph iteration count must be 1, NOT 2
    assert state["iteration_count"] == 1

    events = get_recorded_events()
    assert len(events) == 1
    assert events[0].operation == "portfolio_agent_model"
    assert events[0].outcome == "success"


def test_agent_retry_exhaustion_handled_by_existing_graph_error_path():
    """Verify agent retry exhaustion returns existing failed state structure and single telemetry failure."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ConnectionError("Host unreachable")

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        state = run_llm_portfolio_workflow(
            user_query="Assess portfolio risk",
            llm_client=mock_llm,
        )

    assert state["status"] == "failed"
    assert "Connection failure" in state["error"]
    assert mock_llm.invoke.call_count == 3
    assert mock_sleep.call_count == 2
    assert state["iteration_count"] == 1

    events = get_recorded_events()
    assert len(events) == 1
    assert events[0].operation == "portfolio_agent_model"
    assert events[0].outcome == "failure"
    assert events[0].error_category == "connection"


def test_agent_iteration_limits_remain_correct_after_internal_retries():
    """Verify multi-step workflow respects max_iterations even when internal retries occur."""
    mock_llm = MagicMock()
    tool_args = {
        "portfolio": {
            "portfolio_id": "P-TEST",
            "cash": 5000.0,
            "holdings": [{"symbol": "ALFA", "quantity": 10.0, "price": 100.0}],
        },
        "max_single_holding_pct": 50.0,
        "min_cash_pct": 10.0,
    }

    mock_llm.invoke.side_effect = [
        # Call 1: Transient 429 error
        SyntheticRateLimitError("Rate limit exceeded"),
        # Call 1 (retry): returns tool call
        AIMessage(
            content="",
            tool_calls=[{"name": "assess_portfolio_risk", "args": tool_args, "id": "call_1"}],
        ),
        # Call 2: Final synthesis
        AIMessage(content="Portfolio passed all compliance constraints."),
    ]

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        state = run_llm_portfolio_workflow(
            user_query="Assess portfolio P-TEST",
            llm_client=mock_llm,
            max_iterations=2,
        )

    assert state["status"] == "completed"
    assert state["final_response"] == "Portfolio passed all compliance constraints."
    # Total model invocations across retries: 3
    assert mock_llm.invoke.call_count == 3
    assert mock_sleep.call_count == 1
    # Iteration count must be exactly 2 (Call 1 + Call 2)
    assert state["iteration_count"] == 2


# ---------------------------------------------------------------------------
# 3. SDK Retry Configuration
# ---------------------------------------------------------------------------

def test_model_client_construction_uses_intended_retry_configuration():
    """Verify ChatOpenAI instances constructed by RAG and agent set max_retries=0 to avoid nested SDK retries."""
    with patch.object(settings, "openai_api_key", SecretStr("sk-mock-valid-key")):
        # 1. RAG service client
        service = RagService(documents=[])
        client = service._get_llm_client()
        assert client.max_retries == 0
        assert client.root_client.max_retries == 0

        # 2. Agent node client
        node = make_call_model_node(llm_client=None)
        # Execute node with dummy messages to trigger default client initialization
        with patch("app.services.resilience.execute_with_retry") as mock_exec:
            mock_exec.return_value = AIMessage(content="ok")
            state = {"messages": [], "iteration_count": 0, "max_iterations": 2}
            res = node(state)
            assert res["iteration_count"] == 1


# ---------------------------------------------------------------------------
# 4. Preserved Insufficient Evidence Behavior
# ---------------------------------------------------------------------------

def test_existing_insufficient_evidence_behavior_remains_unchanged():
    """Verify insufficient-evidence queries bypass LLM invocation and retries entirely."""
    mock_llm = MagicMock()
    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
        min_relevance_score=0.99,
    )

    with patch("app.services.resilience.time.sleep") as mock_sleep:
        result = service.answer_question("Query with zero relevant documents")

    assert result.status == "insufficient_evidence"
    assert mock_llm.invoke.call_count == 0
    assert mock_sleep.call_count == 0

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.operation == "rag_answer"
    assert event.model is None
    assert event.outcome == "insufficient_evidence"
