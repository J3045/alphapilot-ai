# pyrefly: ignore[missing-import]
"""Deterministic unit and integration tests for workflow execution budgets."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.agents.portfolio_graph import run_llm_portfolio_workflow
from app.config import Settings, settings
from app.main import app
from app.observability import clear_recorded_events, get_recorded_events
from app.services.rag_service import RagService
from app.services.resilience import (
    WorkflowDeadline,
    WorkflowDeadlineExceededError,
    calculate_backoff,
    execute_with_retry,
    is_retryable_error,
)
from app.services.retrieval_service import FakeEmbeddingClient
from tests.test_resilience import SyntheticRateLimitError


class SimulatedClock:
    """Deterministic monotonic clock for testing deadlines without real-time delays."""

    def __init__(self, current_time: float = 100.0) -> None:
        self.time = current_time

    def now(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


@pytest.fixture(autouse=True)
def reset_telemetry_sink():
    """Ensure in-memory event sink is clean before and after every test."""
    clear_recorded_events()
    yield
    clear_recorded_events()


# ---------------------------------------------------------------------------
# 1. Configuration Validation
# ---------------------------------------------------------------------------

def test_workflow_timeout_seconds_config():
    """Verify workflow_timeout_seconds validation and bounds."""
    # Default is 60.0
    config_default = Settings()
    assert config_default.workflow_timeout_seconds == 60.0

    # Custom valid setting
    config_custom = Settings(workflow_timeout_seconds=45.0)
    assert config_custom.workflow_timeout_seconds == 45.0

    # Bounds: minimum 1.0
    with pytest.raises(ValidationError):
        Settings(workflow_timeout_seconds=0.5)

    with pytest.raises(ValidationError):
        Settings(workflow_timeout_seconds=-10.0)

    # Bounds: maximum 600.0
    with pytest.raises(ValidationError):
        Settings(workflow_timeout_seconds=601.0)


# ---------------------------------------------------------------------------
# 2. Deadline Remaining-Time and Expiration Calculation
# ---------------------------------------------------------------------------

def test_deadline_remaining_time_and_expiration():
    """Verify monotonic deadline calculations, remaining budget, and expiration."""
    clock = SimulatedClock(current_time=100.0)
    deadline = WorkflowDeadline(timeout_seconds=10.0, time_fn=clock.now)

    assert deadline.timeout_seconds == 10.0
    assert deadline.elapsed_seconds() == 0.0
    assert deadline.remaining_seconds() == 10.0
    assert not deadline.is_expired()
    # Does not raise
    deadline.check_or_raise()

    # Advance 4 seconds
    clock.advance(4.0)
    assert deadline.elapsed_seconds() == 4.0
    assert deadline.remaining_seconds() == 6.0
    assert not deadline.is_expired()
    deadline.check_or_raise()

    # Advance 6 more seconds (exactly exhausted)
    clock.advance(6.0)
    assert deadline.elapsed_seconds() == 10.0
    assert deadline.remaining_seconds() == 0.0
    assert deadline.is_expired()
    with pytest.raises(WorkflowDeadlineExceededError) as exc_info:
        deadline.check_or_raise()
    assert "budget of 10.0s exceeded" in str(exc_info.value)

    # Advance further past deadline; remaining_seconds clamped at 0.0
    clock.advance(5.0)
    assert deadline.elapsed_seconds() == 15.0
    assert deadline.remaining_seconds() == 0.0
    assert deadline.is_expired()


# ---------------------------------------------------------------------------
# 3. Expired Deadline Before First Attempt
# ---------------------------------------------------------------------------

def test_expired_deadline_before_first_attempt():
    """Verify execute_with_retry does not invoke fn if deadline is already expired."""
    clock = SimulatedClock(current_time=100.0)
    deadline = WorkflowDeadline(timeout_seconds=5.0, time_fn=clock.now)
    clock.advance(6.0)  # Expire deadline

    fn = MagicMock()
    with pytest.raises(WorkflowDeadlineExceededError):
        execute_with_retry(fn=fn, deadline=deadline)

    assert fn.call_count == 0


# ---------------------------------------------------------------------------
# 4. Deadline Expiration Between Retry Attempts
# ---------------------------------------------------------------------------

def test_deadline_expiration_between_retry_attempts():
    """Verify retry loop aborts if deadline expires while preparing next attempt."""
    clock = SimulatedClock(current_time=100.0)
    deadline = WorkflowDeadline(timeout_seconds=5.0, time_fn=clock.now)

    fn = MagicMock(side_effect=SyntheticRateLimitError("Rate limit reached"))
    mock_sleep = MagicMock()

    def advance_and_sleep(duration):
        mock_sleep(duration)
        clock.advance(6.0)  # Exhaust deadline during/after backoff

    with pytest.raises(WorkflowDeadlineExceededError):
        execute_with_retry(
            fn=fn,
            max_retries=3,
            initial_backoff_seconds=1.0,
            rng=lambda low, high: high,
            sleep_fn=advance_and_sleep,
            deadline=deadline,
        )

    # First attempt ran, but second attempt was blocked by expired deadline
    assert fn.call_count == 1
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# 5. Backoff Exceeding Remaining Budget
# ---------------------------------------------------------------------------

def test_backoff_exceeding_remaining_budget():
    """Verify execute_with_retry raises deadline error if backoff exceeds remaining time."""
    clock = SimulatedClock(current_time=100.0)
    # 2.0 second budget
    deadline = WorkflowDeadline(timeout_seconds=2.0, time_fn=clock.now)

    # Advance 1.5s, leaving 0.5s remaining
    clock.advance(1.5)
    assert deadline.remaining_seconds() == pytest.approx(0.5)

    fn = MagicMock(side_effect=SyntheticRateLimitError("Rate limit reached"))
    mock_sleep = MagicMock()

    # Backoff calculation would yield 1.0s, which exceeds 0.5s remaining
    with pytest.raises(WorkflowDeadlineExceededError) as exc_info:
        execute_with_retry(
            fn=fn,
            max_retries=3,
            initial_backoff_seconds=1.0,
            rng=lambda low, high: high,
            sleep_fn=mock_sleep,
            deadline=deadline,
        )

    assert fn.call_count == 1
    # Does NOT sleep for the excessive backoff
    assert mock_sleep.call_count == 0
    assert "less than required backoff" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Existing Retries Work Without Deadline
# ---------------------------------------------------------------------------

def test_existing_retries_work_without_deadline():
    """Verify execute_with_retry preserves backward compatibility when deadline is None."""
    fn = MagicMock(side_effect=[
        SyntheticRateLimitError("Transient 429"),
        "success_result",
    ])
    mock_sleep = MagicMock()

    result = execute_with_retry(
        fn=fn,
        max_retries=2,
        initial_backoff_seconds=0.1,
        rng=lambda low, high: high,
        sleep_fn=mock_sleep,
        deadline=None,
    )

    assert result == "success_result"
    assert fn.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# 7. RAG Deadline Failure and API Contract
# ---------------------------------------------------------------------------

def test_rag_deadline_failure_result_and_api():
    """Verify RAG handles deadline expiration cleanly and respects API contract."""
    mock_llm = MagicMock()
    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
    )

    # Call with simulated 0.0s deadline (already expired)
    clock = SimulatedClock(current_time=100.0)
    expired_deadline = WorkflowDeadline(timeout_seconds=0.0, time_fn=clock.now)

    with patch("app.services.rag_service.WorkflowDeadline", return_value=expired_deadline):
        result = service.answer_question("What is the cash reserve policy?")

    assert result.status == "failed"
    assert result.answer == ""
    assert result.sources == []
    assert "Workflow execution deadline exceeded" in result.error
    assert mock_llm.invoke.call_count == 0

    # API Contract: /rag/ask returns HTTP 503 on provider/workflow failure
    with patch("app.main.get_default_rag_service", return_value=service):
        with patch("app.services.rag_service.WorkflowDeadline", return_value=expired_deadline):
            with TestClient(app) as client:
                response = client.post("/rag/ask", json={"question": "What is the cash policy?"})
                assert response.status_code == 503
                data = response.json()
                assert "Workflow execution deadline exceeded" in data["detail"]


# ---------------------------------------------------------------------------
# 8. Agent Deadline Failure and API Contract
# ---------------------------------------------------------------------------

def test_agent_deadline_failure_graph_and_api():
    """Verify agent workflow handles deadline expiration without fabricated results."""
    mock_llm = MagicMock()
    clock = SimulatedClock(current_time=100.0)
    expired_deadline = WorkflowDeadline(timeout_seconds=0.0, time_fn=clock.now)

    # Direct workflow invocation
    state = run_llm_portfolio_workflow(
        user_query="Assess my portfolio risk",
        llm_client=mock_llm,
        deadline=expired_deadline,
    )

    assert state.get("status") == "failed"
    assert "Workflow execution deadline exceeded" in state.get("error", "")
    assert state.get("assessment_result") is None
    assert state.get("final_response") is None
    assert mock_llm.invoke.call_count == 0

    # API Contract: /agent/run returns HTTP 200 with handled failure schema
    with patch("app.agents.portfolio_graph.ChatOpenAI", return_value=mock_llm):
        with patch("app.agents.portfolio_graph.WorkflowDeadline", return_value=expired_deadline):
            with TestClient(app) as client:
                response = client.post("/agent/run", json={"query": "Assess risk"})
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "failed"
                assert "Workflow execution deadline exceeded" in data["error"]
                assert data["assessment_result"] is None
                assert data["final_response"] is None


# ---------------------------------------------------------------------------
# 9. Shared Deadline Across Multiple Agent Model Invocations
# ---------------------------------------------------------------------------

def test_shared_deadline_across_multiple_agent_invocations():
    """Verify agent invocations share one monotonic budget across tool calls."""
    clock = SimulatedClock(current_time=100.0)
    shared_deadline = WorkflowDeadline(timeout_seconds=10.0, time_fn=clock.now)

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_1",
            "name": "assess_portfolio_risk",
            "args": {"holdings": [{"symbol": "AAPL", "weight": 1.0}]},
        }],
    )

    def model_invoke(messages):
        # Call 1: advance clock by 4 seconds and return tool call
        clock.advance(4.0)
        return tool_call_msg

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = model_invoke

    # Tool executes, and then before second model invocation, advance clock past deadline
    with patch("app.agents.portfolio_graph.execute_tool") as mock_tool:
        def tool_impl(*args, **kwargs):
            clock.advance(7.0)  # Total elapsed now 11.0s > 10.0s deadline
            return {"risk_level": "moderate"}
        mock_tool.side_effect = tool_impl

        state = run_llm_portfolio_workflow(
            user_query="Assess portfolio",
            llm_client=mock_llm,
            deadline=shared_deadline,
        )

    # Second model call or check aborts due to shared deadline
    assert state.get("status") == "failed"
    assert "Workflow execution deadline exceeded" in state.get("error", "")
    assert state.get("assessment_result") is None
    assert state.get("final_response") is None


# ---------------------------------------------------------------------------
# 10. Telemetry Timeout Recording Without Duplicate Final Events
# ---------------------------------------------------------------------------

def test_telemetry_records_timeout_without_duplicate_events():
    """Verify deadline expiration records outcome='timeout' and error_category='timeout' exactly once."""
    clock = SimulatedClock(current_time=100.0)
    deadline = WorkflowDeadline(timeout_seconds=5.0, time_fn=clock.now)

    def slow_model_call(messages):
        # Model call overruns and deadline expires
        clock.advance(6.0)
        deadline.check_or_raise()

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = slow_model_call

    state = run_llm_portfolio_workflow(
        user_query="Assess portfolio",
        llm_client=mock_llm,
        deadline=deadline,
    )

    assert state.get("status") == "failed"
    assert "Workflow execution deadline exceeded" in state.get("error", "")

    events = get_recorded_events()
    assert len(events) == 1, f"Expected exactly 1 telemetry event, found {len(events)}"
    event = events[0]
    assert event.outcome == "timeout"
    assert event.error_category == "timeout"
    assert event.operation == "portfolio_agent_model"
