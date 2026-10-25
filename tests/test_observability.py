import io
import json
import logging
import time
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from app.agents.portfolio_graph import run_llm_portfolio_workflow
from app.config import settings
from app.evaluation.models import EvalTestCase
from app.evaluation.runner import create_offline_rag_service, run_evaluation
from app.logging_config import JsonFormatter, configure_logging
from app.observability import (
    ModelPricing,
    PricingRegistry,
    TelemetryEvent,
    TokenUsage,
    categorize_error,
    clear_recorded_events,
    create_default_pricing_registry,
    extract_token_usage,
    get_recorded_events,
    record_insufficient_evidence_without_llm,
    resolve_model_name,
    track_llm_operation,
)
from app.services.rag_service import RagService
from app.services.retrieval_service import FakeEmbeddingClient, InMemoryVectorIndex, RetrievalService


@pytest.fixture(autouse=True)
def reset_telemetry_sink():
    """Ensure in-memory event sink is clean before and after every test."""
    clear_recorded_events()
    yield
    clear_recorded_events()


# ---------------------------------------------------------------------------
# 1. Latency Recording (Monotonic)
# ---------------------------------------------------------------------------

def test_latency_recording_monotonic():
    """Verify operation duration is measured via monotonic clock in milliseconds."""
    with track_llm_operation(operation="test_operation", model="test-model") as tracker:
        time.sleep(0.015)  # 15 ms
        tracker.record_success(
            response=AIMessage(content="done"),
            outcome="success",
        )

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]

    assert event.operation == "test_operation"
    assert event.model == "test-model"
    assert event.duration_ms >= 10.0  # monotonic elapsed time >= 10ms
    assert isinstance(event.duration_ms, float)
    assert event.outcome == "success"


# ---------------------------------------------------------------------------
# 2. Provider Usage Metadata Extraction
# ---------------------------------------------------------------------------

def test_provider_usage_extraction_langchain_usage_metadata():
    """Verify token counts extracted from LangChain usage_metadata."""
    msg = AIMessage(
        content="Grounded answer",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        },
    )

    usage = extract_token_usage(msg)
    assert usage.input_tokens == 120
    assert usage.output_tokens == 45
    assert usage.total_tokens == 165


def test_provider_usage_extraction_langchain_response_metadata():
    """Verify token counts extracted from LangChain response_metadata token_usage."""
    msg = AIMessage(
        content="Grounded answer",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
            }
        },
    )

    usage = extract_token_usage(msg)
    assert usage.input_tokens == 200
    assert usage.output_tokens == 80
    assert usage.total_tokens == 280


def test_provider_usage_preserves_different_total_accounting():
    """Verify provider-reported total_tokens is preserved when it differs from input + output."""
    msg = AIMessage(
        content="Response with special accounting",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            # Provider reports 165 because of internal system or reasoning tokens
            "total_tokens": 165,
        },
    )

    usage = extract_token_usage(msg)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    # Must NOT be modified to 150; provider accounting is preserved
    assert usage.total_tokens == 165


# ---------------------------------------------------------------------------
# 3. Missing and Incomplete Token Usage
# ---------------------------------------------------------------------------

def test_missing_and_incomplete_token_usage():
    """Verify unavailable token counts remain explicit None and are never fabricated."""
    # A. Completely missing usage metadata
    msg_no_usage = AIMessage(content="No usage reported")
    usage_none = extract_token_usage(msg_no_usage)
    assert usage_none.input_tokens is None
    assert usage_none.output_tokens is None
    assert usage_none.total_tokens is None

    # B. Input tokens reported, but output and total omitted
    msg_partial = AIMessage(
        content="Partial usage",
        response_metadata={"token_usage": {"prompt_tokens": 55}},
    )
    usage_part = extract_token_usage(msg_partial)
    assert usage_part.input_tokens == 55
    assert usage_part.output_tokens is None
    assert usage_part.total_tokens is None  # Never synthesized

    # C. Both input and output reported, but total omitted
    msg_no_total = AIMessage(
        content="No total",
        response_metadata={"token_usage": {"prompt_tokens": 40, "completion_tokens": 60}},
    )
    usage_no_tot = extract_token_usage(msg_no_total)
    assert usage_no_tot.input_tokens == 40
    assert usage_no_tot.output_tokens == 60
    assert usage_no_tot.total_tokens is None  # Never synthesized as 100

    # D. Non-integer / boolean values are discarded safely
    msg_malformed = AIMessage(
        content="Malformed",
        response_metadata={"token_usage": {"prompt_tokens": True, "completion_tokens": "many", "total_tokens": None}},
    )
    usage_malformed = extract_token_usage(msg_malformed)
    assert usage_malformed.input_tokens is None
    assert usage_malformed.output_tokens is None
    assert usage_malformed.total_tokens is None


# ---------------------------------------------------------------------------
# 4. Known-Model Cost Calculations
# ---------------------------------------------------------------------------

def test_known_model_cost_calculations():
    """Verify cost calculations for explicitly configured models using the formula:
    (input_tokens / 1,000,000) * input_rate + (output_tokens / 1,000,000) * output_rate
    """
    registry = PricingRegistry({
        "gpt-4o-mini": ModelPricing(input_price_per_million=0.15, output_price_per_million=0.60),
        "gpt-4o": ModelPricing(input_price_per_million=2.50, output_price_per_million=10.00),
    })

    # Exact 1 million tokens of each for gpt-4o-mini
    cost_1m = registry.calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost_1m == pytest.approx(0.75, abs=1e-6)

    # Typical interaction: 1000 input tokens, 500 output tokens
    # (1000 / 1_000_000) * 0.15 = 0.00015
    # (500 / 1_000_000) * 0.60 = 0.00030
    # Total = 0.00045
    cost_typical = registry.calculate_cost("gpt-4o-mini", 1000, 500)
    assert cost_typical == pytest.approx(0.00045, abs=1e-8)

    # gpt-4o interaction
    # (10,000 / 1_000_000) * 2.50 = 0.025
    # (2,000 / 1_000_000) * 10.00 = 0.020
    # Total = 0.045
    cost_gpt4o = registry.calculate_cost("gpt-4o", 10000, 2000)
    assert cost_gpt4o == pytest.approx(0.045, abs=1e-8)


# ---------------------------------------------------------------------------
# 5. Unknown-Model Pricing
# ---------------------------------------------------------------------------

def test_unknown_model_pricing():
    """Verify unknown models do not have fabricated prices and return None."""
    registry = create_default_pricing_registry()

    cost = registry.calculate_cost("custom-unregistered-model", 5000, 2000)
    assert cost is None

    # Case-insensitive resolution for registered models
    cost_case = registry.calculate_cost("GPT-4O-MINI", 1000, 500)
    assert cost_case is not None


# ---------------------------------------------------------------------------
# 6. Missing Usage and Missing Pricing
# ---------------------------------------------------------------------------

def test_missing_usage_and_missing_pricing():
    """Verify no cost estimate is returned when usage or pricing is unavailable."""
    registry = create_default_pricing_registry()

    # Model known, but input_tokens is None
    assert registry.calculate_cost("gpt-4o-mini", None, 500) is None

    # Model known, but output_tokens is None
    assert registry.calculate_cost("gpt-4o-mini", 1000, None) is None

    # Model known, but both are None
    assert registry.calculate_cost("gpt-4o-mini", None, None) is None

    # Model unknown, and usage is None
    assert registry.calculate_cost("unknown-model", None, None) is None

    # Negative token counts rejected
    assert registry.calculate_cost("gpt-4o-mini", -10, 500) is None


# ---------------------------------------------------------------------------
# 7. Provider Exceptions and Timeouts
# ---------------------------------------------------------------------------

def test_provider_timeout_exception():
    """Verify timeouts produce an appropriate final event with timeout outcome and duration."""
    with pytest.raises(TimeoutError):
        with track_llm_operation(operation="agent_llm", model="gpt-4o-mini"):
            time.sleep(0.01)
            raise TimeoutError("Request to model provider timed out after 30s")

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.operation == "agent_llm"
    assert event.model == "gpt-4o-mini"
    assert event.outcome == "timeout"
    assert event.error_category == "timeout"
    assert event.duration_ms >= 8.0
    assert event.estimated_cost is None
    assert event.input_tokens is None


def test_provider_authentication_exception_sanitized():
    """Verify auth failure produces a failure event with sanitized error category."""
    with pytest.raises(Exception):
        with track_llm_operation(operation="rag_answer", model="gpt-4o-mini"):
            raise Exception("AuthenticationError: Invalid key sk-secret-abc-123456789")

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.outcome == "failure"
    assert event.error_category == "authentication"
    # Ensure sensitive string is NOT leaked in the telemetry event
    event_dict = event.to_dict()
    event_json = json.dumps(event_dict)
    assert "sk-secret" not in event_json
    assert "123456789" not in event_json


def test_provider_connection_exception():
    """Verify connection failure produces connection error category."""
    with pytest.raises(ConnectionError):
        with track_llm_operation(operation="test_op", model="gpt-4o-mini"):
            raise ConnectionError("Failed to establish a new connection: [Errno 111] Connection refused")

    events = get_recorded_events()
    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].error_category == "connection"


# ---------------------------------------------------------------------------
# 8. Insufficient Evidence without an LLM Invocation
# ---------------------------------------------------------------------------

def test_insufficient_evidence_without_llm_invocation():
    """Verify insufficient-evidence outcome is recorded without falsely reporting an LLM call."""
    mock_llm = MagicMock()
    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
        min_relevance_score=0.99,
    )

    result = service.answer_question("What is the leverage policy?")

    assert result.status == "insufficient_evidence"
    # CRITICAL: Verify LLM was NOT invoked
    assert mock_llm.invoke.call_count == 0

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]

    assert event.operation == "rag_answer"
    # Must NOT report an LLM call
    assert event.model is None
    assert event.input_tokens is None
    assert event.output_tokens is None
    assert event.total_tokens is None
    assert event.estimated_cost is None
    assert event.outcome == "insufficient_evidence"
    assert event.error_category is None
    assert event.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# 9. Privacy-Safe Logging and No Duplicate Completion Events
# ---------------------------------------------------------------------------

def test_privacy_safe_logging_and_no_duplicate_completion_events():
    """Verify concise structured JSON telemetry event without leaking sensitive details."""
    stream = io.StringIO()
    test_handler = logging.StreamHandler(stream)
    test_handler.setFormatter(JsonFormatter())

    obs_logger = logging.getLogger("app.observability")
    obs_logger.addHandler(test_handler)
    obs_logger.setLevel(logging.INFO)

    try:
        mock_response = AIMessage(
            content="Grounded synthetic answer with no private data.",
            usage_metadata={"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
        )

        with track_llm_operation(operation="rag_answer", model="gpt-4o-mini") as tracker:
            tracker.record_success(response=mock_response, outcome="success")

        output = stream.getvalue().strip()
        # Verify exactly ONE log line was emitted
        lines = [line for line in output.split("\n") if line.strip()]
        assert len(lines) == 1

        log_data = json.loads(lines[0])
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "app.observability"
        assert "telemetry" in log_data

        telemetry = log_data["telemetry"]
        assert telemetry["operation"] == "rag_answer"
        assert telemetry["model"] == "gpt-4o-mini"
        assert telemetry["input_tokens"] == 150
        assert telemetry["output_tokens"] == 50
        assert telemetry["total_tokens"] == 200
        assert telemetry["estimated_cost"] is not None
        assert telemetry["outcome"] == "success"

        # Verify privacy: no prompt, answer content, or private data logged in telemetry
        assert "Grounded synthetic answer" not in json.dumps(telemetry)
        assert "api_key" not in json.dumps(telemetry)

    finally:
        obs_logger.removeHandler(test_handler)


# ---------------------------------------------------------------------------
# 10. Existing RAG, Agent, API, and Evaluation Compatibility
# ---------------------------------------------------------------------------

def test_rag_service_full_call_emits_telemetry():
    """Verify RagService.answer_question emits telemetry on successful LLM invocation."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="According to [SYNTH-DOC-002], maintain 5.0% cash.",
        usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
    )

    service = RagService(
        llm_client=mock_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
    )

    result = service.answer_question("What is the cash reserve requirement?")
    assert result.status == "answered"

    events = get_recorded_events()
    assert len(events) == 1
    event = events[0]
    assert event.operation == "rag_answer"
    assert event.model == "gpt-4o-mini"
    assert event.input_tokens == 100
    assert event.output_tokens == 30
    assert event.total_tokens == 130
    assert event.estimated_cost is not None
    assert event.outcome == "success"


def test_portfolio_agent_emits_telemetry_per_invocation():
    """Verify run_llm_portfolio_workflow tracks each LLM invocation without duplicate events."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Portfolio is acceptable.",
        usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
    )

    state = run_llm_portfolio_workflow(
        user_query="How is the portfolio?",
        llm_client=mock_llm,
    )

    assert state["status"] == "completed"
    events = get_recorded_events()
    # 1 LLM call was made
    assert len(events) == 1
    assert events[0].operation == "portfolio_agent_model"
    assert events[0].input_tokens == 80
    assert events[0].output_tokens == 20
    assert events[0].outcome == "success"


def test_offline_evaluation_runner_compatibility():
    """Verify the deterministic offline evaluation runner functions with telemetry enabled."""
    rag_service = create_offline_rag_service(top_k=2)

    test_case = EvalTestCase(
        case_id="EVAL-TEST-001",
        question="What is the single security concentration ceiling?",
        category="concentration",
        expected_status="answered",
        expected_document_ids=["SYNTH-DOC-001"],
    )

    summary = run_evaluation(rag_service=rag_service, dataset=[test_case], top_k=2)

    assert summary.total_cases == 1
    assert len(summary.failed_cases) == 0
    assert summary.recall_at_k == 1.0

    # Verify telemetry was recorded during offline evaluation
    events = get_recorded_events()
    assert len(events) == 1
    assert events[0].operation == "rag_answer"
    assert events[0].outcome == "success"
