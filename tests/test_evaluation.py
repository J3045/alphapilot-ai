from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.evaluation.dataset import GOLDEN_EVAL_DATASET, get_golden_dataset
from app.evaluation.metrics import (
    calculate_evaluation_summary,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.evaluation.models import EvalCaseResult, EvalTestCase, EvaluationSummary
from app.evaluation.runner import format_evaluation_report, run_evaluation
from app.main import app
from app.services.rag_service import RagAnswerResult


# 1. Recall@k metric tests
def test_recall_at_k_exact_match():
    assert calculate_recall_at_k(["DOC-1"], ["DOC-1", "DOC-2"], k=3) == 1.0


def test_recall_at_k_partial_match():
    assert calculate_recall_at_k(["DOC-1", "DOC-2"], ["DOC-1", "DOC-3"], k=3) == 0.5


def test_recall_at_k_no_match():
    assert calculate_recall_at_k(["DOC-1"], ["DOC-2", "DOC-3"], k=3) == 0.0


def test_recall_at_k_duplicate_retrieved_documents():
    # Multiple chunks from DOC-1 should not artificially inflate or distort recall
    assert calculate_recall_at_k(["DOC-1", "DOC-2"], ["DOC-1", "DOC-1", "DOC-2"], k=3) == 1.0


def test_recall_at_k_cutoff_boundary():
    # DOC-1 is at index 2 (rank 3)
    assert calculate_recall_at_k(["DOC-1"], ["DOC-2", "DOC-3", "DOC-1"], k=2) == 0.0
    assert calculate_recall_at_k(["DOC-1"], ["DOC-2", "DOC-3", "DOC-1"], k=3) == 1.0


def test_recall_at_k_empty_expected_docs_returns_none():
    # Unanswerable queries have no expected documents
    assert calculate_recall_at_k([], ["DOC-1"], k=3) is None


def test_recall_at_k_invalid_k_raises_value_error():
    with pytest.raises(ValueError, match="k must be a positive integer"):
        calculate_recall_at_k(["DOC-1"], ["DOC-1"], k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        calculate_recall_at_k(["DOC-1"], ["DOC-1"], k=-1)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        calculate_recall_at_k(["DOC-1"], ["DOC-1"], k="two")  # type: ignore[arg-type]


# 2. Reciprocal Rank (RR / MRR) metric tests
def test_reciprocal_rank_positions():
    # Rank 1 -> 1.0
    assert calculate_reciprocal_rank(["DOC-1"], ["DOC-1", "DOC-2"], k=3) == 1.0

    # Rank 2 -> 0.5
    assert calculate_reciprocal_rank(["DOC-1"], ["DOC-2", "DOC-1"], k=3) == 0.5

    # Rank 3 -> 0.3333...
    assert pytest.approx(calculate_reciprocal_rank(["DOC-1"], ["DOC-2", "DOC-3", "DOC-1"], k=3), 1e-4) == 1.0 / 3

    # Missing from top-k -> 0.0
    assert calculate_reciprocal_rank(["DOC-1"], ["DOC-2", "DOC-3"], k=3) == 0.0


def test_reciprocal_rank_empty_expected_docs_returns_none():
    assert calculate_reciprocal_rank([], ["DOC-1"], k=3) is None


def test_reciprocal_rank_invalid_k_raises_value_error():
    with pytest.raises(ValueError, match="k must be a positive integer"):
        calculate_reciprocal_rank(["DOC-1"], ["DOC-1"], k=0)


# 3. Evaluation summary and edge cases
def test_empty_dataset_evaluation_summary():
    summary = calculate_evaluation_summary([], k=3)
    assert summary.total_cases == 0
    assert summary.answerable_cases == 0
    assert summary.unanswerable_cases == 0
    assert summary.recall_at_k == 0.0
    assert summary.mrr == 0.0
    assert summary.answer_status_accuracy == 0.0
    assert summary.keyword_match_rate == 0.0
    assert summary.failed_cases == []


def test_golden_dataset_structure_and_integrity():
    dataset = get_golden_dataset()
    # Must have at least 8 synthetic policy test cases
    assert len(dataset) >= 8

    valid_doc_ids = {"SYNTH-DOC-001", "SYNTH-DOC-002", "SYNTH-DOC-003"}
    categories = set()

    for case in dataset:
        assert case.case_id.startswith("EVAL-")
        assert len(case.question) > 10
        assert case.expected_status in {"answered", "insufficient_evidence"}
        assert case.is_synthetic is True
        categories.add(case.category)

        if case.expected_status == "answered":
            assert len(case.expected_document_ids) > 0
            for doc_id in case.expected_document_ids:
                assert doc_id in valid_doc_ids
        else:
            assert case.expected_document_ids == []

    # Verify coverage across all three policy areas plus unanswerable cases
    assert "concentration_limits" in categories
    assert "cash_allocation" in categories
    assert "risk_review" in categories


# 4. Evaluation runner offline deterministic execution
def test_evaluation_runner_offline_execution():
    summary = run_evaluation(top_k=3)

    assert summary.total_cases == 10
    assert summary.answerable_cases == 8
    assert summary.unanswerable_cases == 2
    assert summary.k == 3

    # High quality retrieval and accuracy on golden dataset
    assert summary.recall_at_k == 1.0
    assert summary.mrr >= 0.8
    assert summary.answer_status_accuracy == 1.0
    assert summary.keyword_match_rate == 1.0
    assert len(summary.failed_cases) == 0

    # Ensure formatted report renders properly
    report = format_evaluation_report(summary)
    assert "ALPHAPILOT AI — RAG EVALUATION REPORT" in report
    assert "Recall@3:             100.0%" in report
    assert "ALL EVALUATION CASES PASSED" in report


def test_evaluation_runner_detects_failures():
    # Construct a mock RAG service that fails all cases
    failing_service = MagicMock()
    failing_service.answer_question.return_value = RagAnswerResult(
        status="failed",
        answer="",
        sources=[],
        error="Model provider error.",
    )

    test_case = EvalTestCase(
        case_id="FAIL-001",
        question="What is the cash policy?",
        category="cash_allocation",
        expected_status="answered",
        expected_document_ids=["SYNTH-DOC-002"],
        expected_keywords=["5.0%"],
    )

    summary = run_evaluation(
        rag_service=failing_service,
        dataset=[test_case],
        top_k=3,
    )

    assert summary.total_cases == 1
    assert summary.answer_status_accuracy == 0.0
    assert summary.recall_at_k == 0.0
    assert len(summary.failed_cases) == 1
    assert summary.failed_cases[0].case_id == "FAIL-001"


# 5. Preservation of existing endpoints
def test_existing_endpoints_preserved():
    with TestClient(app) as client:
        # 1. Health check
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        assert h_resp.json()["status"] == "ok"

        # 2. Portfolio assessment
        portfolio_payload = {
            "portfolio_id": "PRESERVE-001",
            "cash": 3000.0,
            "holdings": [
                {"symbol": "AAPL", "quantity": 10.0, "price": 100.0},
            ],
        }
        assess_resp = client.post("/portfolio/assess", json=portfolio_payload)
        assert assess_resp.status_code == 200
        assert assess_resp.json()["passed"] is True

        # 3. RAG ask endpoint validation
        rag_resp = client.post("/rag/ask", json={"question": ""})
        assert rag_resp.status_code == 422
