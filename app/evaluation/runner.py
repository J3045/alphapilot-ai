# pyrefly: ignore[missing-import]
"""Deterministic evaluation runner for AlphaPilot AI RAG pipeline."""

import argparse
import logging
import sys
from typing import Sequence
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.evaluation.dataset import GOLDEN_EVAL_DATASET, get_golden_dataset
from app.evaluation.metrics import (
    calculate_evaluation_summary,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.evaluation.models import EvalCaseResult, EvalTestCase, EvaluationSummary
from app.services.rag_service import RagService
from app.services.retrieval_service import FakeEmbeddingClient

logger = logging.getLogger(__name__)


def create_offline_rag_service(top_k: int = 3) -> RagService:
    """Create a fully offline RagService backed by FakeEmbeddingClient and a deterministic mock LLM.

    Requires zero OpenAI credentials, zero network access, and zero API charges.
    """
    fake_embedder = FakeEmbeddingClient(dimension=64)

    def mock_llm_responder(messages):
        # Extract user content from HumanMessage
        user_msg = messages[-1].content
        question_text = user_msg.split("--- QUESTION ---")[-1].strip().lower()

        # Check for unanswerable topics not present in synthetic policy documents
        unanswerable_terms = ("leverage", "margin", "cryptocurrency", "crypto", "staking", "solana")
        if any(term in question_text for term in unanswerable_terms):
            return AIMessage(
                content="The provided policy documents do not contain sufficient evidence to answer this question."
            )

        # Extract evidence text
        evidence_text = ""
        if "Retrieved Evidence Excerpts" in user_msg:
            evidence_text = user_msg.split("--- QUESTION ---")[0]

        # Generate a deterministic grounded response synthesizing available evidence
        return AIMessage(
            content=(
                f"Based on the provided policy evidence: {evidence_text.strip()[:300]}..."
            )
        )

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_llm_responder

    return RagService(
        llm_client=mock_llm,
        embedding_client=fake_embedder,
        min_relevance_score=0.15,
        default_top_k=top_k,
    )


def run_evaluation(
    rag_service: RagService | None = None,
    dataset: Sequence[EvalTestCase] | None = None,
    top_k: int = 3,
) -> EvaluationSummary:
    """Execute the golden dataset evaluation across the provided RAG pipeline.

    Args:
        rag_service: The RAG service instance to evaluate. If None, an offline deterministic service is used.
        dataset: The evaluation cases to run. Defaults to the golden dataset.
        top_k: The retrieval top-k cutoff parameter to evaluate.

    Returns:
        EvaluationSummary containing individual case results and aggregate metrics.
    """
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    cases = dataset if dataset is not None else get_golden_dataset()
    service = rag_service if rag_service is not None else create_offline_rag_service(top_k=top_k)

    case_results: list[EvalCaseResult] = []

    for test_case in cases:
        response = service.answer_question(question=test_case.question, top_k=top_k)

        actual_status = response.status
        status_correct = (actual_status == test_case.expected_status)

        retrieved_doc_ids = [src.document_id for src in response.sources]

        # Retrieval metrics
        recall = calculate_recall_at_k(
            expected_doc_ids=test_case.expected_document_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            k=top_k,
        )
        reciprocal_rank = calculate_reciprocal_rank(
            expected_doc_ids=test_case.expected_document_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            k=top_k,
        )

        # Keyword grounding heuristic
        answer_text = response.answer
        if test_case.expected_keywords:
            # Check keywords against answer or retrieved evidence chunk text
            evidence_and_answer = f"{answer_text} {' '.join(s.chunk_text for s in response.sources)}"
            matched_keywords = [
                kw for kw in test_case.expected_keywords
                if kw.lower() in evidence_and_answer.lower()
            ]
            keyword_matched = (len(matched_keywords) == len(test_case.expected_keywords))
        else:
            matched_keywords = []
            keyword_matched = True

        case_results.append(
            EvalCaseResult(
                case_id=test_case.case_id,
                question=test_case.question,
                category=test_case.category,
                expected_status=test_case.expected_status,
                actual_status=actual_status,
                status_correct=status_correct,
                expected_document_ids=test_case.expected_document_ids,
                retrieved_document_ids=retrieved_doc_ids,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                expected_keywords=test_case.expected_keywords,
                matched_keywords=matched_keywords,
                keyword_matched=keyword_matched,
                answer=answer_text,
            )
        )

    return calculate_evaluation_summary(case_results=case_results, k=top_k)


def format_evaluation_report(summary: EvaluationSummary) -> str:
    """Format an EvaluationSummary into a readable plaintext console report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("                 ALPHAPILOT AI — RAG EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Test Cases:       {summary.total_cases}")
    lines.append(f"  - Answerable Cases:   {summary.answerable_cases}")
    lines.append(f"  - Unanswerable Cases: {summary.unanswerable_cases}")
    lines.append(f"Retrieval Top-K (k):    {summary.k}")
    lines.append("-" * 80)
    lines.append("AGGREGATE QUALITY METRICS:")
    lines.append(f"  • Recall@{summary.k}:             {summary.recall_at_k * 100:.1f}%")
    lines.append(f"  • Mean Reciprocal Rank (MRR): {summary.mrr:.4f}")
    lines.append(f"  • Answer Status Accuracy:     {summary.answer_status_accuracy * 100:.1f}%")
    lines.append(f"  • Keyword Match Heuristic:    {summary.keyword_match_rate * 100:.1f}%")
    lines.append("-" * 80)

    if summary.failed_cases:
        lines.append(f"FAILED CASES ({len(summary.failed_cases)}):")
        for fail in summary.failed_cases:
            lines.append(f"  Case {fail.case_id} [{fail.category}]:")
            lines.append(f"    Q: {fail.question}")
            lines.append(f"    Status: Expected='{fail.expected_status}' | Actual='{fail.actual_status}'")
            lines.append(f"    Docs:   Expected={fail.expected_document_ids} | Retrieved={fail.retrieved_document_ids}")
            lines.append(f"    Recall: {fail.recall_at_k} | RR: {fail.reciprocal_rank}")
            if not fail.keyword_matched:
                lines.append(f"    Missing Keywords: {[k for k in fail.expected_keywords if k not in fail.matched_keywords]}")
    else:
        lines.append("ALL EVALUATION CASES PASSED WITH 100% RECALL AND STATUS ACCURACY!")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running the RAG evaluation suite."""
    parser = argparse.ArgumentParser(
        description="Run the AlphaPilot AI RAG quality evaluation suite.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieval top-k parameter (default: 3).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Opt-in to live OpenAI evaluation (requires OPENAI_API_KEY in .env).",
    )
    args = parser.parse_args()

    if args.live:
        print("[INFO] Running in LIVE mode with configured OpenAI provider...")
        service = RagService(default_top_k=args.top_k)
    else:
        print("[INFO] Running in OFFLINE deterministic mode (zero API credits used)...")
        service = create_offline_rag_service(top_k=args.top_k)

    summary = run_evaluation(rag_service=service, top_k=args.top_k)
    print(format_evaluation_report(summary))

    if summary.failed_cases:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
