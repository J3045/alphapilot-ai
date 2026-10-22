# pyrefly: ignore[missing-import]
"""Retrieval and RAG quality evaluation metrics.

METRIC DEFINITIONS AND ASSUMPTIONS:
1. Recall@k:
   - Measures what proportion of expected ground-truth documents are captured within the top-k retrieved results.
   - For an individual query with non-empty ground-truth document IDs G and top-k retrieved document IDs R_k:
       Recall@k = |G ∩ R_k| / |G|
   - Duplicate retrieved document IDs (e.g., multiple chunks originating from the same document) are deduplicated
     prior to set intersection, ensuring each document is counted at most once.
   - ASSUMPTION: Recall@k is computed only across answerable queries (|G| > 0). Queries designed to test refusal /
     insufficient evidence (|G| == 0) return None for Recall@k and are evaluated via Answer Status Accuracy.

2. Mean Reciprocal Rank (MRR):
   - Measures the position of the first relevant document in the retrieved ranking.
   - For an individual query, if the first relevant document appears at 1-indexed rank r (where 1 <= r <= k):
       RR = 1.0 / r
     If no relevant document appears in the top-k results:
       RR = 0.0
   - MRR is the arithmetic mean of RR values across all answerable queries.
   - ASSUMPTION: Like Recall@k, MRR is evaluated over answerable queries (|G| > 0).

3. Answer Status Accuracy:
   - Measures the proportion of queries where the RAG pipeline returned the expected status ('answered' vs. 'insufficient_evidence').
   - Evaluated across ALL test cases (both answerable and unanswerable).

4. Keyword Grounding Match Rate:
   - BASIC HEURISTIC: Checks whether key ground-truth terms (e.g. '5.0%', '20.0%') appear in the model's answer.
   - NOTE: This is a lightweight heuristic to detect complete hallucinations, NOT a substitute for full semantic evaluation.
"""

from typing import Sequence

from app.evaluation.models import EvalCaseResult, EvaluationSummary


def calculate_recall_at_k(
    expected_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """Calculate Recall@k for an individual query.

    Args:
        expected_doc_ids: Collection of ground-truth document IDs expected for the query.
        retrieved_doc_ids: Ordered list of document IDs retrieved by the index.
        k: Cutoff rank (must be a positive integer).

    Returns:
        Float score between 0.0 and 1.0, or None if expected_doc_ids is empty.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    if not expected_doc_ids:
        return None

    # Deduplicate retrieved document IDs preserving original rank order
    seen: set[str] = set()
    unique_retrieved_top_k: list[str] = []
    for doc_id in retrieved_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_retrieved_top_k.append(doc_id)
        if len(unique_retrieved_top_k) >= k:
            break

    expected_set = set(expected_doc_ids)
    retrieved_set = set(unique_retrieved_top_k)

    hits = len(expected_set.intersection(retrieved_set))
    return hits / len(expected_set)


def calculate_reciprocal_rank(
    expected_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int,
) -> float | None:
    """Calculate Reciprocal Rank (RR) at cutoff k for an individual query.

    Args:
        expected_doc_ids: Collection of ground-truth document IDs expected for the query.
        retrieved_doc_ids: Ordered list of document IDs retrieved by the index.
        k: Cutoff rank (must be a positive integer).

    Returns:
        Float reciprocal rank (e.g., 1.0 for rank 1, 0.5 for rank 2, 0.0 if not found),
        or None if expected_doc_ids is empty.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    if not expected_doc_ids:
        return None

    expected_set = set(expected_doc_ids)
    sliced_retrieved = list(retrieved_doc_ids[:k])

    for rank, doc_id in enumerate(sliced_retrieved, start=1):
        if doc_id in expected_set:
            return 1.0 / rank

    return 0.0


def calculate_evaluation_summary(
    case_results: Sequence[EvalCaseResult],
    k: int,
) -> EvaluationSummary:
    """Aggregate individual test case results into a complete EvaluationSummary.

    Args:
        case_results: Sequence of completed EvalCaseResult records.
        k: Top-k cutoff parameter used for retrieval evaluation.

    Returns:
        EvaluationSummary containing aggregate metrics and list of failed cases.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    total_cases = len(case_results)
    if total_cases == 0:
        return EvaluationSummary(
            total_cases=0,
            answerable_cases=0,
            unanswerable_cases=0,
            k=k,
            recall_at_k=0.0,
            mrr=0.0,
            answer_status_accuracy=0.0,
            keyword_match_rate=0.0,
            failed_cases=[],
        )

    answerable_cases = [c for c in case_results if len(c.expected_document_ids) > 0]
    unanswerable_cases = [c for c in case_results if len(c.expected_document_ids) == 0]

    # 1. Retrieval Metrics (computed across answerable cases with expected documents)
    if answerable_cases:
        recall_values = [c.recall_at_k for c in answerable_cases if c.recall_at_k is not None]
        mean_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0

        rr_values = [c.reciprocal_rank for c in answerable_cases if c.reciprocal_rank is not None]
        mean_mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0

        keyword_matches = sum(1 for c in answerable_cases if c.keyword_matched)
        keyword_rate = keyword_matches / len(answerable_cases)
    else:
        mean_recall = 0.0
        mean_mrr = 0.0
        keyword_rate = 0.0

    # 2. Answer Status Accuracy (evaluated across all cases)
    status_correct_count = sum(1 for c in case_results if c.status_correct)
    status_accuracy = status_correct_count / total_cases

    # 3. Identify failed cases
    failed_cases = [
        c for c in case_results
        if not c.status_correct
        or (c.recall_at_k is not None and c.recall_at_k < 1.0)
        or not c.keyword_matched
    ]

    return EvaluationSummary(
        total_cases=total_cases,
        answerable_cases=len(answerable_cases),
        unanswerable_cases=len(unanswerable_cases),
        k=k,
        recall_at_k=round(mean_recall, 4),
        mrr=round(mean_mrr, 4),
        answer_status_accuracy=round(status_accuracy, 4),
        keyword_match_rate=round(keyword_rate, 4),
        failed_cases=failed_cases,
    )
