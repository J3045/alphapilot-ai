# pyrefly: ignore[missing-import]
"""AlphaPilot AI RAG Evaluation & Quality Metrics Package."""

from app.evaluation.dataset import GOLDEN_EVAL_DATASET, get_golden_dataset
from app.evaluation.metrics import (
    calculate_evaluation_summary,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.evaluation.models import (
    EvalCaseResult,
    EvalTestCase,
    EvaluationSummary,
)
from app.evaluation.runner import run_evaluation

__all__ = [
    "EvalCaseResult",
    "EvalTestCase",
    "EvaluationSummary",
    "GOLDEN_EVAL_DATASET",
    "calculate_evaluation_summary",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "get_golden_dataset",
    "run_evaluation",
]
