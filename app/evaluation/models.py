# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field


class EvalTestCase(BaseModel):
    """A test case in the golden evaluation dataset."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1, description="Unique case identifier.")
    question: str = Field(..., min_length=1, description="Policy question to evaluate.")
    category: str = Field(..., min_length=1, description="Policy topic category.")
    expected_status: str = Field(
        ...,
        description="Expected RAG status ('answered' or 'insufficient_evidence').",
    )
    expected_document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of documents expected to contain the answer. Empty for unanswerable questions.",
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Key terms expected in the grounded answer (used for basic heuristic check).",
    )
    is_synthetic: bool = Field(
        default=True,
        description="Flag confirming dataset is purely synthetic test data.",
    )
    notes: str = Field(
        default="",
        description="Contextual notes on policy grounding or expected behavior.",
    )


class EvalCaseResult(BaseModel):
    """Evaluation result for an individual test case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    category: str
    expected_status: str
    actual_status: str
    status_correct: bool
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    recall_at_k: float | None
    reciprocal_rank: float | None
    expected_keywords: list[str]
    matched_keywords: list[str]
    keyword_matched: bool
    answer: str


class EvaluationSummary(BaseModel):
    """Aggregate evaluation summary across all test cases."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int = Field(..., ge=0, description="Total number of evaluated cases.")
    answerable_cases: int = Field(..., ge=0, description="Count of answerable cases with expected documents.")
    unanswerable_cases: int = Field(..., ge=0, description="Count of cases expecting insufficient evidence.")
    k: int = Field(..., gt=0, description="Retrieval top-k cutoff parameter used.")
    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Mean Recall@k across answerable cases.")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank across answerable cases.")
    answer_status_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of cases where actual status matched expected status.",
    )
    keyword_match_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Basic heuristic: proportion of answerable cases containing expected keywords.",
    )
    failed_cases: list[EvalCaseResult] = Field(
        default_factory=list,
        description="List of cases that failed status check, recall, or keyword heuristics.",
    )
