# pyrefly: ignore[missing-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.portfolio import Portfolio


class AgentRunRequest(BaseModel):
    """Request payload for the LLM portfolio agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query or directive for the portfolio agent (1-2000 characters).",
    )
    portfolio: Portfolio | None = Field(
        default=None,
        description="Optional portfolio context to assess alongside the query.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query cannot be empty or whitespace-only.")
        if len(stripped) > 2000:
            raise ValueError("query cannot exceed 2000 characters.")
        return stripped


class AgentRunResponse(BaseModel):
    """Response payload returned by the LLM portfolio agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Execution status: 'completed', 'failed', or 'pending'.",
    )
    assessment_result: dict | None = Field(
        default=None,
        description="Deterministic risk assessment output if tool was executed.",
    )
    final_response: str | None = Field(
        default=None,
        description="Natural language response synthesized by the model.",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized, safe error description if execution failed.",
    )
