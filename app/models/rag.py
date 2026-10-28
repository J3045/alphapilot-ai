# pyrefly: ignore[missing-import]
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceReference(BaseModel):
    """Source reference representing a retrieved evidence chunk used in RAG synthesis."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the originating research document.",
    )
    document_title: str = Field(
        ...,
        min_length=1,
        description="Title of the originating research document.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="0-based sequential chunk index within the document.",
    )
    chunk_text: str = Field(
        ...,
        min_length=1,
        description="Text content of the retrieved chunk excerpt.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved source metadata from the originating document.",
    )
    score: float = Field(
        ...,
        description="Similarity relevance score of this chunk against the question.",
    )

    @property
    def title(self) -> str:
        """Convenience alias for document_title."""
        return self.document_title

    @property
    def text(self) -> str:
        """Convenience alias for chunk_text."""
        return self.chunk_text


class RagAskRequest(BaseModel):
    """Request payload for the RAG question-answering endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Non-empty research or policy question to answer (max 2000 characters).",
    )
    top_k: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Maximum number of context chunks to retrieve (1-10, default: 3).",
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be empty or whitespace-only.")
        if len(stripped) > 2000:
            raise ValueError("question cannot exceed 2000 characters.")
        return stripped


class RagAskResponse(BaseModel):
    """Response payload for the RAG question-answering endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="Outcome status: 'answered', 'insufficient_evidence', or 'failed'.",
    )
    answer: str = Field(
        ...,
        description="Grounded response synthesized strictly from retrieved evidence.",
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="List of source evidence references supporting the answer.",
    )
    error: str | None = Field(
        default=None,
        description="Safe error description if status is 'failed'.",
    )
