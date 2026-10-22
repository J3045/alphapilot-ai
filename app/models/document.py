# pyrefly: ignore[missing-import]
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchDocument(BaseModel):
    """Input research document model for RAG ingestion.

    Represents a policy, guideline, or research document before chunking.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        min_length=1,
        description="Stable, unique identifier for the research document.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Non-empty title of the research document.",
    )
    content: str = Field(
        ...,
        description="Raw text content of the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional source metadata suitable for later citations.",
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("document_id cannot be empty or whitespace-only.")
        return stripped

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title cannot be empty or whitespace-only.")
        return stripped


class TextChunk(BaseModel):
    """Output text chunk model produced by the RAG document ingestion service.

    Preserves parent document identifiers and metadata for downstream retrieval and citations.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Deterministic unique identifier for the chunk (e.g. '{document_id}_chunk_{index}').",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the originating research document.",
    )
    document_title: str = Field(
        ...,
        min_length=1,
        description="Title of the originating research document.",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="0-based sequential index of the chunk within the document.",
    )
    chunk_text: str = Field(
        ...,
        min_length=1,
        description="Non-empty text content of this chunk.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved source metadata from the parent document.",
    )

    @property
    def title(self) -> str:
        """Alias for document_title for citation convenience."""
        return self.document_title

    @property
    def text(self) -> str:
        """Alias for chunk_text for convenience."""
        return self.chunk_text

    @field_validator("chunk_id", "document_id", "document_title")
    @classmethod
    def validate_identifier_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Identifier string cannot be empty or whitespace-only.")
        return stripped

    @field_validator("chunk_text")
    @classmethod
    def validate_chunk_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("chunk_text cannot be empty or whitespace-only.")
        return value
