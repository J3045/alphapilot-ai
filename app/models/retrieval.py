# pyrefly: ignore[missing-import]
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.models.document import TextChunk


class RetrievalResult(BaseModel):
    """Result model returned by the semantic retrieval service.

    Preserves parent document identifiers, document title, chunk index, text,
    and source metadata alongside the similarity relevance score.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Deterministic unique identifier for the chunk.",
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
    score: float = Field(
        ...,
        description="Similarity relevance score between the query and chunk vector.",
    )

    @property
    def title(self) -> str:
        """Alias for document_title for citation and formatting convenience."""
        return self.document_title

    @property
    def text(self) -> str:
        """Alias for chunk_text for convenience."""
        return self.chunk_text

    @classmethod
    def from_chunk(cls, chunk: TextChunk, score: float) -> "RetrievalResult":
        """Factory constructor building a RetrievalResult from a TextChunk and score."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
            metadata=dict(chunk.metadata),
            score=round(score, 6),
        )
