# pyrefly: ignore[missing-import]
import logging
from typing import Sequence

from app.models.document import ResearchDocument, TextChunk

logger = logging.getLogger(__name__)

# Sensible default parameters for character-based chunking:
# - DEFAULT_CHUNK_SIZE: 500 characters corresponds to ~75–100 words in English,
#   which encapsulates a coherent, standalone policy rule or paragraph.
# - DEFAULT_CHUNK_OVERLAP: 100 characters (~15–20 words, ~20% overlap) ensures
#   critical context (e.g. exceptions, caveats, thresholds) spanning chunk boundaries
#   is not lost during retrieval.
#
# NOTE ON STRATEGY:
# Character-based windowing is intentionally implemented as a deterministic, lightweight,
# and dependency-free baseline for Hour 6. A known limitation is that character-based
# chunking may split words or sentences mid-phrase. Advanced tokenization or semantic/sentence
# chunking will be considered in future iterations.
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap parameters.

    Args:
        chunk_size: The character window size for chunking. Must be a positive integer.
        chunk_overlap: The character overlap between consecutive chunks. Must be
            non-negative and strictly smaller than chunk_size.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size}."
        )
    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap must be non-negative, got {chunk_overlap}."
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than "
            f"chunk_size ({chunk_size})."
        )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split raw text into overlapping character-based chunks.

    Args:
        text: The raw text string to divide into chunks.
        chunk_size: Maximum character length per chunk (default: 500).
        chunk_overlap: Overlapping character count between adjacent chunks (default: 100).

    Returns:
        An ordered list of non-empty text chunk strings.

    Raises:
        ValueError: If chunking parameters are invalid.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)

    if not text or not text.strip():
        return []

    clean_text = text.strip()
    text_len = len(clean_text)
    step = chunk_size - chunk_overlap

    # step >= 1 is mathematically guaranteed because chunk_overlap < chunk_size.
    # Therefore, 'start' strictly increases every iteration, preventing infinite loops.
    chunks: list[str] = []
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = clean_text[start:end]

        if chunk_str.strip():
            chunks.append(chunk_str)

        if end >= text_len:
            break
        start += step

    return chunks


def chunk_document(
    document: ResearchDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Convert a single ResearchDocument into an ordered list of TextChunk objects.

    Preserves parent document identifiers, document title, and source metadata.
    Handles empty or whitespace-only documents gracefully by returning an empty list.

    Args:
        document: The input ResearchDocument to chunk.
        chunk_size: Maximum character length per chunk (default: 500).
        chunk_overlap: Overlapping character count between adjacent chunks (default: 100).

    Returns:
        An ordered list of TextChunk objects with deterministic 0-based indices.

    Raises:
        ValueError: If chunking parameters are invalid.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)

    if not document.content or not document.content.strip():
        logger.debug(
            "document_skipped_empty_or_whitespace",
            extra={"document_id": document.document_id},
        )
        return []

    text_slices = chunk_text(
        text=document.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[TextChunk] = []
    for idx, slice_text in enumerate(text_slices):
        chunk = TextChunk(
            chunk_id=f"{document.document_id}_chunk_{idx}",
            document_id=document.document_id,
            document_title=document.title,
            chunk_index=idx,
            chunk_text=slice_text,
            metadata=dict(document.metadata),
        )
        chunks.append(chunk)

    return chunks


def ingest_documents(
    documents: Sequence[ResearchDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Ingest a collection of ResearchDocument objects and convert them into TextChunks.

    Args:
        documents: A sequence of ResearchDocument objects to ingest.
        chunk_size: Maximum character length per chunk (default: 500).
        chunk_overlap: Overlapping character count between adjacent chunks (default: 100).

    Returns:
        An ordered list of TextChunk objects across all ingested documents.

    Raises:
        ValueError: If chunking parameters are invalid.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)

    if not documents:
        logger.info("ingest_documents_empty_collection")
        return []

    all_chunks: list[TextChunk] = []
    for doc in documents:
        doc_chunks = chunk_document(
            document=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(doc_chunks)

    # Log operational metadata only - never log raw document text or sensitive content
    logger.info(
        "documents_ingested",
        extra={
            "document_count": len(documents),
            "chunk_count": len(all_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    return all_chunks


class DocumentIngestionService:
    """Service encapsulating RAG document ingestion and chunking configuration.

    Maintains separation of concerns from FastAPI routes and agent workflows.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        validate_chunk_parameters(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, documents: Sequence[ResearchDocument]) -> list[TextChunk]:
        """Ingest a collection of research documents using the configured parameters."""
        return ingest_documents(
            documents=documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def ingest_document(self, document: ResearchDocument) -> list[TextChunk]:
        """Ingest a single research document using the configured parameters."""
        return chunk_document(
            document=document,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
