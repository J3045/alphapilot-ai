# pyrefly: ignore[missing-import]
import logging
import math
import re
from typing import NamedTuple, Protocol, Sequence, runtime_checkable

from app.models.document import TextChunk
from app.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RetrievalError(Exception):
    """Base exception for all retrieval and vector index errors."""
    pass


class EmbeddingProviderError(RetrievalError):
    """Raised when an embedding provider fails or is unreachable."""
    pass


# ---------------------------------------------------------------------------
# Embedding Provider Protocol & Implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol defining the embedding provider interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate float embedding vectors for a list of document chunk texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate a float embedding vector for a single search query."""
        ...


# Common English stopwords and synthetic boilerplate words to ignore in test embeddings
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "it", "this", "that", "are", "be", "as", "not", "all",
    "synthetic", "test", "document", "disclaimer", "does", "text", "testing",
}


class FakeEmbeddingClient:
    """Deterministic embedding client for testing without external API calls or charges.

    Transforms text into a fixed-dimension vector using deterministic token hashing
    and L2 unit normalization. Texts with matching words share vector dimensions,
    yielding realistic similarity scores for unit tests.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension

        vec = [0.0] * self.dimension
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        for token in tokens:
            if token in _STOP_WORDS:
                continue
            # Polynomial rolling hash
            h = 0
            for char in token:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            idx = h % self.dimension
            vec[idx] += 1.0

        # L2 normalize vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_openai_embedding_client(
    api_key: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> EmbeddingClient:
    """Create an OpenAI embedding client using the configured settings.

    Never exposes API keys in logs or exceptions.

    Raises:
        ValueError: If no OpenAI API key is configured.
        EmbeddingProviderError: If the provider fails to initialize.
    """
    from app.config import settings

    key = api_key
    if key is None and settings.openai_api_key is not None:
        key = settings.openai_api_key.get_secret_value()

    if not key:
        raise ValueError(
            "OpenAI API key is not configured. Set OPENAI_API_KEY in the environment or .env file."
        )

    model_name = model or settings.openai_embedding_model
    timeout_sec = timeout or settings.openai_timeout_seconds

    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            openai_api_key=key,
            model=model_name,
            request_timeout=timeout_sec,
        )
    except Exception as exc:
        raise EmbeddingProviderError(
            f"Failed to initialize OpenAIEmbeddings provider: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Vector Math & In-Memory Vector Index
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine similarity between two float vectors.

    Returns:
        A float value between -1.0 and 1.0. Returns 0.0 if either vector has zero magnitude.

    Raises:
        ValueError: If vector lengths do not match.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class _IndexEntry(NamedTuple):
    chunk: TextChunk
    vector: list[float]


class InMemoryVectorIndex:
    """Minimal in-memory vector index suitable for MVP semantic retrieval.

    Stores chunks and their embedding vectors in memory and performs linear
    exact nearest-neighbor search via cosine similarity.

    Design rationale for MVP:
    - Zero external infrastructure (no Chroma, Pinecone, Qdrant, or Postgres required).
    - Fully deterministic and transparent.
    - Zero network dependencies, instant startup, and fast execution for MVP workloads.
    """

    def __init__(self) -> None:
        self._entries: list[_IndexEntry] = []

    def add(self, chunk: TextChunk, vector: list[float]) -> None:
        """Add a single chunk and its embedding vector to the index."""
        self._entries.append(_IndexEntry(chunk=chunk, vector=vector))

    def add_batch(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[list[float]],
    ) -> None:
        """Add a batch of chunks and corresponding embedding vectors."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Count mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)})."
            )
        for chunk, vector in zip(chunks, vectors):
            self.add(chunk, vector)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
    ) -> list[tuple[TextChunk, float]]:
        """Search for the top-k most similar chunks by cosine similarity.

        Returns:
            A list of (chunk, similarity_score) tuples, ordered descending by score.
        """
        if not self._entries:
            return []

        scored_entries: list[tuple[TextChunk, float]] = []
        for entry in self._entries:
            score = cosine_similarity(query_vector, entry.vector)
            scored_entries.append((entry.chunk, score))

        # Sort descending by score
        scored_entries.sort(key=lambda item: item[1], reverse=True)
        return scored_entries[:top_k]

    def clear(self) -> None:
        """Remove all entries from the index."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------------

class RetrievalService:
    """Service that indexes document chunks and retrieves top-k relevant evidence.

    Maintains complete separation of concerns from FastAPI routes and the LLM agent graph.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_index: InMemoryVectorIndex | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_index = vector_index if vector_index is not None else InMemoryVectorIndex()

    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        """Embed and index a sequence of TextChunk objects.

        Args:
            chunks: Sequence of TextChunk objects to embed and store.

        Returns:
            The number of chunks successfully indexed.

        Raises:
            EmbeddingProviderError: If the embedding client fails.
        """
        if not chunks:
            logger.info("retrieval_index_empty_chunks")
            return 0

        texts = [chunk.chunk_text for chunk in chunks]

        try:
            vectors = self.embedding_client.embed_documents(texts)
        except Exception as exc:
            logger.error("retrieval_index_embedding_failed: %s", exc)
            raise EmbeddingProviderError(
                f"Embedding provider failed while indexing document chunks: {exc}"
            ) from exc

        self.vector_index.add_batch(chunks, vectors)

        logger.info(
            "chunks_indexed",
            extra={
                "indexed_count": len(chunks),
                "total_index_size": len(self.vector_index),
            },
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a user search query.

        Args:
            query: The search query string.
            top_k: Number of most relevant results to return (default: 3).

        Returns:
            An ordered list of RetrievalResult objects ranked by descending relevance score.

        Raises:
            ValueError: If top_k is not a positive integer.
            EmbeddingProviderError: If the embedding client fails to embed the query.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        # Handle empty index safely without invoking embedding provider
        if len(self.vector_index) == 0:
            logger.debug("retrieval_skipped_empty_index")
            return []

        # Handle blank query safely without invoking embedding provider
        if not query or not query.strip():
            logger.debug("retrieval_skipped_blank_query")
            return []

        try:
            query_vector = self.embedding_client.embed_query(query.strip())
        except Exception as exc:
            logger.error("retrieval_query_embedding_failed: %s", exc)
            raise EmbeddingProviderError(
                f"Embedding provider failed while processing query: {exc}"
            ) from exc

        raw_results = self.vector_index.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        results = [
            RetrievalResult.from_chunk(chunk=chunk, score=score)
            for chunk, score in raw_results
        ]

        logger.info(
            "retrieval_executed",
            extra={
                "top_k": top_k,
                "result_count": len(results),
                "top_score": results[0].score if results else None,
            },
        )

        return results
