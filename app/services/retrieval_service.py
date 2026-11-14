# pyrefly: ignore[missing-import]
import asyncio
import logging
import math
import re
import threading
from typing import NamedTuple, Protocol, Sequence, runtime_checkable

from app.models.document import ResearchDocument, TextChunk
from app.models.retrieval import RetrievalResult
from app.services.resilience import supports_async_method

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


class VectorStoreError(RetrievalError):
    """Base exception for persistent vector storage failures."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when persistent vector storage cannot be reached or is unhealthy."""
    pass


class EmbeddingDimensionMismatchError(VectorStoreError):
    """Raised when embedding vectors do not match expected model or schema dimension."""
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

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async interface matching OpenAIEmbeddings.aembed_documents for deterministic tests."""
        return [self._embed(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Async interface matching OpenAIEmbeddings.aembed_query for deterministic tests."""
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

def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
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
    vector: tuple[float, ...]


class InMemoryVectorIndex:
    """Minimal in-memory vector index suitable for MVP semantic retrieval.

    Stores chunks and their embedding vectors in memory and performs linear
    exact nearest-neighbor search via cosine similarity.

    Concurrency & Process Boundaries:
    - Protected by an internal RLock so concurrent searches snapshot entries safely
      without blocking cosine similarity calculations against each other.
    - Stored vectors are converted to immutable tuples on write.
    - State is strictly process-local: multiple Uvicorn workers maintain separate copies.

    Design rationale for MVP:
    - Zero external infrastructure (no Chroma, Pinecone, Qdrant, or Postgres required).
    - Fully deterministic and transparent.
    - Zero network dependencies, instant startup, and fast execution for MVP workloads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[_IndexEntry] = []

    def add(self, chunk: TextChunk, vector: Sequence[float]) -> None:
        """Add a single chunk and its embedding vector to the index."""
        entry = _IndexEntry(chunk=chunk, vector=tuple(vector))
        with self._lock:
            self._entries.append(entry)

    def add_batch(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Add a batch of chunks and corresponding embedding vectors atomically."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Count mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)})."
            )
        new_entries = [
            _IndexEntry(chunk=chunk, vector=tuple(vector))
            for chunk, vector in zip(chunks, vectors)
        ]
        with self._lock:
            self._entries.extend(new_entries)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
    ) -> list[tuple[TextChunk, float]]:
        """Search for the top-k most similar chunks by cosine similarity.

        Takes a brief lock only to snapshot entries, then executes cosine math
        lock-free so concurrent readers do not serialize on scoring.

        Returns:
            A list of (chunk, similarity_score) tuples, ordered descending by score.
        """
        with self._lock:
            if not self._entries:
                return []
            entries_snapshot = tuple(self._entries)

        scored_entries: list[tuple[TextChunk, float]] = []
        for entry in entries_snapshot:
            score = cosine_similarity(query_vector, entry.vector)
            scored_entries.append((entry.chunk, score))

        # Sort descending by score
        scored_entries.sort(key=lambda item: item[1], reverse=True)
        return scored_entries[:top_k]

    def clear(self) -> None:
        """Remove all entries from the index."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Shared Vector Validation & Async Vector Store Protocol
# ---------------------------------------------------------------------------

def validate_embedding_vector(
    vector: Sequence[float],
    expected_dimension: int,
) -> list[float]:
    """Validate that an embedding vector matches expected_dimension and contains finite floats."""
    if expected_dimension <= 0:
        raise ValueError(f"expected_dimension must be positive, got {expected_dimension}.")
    if len(vector) != expected_dimension:
        raise EmbeddingDimensionMismatchError(
            f"Embedding vector dimension mismatch: expected {expected_dimension}, got {len(vector)}."
        )

    validated: list[float] = []
    for idx, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VectorStoreError(
                f"Embedding vector element at index {idx} is not a numeric value."
            )
        float_val = float(value)
        if not math.isfinite(float_val):
            raise VectorStoreError(
                f"Embedding vector element at index {idx} must be finite (got {float_val})."
            )
        validated.append(float_val)
    return validated


def validate_chunk_vector_batch(
    chunks: Sequence[TextChunk],
    vectors: Sequence[Sequence[float]],
    expected_dimension: int | None = None,
) -> tuple[int, list[list[float]]]:
    """Validate batch alignment, positive dimension parity, and finite float values."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Count mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)})."
        )
    if not chunks:
        dim = expected_dimension if expected_dimension is not None else 0
        return dim, []

    target_dim = expected_dimension if expected_dimension is not None else len(vectors[0])
    if target_dim <= 0:
        raise EmbeddingDimensionMismatchError(
            f"Embedding dimension must be positive, got {target_dim}."
        )

    validated_vectors = [
        validate_embedding_vector(vector=vec, expected_dimension=target_dim)
        for vec in vectors
    ]
    return target_dim, validated_vectors


@runtime_checkable
class AsyncVectorStore(Protocol):
    """Protocol for persistent asynchronous vector storage backends (e.g. PostgreSQL + pgvector)."""

    embedding_dimension: int
    embedding_model: str

    async def aadd_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        documents: Sequence[ResearchDocument] | None = None,
        embedding_model: str | None = None,
    ) -> int:
        """Persist chunks and their embedding vectors."""
        ...

    async def asearch(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        embedding_model: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top-k similar RetrievalResult items ranked descending by similarity."""
        ...

    async def acount(self) -> int:
        """Return total persisted chunk count."""
        ...


# ---------------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------------

class RetrievalService:
    """Service that indexes document chunks and retrieves top-k relevant evidence.

    Supports both the default process-local InMemoryVectorIndex (for offline dev/tests)
    and an injected persistent AsyncVectorStore (for PostgreSQL + pgvector).
    Maintains complete separation of concerns from FastAPI routes and the LLM agent graph.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_index: InMemoryVectorIndex | None = None,
        vector_store: AsyncVectorStore | None = None,
        embedding_model: str | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        if vector_store is not None:
            self.vector_index = vector_index
            self.embedding_model = embedding_model or getattr(vector_store, "embedding_model", None)
            self.expected_dimension = (
                expected_dimension
                if expected_dimension is not None
                else getattr(vector_store, "embedding_dimension", None)
            )
        else:
            self.vector_index = vector_index if vector_index is not None else InMemoryVectorIndex()
            self.embedding_model = embedding_model
            self.expected_dimension = expected_dimension

    def _require_in_memory_index(self) -> InMemoryVectorIndex:
        if self.vector_index is None:
            raise VectorStoreError(
                "Synchronous in-memory index operation is unavailable when an async vector_store is configured; "
                "use async retrieval/indexing methods instead."
            )
        return self.vector_index

    def _store_indexed_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        index = self._require_in_memory_index()
        index.add_batch(chunks, vectors)
        logger.info(
            "chunks_indexed",
            extra={
                "indexed_count": len(chunks),
                "total_index_size": len(index),
                "backend": "in_memory",
            },
        )
        return len(chunks)

    def _validate_top_k_and_query(self, query: str, top_k: int) -> str | None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        if not query or not query.strip():
            logger.debug("retrieval_skipped_blank_query")
            return None
        return query.strip()

    def _validate_retrieve_query(self, query: str, top_k: int) -> str | None:
        clean_query = self._validate_top_k_and_query(query=query, top_k=top_k)
        if clean_query is None:
            return None

        index = self._require_in_memory_index()
        # Handle empty index safely without invoking embedding provider
        if len(index) == 0:
            logger.debug("retrieval_skipped_empty_index")
            return None

        return clean_query

    def _search_and_map(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        index = self._require_in_memory_index()
        raw_results = index.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        results = [
            RetrievalResult.from_chunk(chunk=chunk, score=score)
            for chunk, score in raw_results
        ]
        self._log_retrieval_executed(top_k=top_k, results=results, backend="in_memory")
        return results

    def _log_retrieval_executed(
        self,
        top_k: int,
        results: Sequence[RetrievalResult],
        backend: str,
    ) -> None:
        logger.info(
            "retrieval_executed",
            extra={
                "backend": backend,
                "top_k": top_k,
                "result_count": len(results),
                "top_score": results[0].score if results else None,
            },
        )

    async def _embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        try:
            if supports_async_method(self.embedding_client, "aembed_documents"):
                return await self.embedding_client.aembed_documents(texts)  # type: ignore[attr-defined]
            return await asyncio.to_thread(self.embedding_client.embed_documents, texts)
        except Exception as exc:
            logger.error("retrieval_index_embedding_failed: %s", exc)
            raise EmbeddingProviderError(
                f"Embedding provider failed while indexing document chunks: {exc}"
            ) from exc

    async def _embed_query_async(self, clean_query: str) -> list[float]:
        try:
            if supports_async_method(self.embedding_client, "aembed_query"):
                return await self.embedding_client.aembed_query(clean_query)  # type: ignore[attr-defined]
            return await asyncio.to_thread(self.embedding_client.embed_query, clean_query)
        except Exception as exc:
            logger.error("retrieval_query_embedding_failed: %s", exc)
            raise EmbeddingProviderError(
                f"Embedding provider failed while processing query: {exc}"
            ) from exc

    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        """Embed and index a sequence of TextChunk objects synchronously (in-memory only).

        Args:
            chunks: Sequence of TextChunk objects to embed and store.

        Returns:
            The number of chunks successfully indexed.

        Raises:
            EmbeddingProviderError: If the embedding client fails.
            VectorStoreError: If an async-only persistent store is configured.
        """
        self._require_in_memory_index()
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

        if self.expected_dimension is not None:
            _, validated_vectors = validate_chunk_vector_batch(
                chunks=chunks,
                vectors=vectors,
                expected_dimension=self.expected_dimension,
            )
        else:
            validated_vectors = vectors

        return self._store_indexed_chunks(chunks, validated_vectors)

    async def aindex_chunks(
        self,
        chunks: Sequence[TextChunk],
        documents: Sequence[ResearchDocument] | None = None,
    ) -> int:
        """Asynchronously embed and index a sequence of TextChunk objects."""
        if not chunks:
            logger.info("retrieval_index_empty_chunks")
            return 0

        texts = [chunk.chunk_text for chunk in chunks]
        vectors = await self._embed_documents_async(texts)

        if self.expected_dimension is not None:
            _, validated_vectors = validate_chunk_vector_batch(
                chunks=chunks,
                vectors=vectors,
                expected_dimension=self.expected_dimension,
            )
        else:
            validated_vectors = vectors

        if self.vector_store is not None:
            count = await self.vector_store.aadd_chunks(
                chunks=chunks,
                vectors=validated_vectors,
                documents=documents,
                embedding_model=self.embedding_model,
            )
            logger.info(
                "chunks_indexed",
                extra={
                    "indexed_count": count,
                    "backend": "postgres",
                },
            )
            return count

        return self._store_indexed_chunks(chunks, validated_vectors)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks synchronously (in-memory only)."""
        clean_query = self._validate_retrieve_query(query=query, top_k=top_k)
        if clean_query is None:
            return []

        try:
            query_vector = self.embedding_client.embed_query(clean_query)
        except Exception as exc:
            logger.error("retrieval_query_embedding_failed: %s", exc)
            raise EmbeddingProviderError(
                f"Embedding provider failed while processing query: {exc}"
            ) from exc

        if self.expected_dimension is not None:
            query_vector = validate_embedding_vector(
                vector=query_vector,
                expected_dimension=self.expected_dimension,
            )

        return self._search_and_map(query_vector=query_vector, top_k=top_k)

    async def aretrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Asynchronously retrieve the top-k most relevant chunks for a user search query."""
        if self.vector_store is not None:
            clean_query = self._validate_top_k_and_query(query=query, top_k=top_k)
            if clean_query is None:
                return []

            if await self.vector_store.acount() == 0:
                logger.debug("retrieval_skipped_empty_index", extra={"backend": "postgres"})
                return []

            query_vector = await self._embed_query_async(clean_query)
            if self.expected_dimension is not None:
                query_vector = validate_embedding_vector(
                    vector=query_vector,
                    expected_dimension=self.expected_dimension,
                )

            results = await self.vector_store.asearch(
                query_vector=query_vector,
                top_k=top_k,
                embedding_model=self.embedding_model,
            )
            self._log_retrieval_executed(top_k=top_k, results=results, backend="postgres")
            return results

        clean_query = self._validate_retrieve_query(query=query, top_k=top_k)
        if clean_query is None:
            return []

        query_vector = await self._embed_query_async(clean_query)
        if self.expected_dimension is not None:
            query_vector = validate_embedding_vector(
                vector=query_vector,
                expected_dimension=self.expected_dimension,
            )

        return self._search_and_map(query_vector=query_vector, top_k=top_k)
