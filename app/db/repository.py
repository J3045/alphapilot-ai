# pyrefly: ignore[missing-import]
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import logging
from typing import Any, AsyncIterator, Callable, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import DocumentChunkRecord, ResearchDocumentRecord
from app.logging_config import redact_sensitive_text
from app.models.document import ResearchDocument, TextChunk
from app.models.retrieval import RetrievalResult
from app.services.retrieval_service import (
    EmbeddingDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
    validate_chunk_vector_batch,
    validate_embedding_vector,
)

logger = logging.getLogger(__name__)

_INSERT_BATCH_SIZE = 500
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class PostgresVectorRepository:
    """Async PostgreSQL + pgvector persistence repository for RAG documents and chunks.

    Isolates SQL, transaction management, upsert/idempotency, and cosine similarity
    search from RAG orchestration and FastAPI routes.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        embedding_dimension: int | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.embedding_dimension = (
            embedding_dimension if embedding_dimension is not None else settings.embedding_dimension
        )
        self.embedding_model = embedding_model or settings.openai_embedding_model

        if self.embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {self.embedding_dimension}."
            )

    @asynccontextmanager
    async def _db_operation(self, operation: str) -> AsyncIterator[AsyncSession]:
        """Acquire a session and translate driver/SQLAlchemy errors safely."""
        try:
            async with self._session_factory() as session:
                yield session
        except VectorStoreError:
            raise
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            safe_detail = redact_sensitive_text(str(exc))
            logger.error(
                "postgres_repository_sql_error",
                extra={
                    "operation": operation,
                    "error_type": type(exc).__name__,
                    "detail": safe_detail[:300],
                },
            )
            raise self._classify_db_exception(exc, operation=operation) from exc
        except (OSError, TimeoutError) as exc:
            logger.error(
                "postgres_repository_connection_error",
                extra={"operation": operation, "error_type": type(exc).__name__},
            )
            raise VectorStoreConnectionError(
                f"Database connection error during '{operation}' ({type(exc).__name__})."
            ) from exc
        except Exception as exc:
            logger.error(
                "postgres_repository_unexpected_error",
                extra={"operation": operation, "error_type": type(exc).__name__},
            )
            raise VectorStoreError(
                f"Unexpected vector repository error during '{operation}' ({type(exc).__name__})."
            ) from exc

    @staticmethod
    def _classify_db_exception(exc: Exception, operation: str) -> VectorStoreError:
        exc_msg = str(exc).lower()
        exc_name = type(exc).__name__.lower()
        connection_markers = (
            "connection",
            "connect",
            "timeout",
            "refused",
            "closed",
            "operational",
            "interface",
        )
        if any(marker in exc_msg or marker in exc_name for marker in connection_markers):
            return VectorStoreConnectionError(
                f"PostgreSQL connection failure during '{operation}' ({type(exc).__name__})."
            )
        return VectorStoreError(
            f"PostgreSQL vector operation failed during '{operation}' ({type(exc).__name__})."
        )

    def _resolve_model(self, embedding_model: str | None) -> str:
        target_model = embedding_model or self.embedding_model
        if target_model != self.embedding_model:
            raise EmbeddingDimensionMismatchError(
                f"Embedding model mismatch: repository is configured for '{self.embedding_model}', "
                f"received '{target_model}'."
            )
        return target_model

    async def _assert_compatible_existing_vectors(
        self,
        session: AsyncSession,
        target_model: str,
    ) -> None:
        stmt = (
            select(
                DocumentChunkRecord.embedding_model,
                DocumentChunkRecord.embedding_dimension,
            )
            .where(
                (DocumentChunkRecord.embedding_model != target_model)
                | (DocumentChunkRecord.embedding_dimension != self.embedding_dimension)
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        conflict = result.first()
        if conflict is not None:
            existing_model, existing_dim = conflict
            raise EmbeddingDimensionMismatchError(
                f"Existing stored vectors use model='{existing_model}' dimension={existing_dim}, "
                f"incompatible with configured model='{target_model}' dimension={self.embedding_dimension}."
            )

    async def upsert_document_chunks(
        self,
        documents: Sequence[ResearchDocument] | None,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        embedding_model: str | None = None,
    ) -> int:
        """Atomically persist documents and chunk embeddings with idempotent upsert semantics."""
        if not chunks:
            return 0

        target_model = self._resolve_model(embedding_model)
        _, validated_vectors = validate_chunk_vector_batch(
            chunks=chunks,
            vectors=vectors,
            expected_dimension=self.embedding_dimension,
        )

        async with self._db_operation("upsert_document_chunks") as session:
            async with session.begin():
                await self._assert_compatible_existing_vectors(session, target_model)

                if documents:
                    doc_rows = [
                        {
                            "document_id": doc.document_id,
                            "title": doc.title,
                            "content": doc.content,
                            "metadata": dict(doc.metadata),
                        }
                        for doc in documents
                    ]
                    doc_stmt = pg_insert(ResearchDocumentRecord).values(doc_rows)
                    doc_stmt = doc_stmt.on_conflict_do_update(
                        index_elements=["document_id"],
                        set_={
                            "title": doc_stmt.excluded.title,
                            "content": doc_stmt.excluded.content,
                            "metadata": doc_stmt.excluded.metadata,
                            "updated_at": func.now(),
                        },
                    )
                    await session.execute(doc_stmt)

                    doc_ids = sorted({doc.document_id for doc in documents})
                    new_chunk_ids = [chunk.chunk_id for chunk in chunks]
                    stale_delete_stmt = delete(DocumentChunkRecord).where(
                        DocumentChunkRecord.document_id.in_(doc_ids),
                        ~DocumentChunkRecord.chunk_id.in_(new_chunk_ids),
                    )
                    await session.execute(stale_delete_stmt)
                else:
                    # Ensure parent research_documents rows exist for standalone chunk writes
                    seen_docs: dict[str, str] = {}
                    for chunk in chunks:
                        seen_docs.setdefault(chunk.document_id, chunk.document_title)
                    stub_rows = [
                        {
                            "document_id": doc_id,
                            "title": title,
                            "content": "",
                            "metadata": {},
                        }
                        for doc_id, title in seen_docs.items()
                    ]
                    stub_stmt = pg_insert(ResearchDocumentRecord).values(stub_rows)
                    stub_stmt = stub_stmt.on_conflict_do_nothing(
                        index_elements=["document_id"]
                    )
                    await session.execute(stub_stmt)

                chunk_rows = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "chunk_index": chunk.chunk_index,
                        "chunk_text": chunk.chunk_text,
                        "metadata": dict(chunk.metadata),
                        "embedding_model": target_model,
                        "embedding_dimension": self.embedding_dimension,
                        "embedding": vector,
                    }
                    for chunk, vector in zip(chunks, validated_vectors)
                ]

                for start in range(0, len(chunk_rows), _INSERT_BATCH_SIZE):
                    batch = chunk_rows[start : start + _INSERT_BATCH_SIZE]
                    chunk_stmt = pg_insert(DocumentChunkRecord).values(batch)
                    chunk_stmt = chunk_stmt.on_conflict_do_update(
                        index_elements=["chunk_id"],
                        set_={
                            "document_id": chunk_stmt.excluded.document_id,
                            "document_title": chunk_stmt.excluded.document_title,
                            "chunk_index": chunk_stmt.excluded.chunk_index,
                            "chunk_text": chunk_stmt.excluded.chunk_text,
                            "metadata": chunk_stmt.excluded.metadata,
                            "embedding_model": chunk_stmt.excluded.embedding_model,
                            "embedding_dimension": chunk_stmt.excluded.embedding_dimension,
                            "embedding": chunk_stmt.excluded.embedding,
                            "updated_at": func.now(),
                        },
                    )
                    await session.execute(chunk_stmt)

        return len(chunks)

    async def aadd_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        documents: Sequence[ResearchDocument] | None = None,
        embedding_model: str | None = None,
    ) -> int:
        """AsyncVectorStore protocol adapter delegating to transactional upsert."""
        return await self.upsert_document_chunks(
            documents=documents,
            chunks=chunks,
            vectors=vectors,
            embedding_model=embedding_model,
        )

    async def asearch(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        embedding_model: str | None = None,
    ) -> list[RetrievalResult]:
        """Search PostgreSQL + pgvector for top-k chunks ordered by cosine similarity."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        target_model = self._resolve_model(embedding_model)
        validated_query = validate_embedding_vector(
            vector=query_vector,
            expected_dimension=self.embedding_dimension,
        )

        async with self._db_operation("search_similar") as session:
            distance_expr = DocumentChunkRecord.embedding.cosine_distance(validated_query)
            stmt = (
                select(
                    DocumentChunkRecord.chunk_id,
                    DocumentChunkRecord.document_id,
                    DocumentChunkRecord.document_title,
                    DocumentChunkRecord.chunk_index,
                    DocumentChunkRecord.chunk_text,
                    DocumentChunkRecord.chunk_metadata,
                    (1.0 - distance_expr).label("similarity_score"),
                )
                .where(
                    DocumentChunkRecord.embedding_model == target_model,
                    DocumentChunkRecord.embedding_dimension == self.embedding_dimension,
                )
                .order_by(distance_expr.asc())
                .limit(top_k)
            )
            result = await session.execute(stmt)
            rows = result.all()

        return [
            RetrievalResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_title=row.document_title,
                chunk_index=row.chunk_index,
                chunk_text=row.chunk_text,
                metadata=dict(row.chunk_metadata or {}),
                score=round(float(row.similarity_score), 6),
            )
            for row in rows
        ]

    async def get_document_metadata(self, document_id: str) -> dict[str, Any] | None:
        """Retrieve document metadata and persisted chunk count by document_id."""
        clean_id = document_id.strip()
        if not clean_id:
            raise ValueError("document_id cannot be empty.")

        async with self._db_operation("get_document_metadata") as session:
            stmt = (
                select(
                    ResearchDocumentRecord.document_id,
                    ResearchDocumentRecord.title,
                    ResearchDocumentRecord.doc_metadata,
                    ResearchDocumentRecord.created_at,
                    ResearchDocumentRecord.updated_at,
                    func.count(DocumentChunkRecord.chunk_id).label("chunk_count"),
                )
                .outerjoin(
                    DocumentChunkRecord,
                    DocumentChunkRecord.document_id == ResearchDocumentRecord.document_id,
                )
                .where(ResearchDocumentRecord.document_id == clean_id)
                .group_by(ResearchDocumentRecord.document_id)
            )
            result = await session.execute(stmt)
            row = result.first()
            if row is None:
                return None

            return {
                "document_id": row.document_id,
                "title": row.title,
                "metadata": dict(row.doc_metadata or {}),
                "chunk_count": int(row.chunk_count or 0),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }

    async def delete_document(self, document_id: str) -> bool:
        """Delete a research document and cascade-delete all associated chunks."""
        clean_id = document_id.strip()
        if not clean_id:
            raise ValueError("document_id cannot be empty.")

        async with self._db_operation("delete_document") as session:
            async with session.begin():
                stmt = delete(ResearchDocumentRecord).where(
                    ResearchDocumentRecord.document_id == clean_id
                )
                result = await session.execute(stmt)
                return bool(result.rowcount and result.rowcount > 0)

    async def acount(self) -> int:
        """Return total number of stored chunk vectors."""
        async with self._db_operation("count_chunks") as session:
            stmt = select(func.count(DocumentChunkRecord.chunk_id))
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

    async def count_documents(self) -> int:
        """Return total number of stored research documents."""
        async with self._db_operation("count_documents") as session:
            stmt = select(func.count(ResearchDocumentRecord.document_id))
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

