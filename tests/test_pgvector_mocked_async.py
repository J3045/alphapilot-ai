import asyncio
import importlib.util
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.document import ResearchDocument, TextChunk
from app.models.retrieval import RetrievalResult
from app.services.rag_service import RagService
from app.services.retrieval_service import (
    EmbeddingDimensionMismatchError,
    FakeEmbeddingClient,
    RetrievalService,
    VectorStoreConnectionError,
    cosine_similarity,
)


class _SpyEmbeddingClient(FakeEmbeddingClient):
    """Deterministic embedding client tracking embed_documents/embed_query invocations."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.doc_embed_calls: int = 0
        self.query_embed_calls: int = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_embed_calls += 1
        return await super().aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.query_embed_calls += 1
        return await super().aembed_query(text)


class _StatefulFakeVectorStore:
    """In-process fake persistent AsyncVectorStore simulating durable storage across RagService restarts."""

    def __init__(
        self,
        embedding_dimension: int = 64,
        embedding_model: str = "fake-64d",
    ) -> None:
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self._documents: dict[str, ResearchDocument] = {}
        self._chunks: dict[str, tuple[TextChunk, list[float]]] = {}
        self.fail_on_operation: Exception | None = None

    async def aadd_chunks(
        self,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        documents: Sequence[ResearchDocument] | None = None,
        embedding_model: str | None = None,
    ) -> int:
        if self.fail_on_operation is not None:
            raise self.fail_on_operation
        if documents:
            for doc in documents:
                self._documents[doc.document_id] = doc
            doc_ids = {doc.document_id for doc in documents}
            new_ids = {c.chunk_id for c in chunks}
            stale = [
                cid
                for cid, (existing_chunk, _) in self._chunks.items()
                if existing_chunk.document_id in doc_ids and cid not in new_ids
            ]
            for cid in stale:
                del self._chunks[cid]

        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = (chunk, list(vec))
        return len(chunks)

    async def asearch(
        self,
        query_vector: Sequence[float],
        top_k: int = 3,
        embedding_model: str | None = None,
    ) -> list[RetrievalResult]:
        if self.fail_on_operation is not None:
            raise self.fail_on_operation
        scored = [
            RetrievalResult.from_chunk(chunk, cosine_similarity(query_vector, vec))
            for chunk, vec in self._chunks.values()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def acount(self) -> int:
        if self.fail_on_operation is not None:
            raise self.fail_on_operation
        return len(self._chunks)


def _sample_docs() -> list[ResearchDocument]:
    return [
        ResearchDocument(
            document_id="POL-CONC",
            title="Concentration Risk Guideline",
            content="Maximum single-equity concentration limit is 10 percent of net asset value.",
            metadata={"department": "Risk", "effective_year": 2026},
        ),
        ResearchDocument(
            document_id="POL-CASH",
            title="Cash Liquidity Buffer Policy",
            content="Portfolios must maintain a minimum 5 percent liquid cash reserve buffer requirement at all times.",
            metadata={"department": "Treasury", "effective_year": 2026},
        ),
    ]


class TestRagPersistentLifecycleMockedAsync:
    def test_restart_reuses_persisted_vectors_without_reembedding(self):
        async def _scenario():
            store = _StatefulFakeVectorStore(embedding_dimension=64, embedding_model="fake-64d")

            async def _respond(messages):
                prompt = str(messages[-1].content).lower()
                if "pol-cash" in prompt:
                    text = "Minimum liquid cash reserve buffer requirement is 5 percent [POL-CASH]."
                else:
                    text = "Maximum single-equity limit is 10 percent [POL-CONC]."
                return SimpleNamespace(
                    content=text,
                    usage_metadata={"input_tokens": 20, "output_tokens": 10},
                )

            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=_respond)

            # Startup 1: Empty persistent store triggers initial chunk ingestion & embedding
            embedder_1 = _SpyEmbeddingClient(dimension=64)
            rag_1 = RagService(
                llm_client=mock_llm,
                embedding_client=embedder_1,
                documents=_sample_docs(),
                storage_backend="postgres",
                vector_store=store,
            )
            res_1 = await rag_1.answer_question_async("What is the single-equity concentration limit?")
            assert res_1.status == "answered"
            assert embedder_1.doc_embed_calls == 1
            assert embedder_1.query_embed_calls == 1
            assert await store.acount() == 2
            assert res_1.sources[0].document_id == "POL-CONC"
            assert res_1.sources[0].score >= rag_1.min_relevance_score
            assert res_1.sources[0].metadata == {"department": "Risk", "effective_year": 2026}

            # Simulate process restart: brand new RagService & embedder pointed at same store
            embedder_2 = _SpyEmbeddingClient(dimension=64)
            rag_2 = RagService(
                llm_client=mock_llm,
                embedding_client=embedder_2,
                documents=_sample_docs(),
                storage_backend="postgres",
                vector_store=store,
            )
            res_2 = await rag_2.answer_question_async("What is the cash reserve buffer requirement?")
            assert res_2.status == "answered"
            # Crucially, documents were ALREADY persisted so embed_documents is NOT called again!
            assert embedder_2.doc_embed_calls == 0
            assert embedder_2.query_embed_calls == 1
            assert res_2.sources[0].document_id == "POL-CASH"
            assert res_2.sources[0].score >= rag_2.min_relevance_score
            assert res_2.sources[0].metadata == {"department": "Treasury", "effective_year": 2026}
            assert mock_llm.ainvoke.await_count == 2

            # Regression guard: unrelated low-similarity query still returns insufficient_evidence
            # on restarted service without invoking the LLM
            res_low = await rag_2.answer_question_async(
                "What is the margin leverage limit on Solana staking?"
            )
            assert res_low.status == "insufficient_evidence"
            assert res_low.sources == []
            assert mock_llm.ainvoke.await_count == 2

            # Regression guard: strict embedding dimension check remains enforced on restart
            rag_mismatch = RagService(
                llm_client=mock_llm,
                embedding_client=_SpyEmbeddingClient(dimension=32),
                documents=_sample_docs(),
                storage_backend="postgres",
                vector_store=store,
            )
            res_mismatch = await rag_mismatch.answer_question_async("What is the cash reserve?")
            assert res_mismatch.status == "failed"
            assert "dimension or model mismatch" in (res_mismatch.error or "").lower()
            assert mock_llm.ainvoke.await_count == 2

        asyncio.run(_scenario())

    def test_db_failure_fails_closed_without_in_memory_fallback(self):
        async def _scenario():
            store = _StatefulFakeVectorStore(embedding_dimension=64)
            store.fail_on_operation = VectorStoreConnectionError(
                "postgresql+asyncpg://postgres:SecretDBPass@db:5432/alphapilot unreachable"
            )
            rag = RagService(
                llm_client=MagicMock(),
                embedding_client=FakeEmbeddingClient(dimension=64),
                documents=_sample_docs(),
                storage_backend="postgres",
                vector_store=store,
            )
            res = await rag.answer_question_async("What is the concentration limit?")
            assert res.status == "failed"
            assert res.answer == ""
            assert "SecretDBPass" not in (res.error or "")
            assert "connection failed" in (res.error or "").lower()

        asyncio.run(_scenario())

    def test_concurrent_indexing_and_retrieval_against_async_store(self):
        async def _scenario():
            store = _StatefulFakeVectorStore(embedding_dimension=64)
            retrieval = RetrievalService(
                embedding_client=FakeEmbeddingClient(dimension=64),
                vector_store=store,
                expected_dimension=64,
            )
            docs = _sample_docs()
            chunks = [
                TextChunk(
                    chunk_id=f"{d.document_id}_chunk_0",
                    document_id=d.document_id,
                    document_title=d.title,
                    chunk_index=0,
                    chunk_text=d.content,
                    metadata=d.metadata,
                )
                for d in docs
            ]
            # Concurrent duplicate upserts should remain idempotent
            counts = await asyncio.gather(
                retrieval.aindex_chunks(chunks, documents=docs),
                retrieval.aindex_chunks(chunks, documents=docs),
            )
            assert counts == [2, 2]
            assert await store.acount() == 2

            queries = [
                retrieval.aretrieve("single-equity concentration limit", top_k=1)
                for _ in range(8)
            ]
            batched = await asyncio.gather(*queries)
            assert len(batched) == 8
            for result_list in batched:
                assert len(result_list) == 1
                assert result_list[0].document_id == "POL-CONC"

        asyncio.run(_scenario())


class TestSqlAlchemySchemaMigrationsAndRepositoryMocked:
    def test_orm_schema_constraints_and_hnsw_index(self):
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        from app.db.models import DocumentChunkRecord, ResearchDocumentRecord

        doc_table = ResearchDocumentRecord.__table__
        chunk_table = DocumentChunkRecord.__table__

        assert set(doc_table.c.keys()) >= {
            "document_id",
            "title",
            "content",
            "metadata",
            "created_at",
            "updated_at",
        }
        assert set(chunk_table.c.keys()) >= {
            "chunk_id",
            "document_id",
            "document_title",
            "chunk_index",
            "chunk_text",
            "metadata",
            "embedding_model",
            "embedding_dimension",
            "embedding",
            "created_at",
            "updated_at",
        }

        fks = list(chunk_table.foreign_keys)
        assert len(fks) == 1
        assert fks[0].target_fullname == "research_documents.document_id"
        assert fks[0].ondelete == "CASCADE"

        unique_names = {c.name for c in chunk_table.constraints}
        assert "uq_document_chunks_document_id_chunk_index" in unique_names

        indexes_by_name = {idx.name: idx for idx in chunk_table.indexes}
        assert "ix_document_chunks_embedding_hnsw" in indexes_by_name
        hnsw_idx = indexes_by_name["ix_document_chunks_embedding_hnsw"]
        assert hnsw_idx.dialect_options["postgresql"]["using"] == "hnsw"
        assert hnsw_idx.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}

    def test_alembic_initial_migration_creates_extension_tables_and_downgrades(self, monkeypatch):
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        pytest.importorskip("alembic")

        migration_path = (
            Path(__file__).resolve().parent.parent
            / "alembic"
            / "versions"
            / "0001_create_pgvector_rag_tables.py"
        )
        spec = importlib.util.spec_from_file_location("mig_0001", migration_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_op = MagicMock()
        monkeypatch.setattr(mod, "op", mock_op)

        mod.upgrade()
        mock_op.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")
        created_tables = [call.args[0] for call in mock_op.create_table.call_args_list]
        assert created_tables == ["research_documents", "document_chunks"]
        created_indexes = [call.args[0] for call in mock_op.create_index.call_args_list]
        assert "ix_document_chunks_embedding_hnsw" in created_indexes

        mod.downgrade()
        dropped_tables = [call.args[0] for call in mock_op.drop_table.call_args_list]
        assert dropped_tables == ["document_chunks", "research_documents"]

    def test_database_manager_pool_health_check_and_missing_extension_guard(self, monkeypatch):
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        from app.db import session as db_session_mod

        fake_conn = MagicMock()
        fake_conn.execute = AsyncMock(
            side_effect=[
                MagicMock(),
                MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
            ]
        )

        @asynccontextmanager
        async def _fake_connect():
            yield fake_conn

        fake_engine = MagicMock()
        fake_engine.connect = _fake_connect
        fake_engine.dispose = AsyncMock()
        create_engine_mock = MagicMock(return_value=fake_engine)
        monkeypatch.setattr(db_session_mod, "create_async_engine", create_engine_mock)

        manager = db_session_mod.DatabaseManager(
            database_url="postgresql://postgres:TopSecretPw@localhost:5432/alphapilot",
            pool_size=7,
            max_overflow=3,
            pool_timeout_seconds=12.0,
        )
        assert "TopSecretPw" not in manager.redacted_url
        assert create_engine_mock.call_args.kwargs["pool_size"] == 7
        assert create_engine_mock.call_args.kwargs["max_overflow"] == 3
        assert create_engine_mock.call_args.kwargs["pool_pre_ping"] is True

        async def _run():
            from app.services.retrieval_service import VectorStoreError

            with pytest.raises(VectorStoreError, match="pgvector extension is not enabled"):
                await manager.verify_connection_and_pgvector()
            await manager.dispose()
            fake_engine.dispose.assert_awaited_once()

        asyncio.run(_run())

    def test_postgres_repository_crud_search_duplicate_and_error_translation(self):
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        from sqlalchemy.exc import OperationalError
        from app.db.repository import PostgresVectorRepository

        async def _run():
            # 1. Mock session for search ordering, metadata preservation, and empty case
            mock_session = MagicMock()
            mock_row = SimpleNamespace(
                chunk_id="DOC-1_chunk_0",
                document_id="DOC-1",
                document_title="Risk Policy",
                chunk_index=0,
                chunk_text="Equity concentration capped at 10%.",
                chunk_metadata={"source": "risk"},
                similarity_score=0.9123456,
            )
            mock_session.execute = AsyncMock(
                side_effect=[
                    MagicMock(all=MagicMock(return_value=[mock_row])),
                    MagicMock(all=MagicMock(return_value=[])),
                    MagicMock(
                        first=MagicMock(
                            return_value=SimpleNamespace(
                                document_id="DOC-1",
                                title="Risk Policy",
                                doc_metadata={"source": "risk"},
                                chunk_count=2,
                                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                            )
                        )
                    ),
                    MagicMock(rowcount=1),
                ]
            )

            @asynccontextmanager
            async def _begin():
                yield

            mock_session.begin = _begin

            @asynccontextmanager
            async def _factory():
                yield mock_session

            repo = PostgresVectorRepository(
                session_factory=_factory,
                embedding_dimension=4,
                embedding_model="fake-4d",
            )

            hits = await repo.asearch([1.0, 0.0, 0.0, 0.0], top_k=3)
            assert len(hits) == 1
            assert hits[0].chunk_id == "DOC-1_chunk_0"
            assert hits[0].score == 0.912346
            assert hits[0].metadata == {"source": "risk"}

            empty_hits = await repo.asearch([1.0, 0.0, 0.0, 0.0], top_k=3)
            assert empty_hits == []

            meta = await repo.get_document_metadata("DOC-1")
            assert meta is not None
            assert meta["chunk_count"] == 2
            assert meta["metadata"] == {"source": "risk"}

            assert await repo.delete_document("DOC-1") is True

            # 2. Verify dimension conflict detection on upsert against existing DB rows
            conflict_session = MagicMock()
            conflict_session.begin = _begin
            conflict_session.execute = AsyncMock(
                return_value=MagicMock(first=MagicMock(return_value=("other-model", 1536)))
            )

            @asynccontextmanager
            async def _conflict_factory():
                yield conflict_session

            conflict_repo = PostgresVectorRepository(
                session_factory=_conflict_factory,
                embedding_dimension=4,
                embedding_model="fake-4d",
            )
            doc = _sample_docs()[0]
            chunk = TextChunk(
                chunk_id="POL-CONC_chunk_0",
                document_id="POL-CONC",
                document_title=doc.title,
                chunk_index=0,
                chunk_text=doc.content,
                metadata=doc.metadata,
            )
            with pytest.raises(EmbeddingDimensionMismatchError, match="incompatible"):
                await conflict_repo.upsert_document_chunks(
                    documents=[doc],
                    chunks=[chunk],
                    vectors=[[1.0, 0.0, 0.0, 0.0]],
                )

            # 3. Verify SQLAlchemy driver failure translates safely to VectorStoreConnectionError
            failing_session = MagicMock()
            failing_session.execute = AsyncMock(
                side_effect=OperationalError(
                    "SELECT 1",
                    {},
                    Exception("connection refused postgres://postgres:LeakPw@db:5432/app"),
                )
            )

            @asynccontextmanager
            async def _failing_factory():
                yield failing_session

            failing_repo = PostgresVectorRepository(
                session_factory=_failing_factory,
                embedding_dimension=4,
                embedding_model="fake-4d",
            )
            with pytest.raises(VectorStoreConnectionError) as exc_info:
                await failing_repo.acount()
            assert "LeakPw" not in str(exc_info.value)

        asyncio.run(_run())

