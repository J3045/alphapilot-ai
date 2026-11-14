import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.document import ResearchDocument, TextChunk
from app.services.document_ingestion import ingest_documents
from app.services.rag_service import RagService
from app.services.retrieval_service import (
    FakeEmbeddingClient,
    RetrievalService,
    VectorStoreConnectionError,
    cosine_similarity,
)
from tests.pg_test_utils import require_isolated_pg_test_url

_TEST_DOCUMENT_IDS = ("INT-RISK-001", "INT-LIQ-002")


def _pgvector_integration_skip_reason() -> str | None:
    try:
        require_isolated_pg_test_url(allow_legacy_pgvector_flag=True)
        return None
    except pytest.skip.Exception as exc:
        return str(exc.msg)
    except RuntimeError:
        return None


_SKIP_REASON = _pgvector_integration_skip_reason()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


class _Spy1536EmbeddingClient(FakeEmbeddingClient):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension=dimension)
        self.doc_embed_calls: int = 0
        self.query_embed_calls: int = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_embed_calls += 1
        return await super().aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.query_embed_calls += 1
        return await super().aembed_query(text)


def _build_real_test_docs() -> list[ResearchDocument]:
    return [
        ResearchDocument(
            document_id="INT-RISK-001",
            title="Equity Concentration Policy",
            content=(
                "No single equity holding may exceed 10 percent of total portfolio market value. "
                "Breaches require rebalancing within 5 trading days."
            ),
            metadata={"source": "risk_committee", "tier": "mandatory"},
        ),
        ResearchDocument(
            document_id="INT-LIQ-002",
            title="Cash Reserve Buffer Standard",
            content=(
                "Portfolios must maintain a minimum 5 percent allocation to liquid treasury cash reserves "
                "to cover redemptions without forced liquidation."
            ),
            metadata={"source": "treasury", "tier": "mandatory"},
        ),
    ]


@pytest.fixture(scope="module")
def migrated_database_url() -> str:
    from alembic import command
    from alembic.config import Config

    db_url = require_isolated_pg_test_url(allow_legacy_pgvector_flag=True)
    alembic_cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")
    return db_url


@pytest.fixture(autouse=True)
def clean_pgvector_tables(migrated_database_url: str):
    from sqlalchemy import delete
    from app.db.models import DocumentChunkRecord, ResearchDocumentRecord
    from app.db.session import DatabaseManager

    async def _cleanup_test_rows():
        manager = DatabaseManager(database_url=migrated_database_url)
        try:
            async with manager.session() as session:
                async with session.begin():
                    await session.execute(
                        delete(DocumentChunkRecord).where(
                            DocumentChunkRecord.document_id.in_(_TEST_DOCUMENT_IDS)
                        )
                    )
                    await session.execute(
                        delete(ResearchDocumentRecord).where(
                            ResearchDocumentRecord.document_id.in_(_TEST_DOCUMENT_IDS)
                        )
                    )
        finally:
            await manager.dispose()

    asyncio.run(_cleanup_test_rows())
    yield
    asyncio.run(_cleanup_test_rows())


class TestPostgresPgvectorIntegration:
    def test_real_pgvector_insert_duplicate_similarity_metadata_and_delete(
        self, migrated_database_url: str
    ):
        from app.db.repository import PostgresVectorRepository
        from app.db.session import DatabaseManager

        async def _scenario():
            dim = settings.embedding_dimension
            embedder = FakeEmbeddingClient(dimension=dim)
            manager = DatabaseManager(database_url=migrated_database_url)
            try:
                await manager.verify_connection_and_pgvector()
                repo = PostgresVectorRepository(
                    session_factory=manager.session,
                    embedding_dimension=dim,
                    embedding_model=settings.openai_embedding_model,
                )

                baseline_count = await repo.acount()

                docs = _build_real_test_docs()
                chunks = ingest_documents(docs)
                vectors = embedder.embed_documents([c.chunk_text for c in chunks])

                inserted = await repo.upsert_document_chunks(
                    documents=docs,
                    chunks=chunks,
                    vectors=vectors,
                )
                assert inserted == len(chunks)
                assert await repo.acount() == baseline_count + len(chunks)

                # Metadata preservation check
                meta = await repo.get_document_metadata("INT-RISK-001")
                assert meta is not None
                assert meta["title"] == "Equity Concentration Policy"
                assert meta["metadata"] == {"source": "risk_committee", "tier": "mandatory"}
                assert meta["chunk_count"] >= 1

                # Duplicate ingestion must be idempotent without creating duplicate chunk rows
                inserted_again = await repo.upsert_document_chunks(
                    documents=docs,
                    chunks=chunks,
                    vectors=vectors,
                )
                assert inserted_again == len(chunks)
                assert await repo.acount() == baseline_count + len(chunks)

                # Vector similarity ranking parity check against exact cosine similarity
                query_vec = embedder.embed_query("single equity holding exceed 10 percent rebalancing")
                hits = await repo.asearch(query_vec, top_k=2)
                assert len(hits) == 2
                assert hits[0].document_id == "INT-RISK-001"
                assert hits[0].score >= hits[1].score
                expected_top_score = round(cosine_similarity(query_vec, vectors[0]), 6)
                assert hits[0].score == pytest.approx(expected_top_score, abs=1e-4)

                # Deletion cascades to associated chunks
                deleted = await repo.delete_document("INT-RISK-001")
                assert deleted is True
                assert await repo.get_document_metadata("INT-RISK-001") is None
                assert await repo.acount() == baseline_count + sum(
                    1 for c in chunks if c.document_id == "INT-LIQ-002"
                )
            finally:
                await manager.dispose()

        asyncio.run(_scenario())

    def test_real_pgvector_restart_persistence_without_reembedding(
        self, migrated_database_url: str
    ):
        from app.db.repository import PostgresVectorRepository
        from app.db.session import DatabaseManager

        async def _scenario():
            dim = settings.embedding_dimension
            docs = _build_real_test_docs()
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                return_value=SimpleNamespace(
                    content="Portfolios must hold at least 5 percent liquid cash reserves [INT-LIQ-002].",
                    usage_metadata={"input_tokens": 25, "output_tokens": 12},
                )
            )

            # Boot 1: index & persist documents, then dispose pool cleanly (simulated shutdown)
            manager_1 = DatabaseManager(database_url=migrated_database_url)
            repo_1 = PostgresVectorRepository(
                session_factory=manager_1.session,
                embedding_dimension=dim,
                embedding_model=settings.openai_embedding_model,
            )
            embedder_1 = _Spy1536EmbeddingClient(dimension=dim)
            rag_1 = RagService(
                llm_client=mock_llm,
                embedding_client=embedder_1,
                documents=docs,
                storage_backend="postgres",
                vector_store=repo_1,
                database_manager=manager_1,
            )
            rag_1._owns_database_manager = True
            res_1 = await rag_1.answer_question_async("What is the cash reserve minimum?")
            assert res_1.status == "answered"
            assert embedder_1.doc_embed_calls == 1
            await rag_1.aclose()

            # Boot 2: fresh DatabaseManager + fresh RagService pointed at persisted DB
            manager_2 = DatabaseManager(database_url=migrated_database_url)
            repo_2 = PostgresVectorRepository(
                session_factory=manager_2.session,
                embedding_dimension=dim,
                embedding_model=settings.openai_embedding_model,
            )
            embedder_2 = _Spy1536EmbeddingClient(dimension=dim)
            rag_2 = RagService(
                llm_client=mock_llm,
                embedding_client=embedder_2,
                documents=docs,
                storage_backend="postgres",
                vector_store=repo_2,
                database_manager=manager_2,
            )
            rag_2._owns_database_manager = True
            res_2 = await rag_2.answer_question_async("What is the liquid treasury cash reserve?")
            assert res_2.status == "answered"
            assert embedder_2.doc_embed_calls == 0
            assert any(src.document_id == "INT-LIQ-002" for src in res_2.sources)
            await rag_2.aclose()

        asyncio.run(_scenario())

    def test_real_pgvector_concurrent_queries_and_unreachable_db_failure(
        self, migrated_database_url: str
    ):
        from app.db.repository import PostgresVectorRepository
        from app.db.session import DatabaseManager

        async def _scenario():
            dim = settings.embedding_dimension
            docs = _build_real_test_docs()
            chunks = ingest_documents(docs)
            embedder = FakeEmbeddingClient(dimension=dim)

            manager = DatabaseManager(database_url=migrated_database_url)
            try:
                repo = PostgresVectorRepository(
                    session_factory=manager.session,
                    embedding_dimension=dim,
                    embedding_model=settings.openai_embedding_model,
                )
                retrieval = RetrievalService(
                    embedding_client=embedder,
                    vector_store=repo,
                    expected_dimension=dim,
                )
                await retrieval.aindex_chunks(chunks, documents=docs)

                results = await asyncio.gather(
                    *[retrieval.aretrieve("10 percent single equity concentration", top_k=1) for _ in range(6)]
                )
                assert all(len(r) == 1 and r[0].document_id == "INT-RISK-001" for r in results)
            finally:
                await manager.dispose()

            # Bad database target must fail closed with VectorStoreConnectionError / status=failed
            bad_manager = DatabaseManager(
                database_url="postgresql+asyncpg://postgres:WrongPass@127.0.0.1:59999/nonexistent",
                pool_timeout_seconds=1.0,
            )
            bad_repo = PostgresVectorRepository(
                session_factory=bad_manager.session,
                embedding_dimension=dim,
                embedding_model=settings.openai_embedding_model,
            )
            rag_bad = RagService(
                llm_client=MagicMock(),
                embedding_client=embedder,
                documents=docs,
                storage_backend="postgres",
                vector_store=bad_repo,
                database_manager=bad_manager,
            )
            rag_bad._owns_database_manager = True
            failed_res = await rag_bad.answer_question_async("What is the cash buffer?")
            assert failed_res.status == "failed"
            assert "WrongPass" not in (failed_res.error or "")
            await rag_bad.aclose()

        asyncio.run(_scenario())

