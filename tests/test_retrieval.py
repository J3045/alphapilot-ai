from unittest.mock import MagicMock
import pytest
# pyrefly: ignore[missing-import]
from pydantic import ValidationError

from app.models.document import ResearchDocument, TextChunk
from app.models.retrieval import RetrievalResult
from app.services.document_ingestion import ingest_documents
from app.services.retrieval_service import (
    EmbeddingProviderError,
    FakeEmbeddingClient,
    InMemoryVectorIndex,
    RetrievalService,
    cosine_similarity,
    get_openai_embedding_client,
)
from app.services.sample_documents import get_sample_documents


@pytest.fixture
def fake_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient(dimension=64)


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    docs = get_sample_documents()
    return ingest_documents(docs, chunk_size=300, chunk_overlap=50)


@pytest.fixture
def populated_service(fake_client: FakeEmbeddingClient, sample_chunks: list[TextChunk]) -> RetrievalService:
    service = RetrievalService(embedding_client=fake_client)
    service.index_chunks(sample_chunks)
    return service


# 1. Indexing and retrieving relevant synthetic chunks
def test_indexing_and_retrieving_relevant_synthetic_chunks(
    populated_service: RetrievalService,
):
    # Query for cash allocation / liquidity policy
    cash_results = populated_service.retrieve(
        query="minimum cash reserve buffer liquidity requirements",
        top_k=3,
    )
    assert len(cash_results) == 3
    # Top result should be from SYNTH-DOC-002 (Cash Reserve & Liquidity Standards)
    assert cash_results[0].document_id == "SYNTH-DOC-002"
    assert "Cash" in cash_results[0].document_title

    # Query for concentration policy
    concentration_results = populated_service.retrieve(
        query="portfolio concentration limits and single security ceiling breach",
        top_k=3,
    )
    assert len(concentration_results) == 3
    assert concentration_results[0].document_id == "SYNTH-DOC-001"
    assert "Concentration" in concentration_results[0].document_title

    # Query for escalation protocol
    escalation_results = populated_service.retrieve(
        query="governance committee review escalation path audit",
        top_k=3,
    )
    assert len(escalation_results) == 3
    assert escalation_results[0].document_id == "SYNTH-DOC-003"
    assert "Escalation" in escalation_results[0].document_title


# 2. Correct result ordering
def test_correct_result_ordering(populated_service: RetrievalService):
    results = populated_service.retrieve("portfolio concentration limit risk", top_k=5)
    assert len(results) > 1

    # Verify strictly descending order of scores
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


# 3. Metadata preservation
def test_metadata_preservation(fake_client: FakeEmbeddingClient):
    custom_metadata = {
        "source": "special_audit.pdf",
        "category": "compliance",
        "author": "Head of Risk",
        "version": 4,
        "synthetic": True,
    }
    doc = ResearchDocument(
        document_id="AUDIT-DOC-99",
        title="Special Audit Guidelines",
        content="Mandatory special compliance guidelines for automated portfolio monitoring.",
        metadata=custom_metadata,
    )
    chunks = ingest_documents([doc], chunk_size=200, chunk_overlap=30)
    service = RetrievalService(embedding_client=fake_client)
    service.index_chunks(chunks)

    results = service.retrieve("compliance audit monitoring", top_k=1)
    assert len(results) == 1
    result = results[0]

    assert result.document_id == "AUDIT-DOC-99"
    assert result.document_title == "Special Audit Guidelines"
    assert result.title == "Special Audit Guidelines"
    assert result.chunk_index == 0
    assert "compliance" in result.chunk_text.lower()
    assert result.text == result.chunk_text
    assert result.metadata == custom_metadata
    assert isinstance(result.score, float)

    # Mutating result metadata does not affect underlying chunk
    result.metadata["new_key"] = "tampered"
    assert "new_key" not in chunks[0].metadata


# 4. Empty index behavior
def test_empty_index_behavior(fake_client: FakeEmbeddingClient):
    empty_service = RetrievalService(embedding_client=fake_client)
    assert len(empty_service.vector_index) == 0

    results = empty_service.retrieve("any query about cash or concentration", top_k=3)
    assert results == []


# 5. Invalid queries and top-k values
def test_blank_queries_return_empty_list(populated_service: RetrievalService):
    assert populated_service.retrieve("", top_k=3) == []
    assert populated_service.retrieve("   ", top_k=3) == []
    assert populated_service.retrieve("\n\t  ", top_k=3) == []


def test_invalid_top_k_values_raise_value_error(populated_service: RetrievalService):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        populated_service.retrieve("valid query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        populated_service.retrieve("valid query", top_k=-1)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        populated_service.retrieve("valid query", top_k="five")  # type: ignore[arg-type]


# 6. Embedding failures
def test_embedding_failure_on_indexing_raises_domain_error():
    mock_client = MagicMock()
    mock_client.embed_documents.side_effect = RuntimeError("OpenAI API unreachable: 503")

    service = RetrievalService(embedding_client=mock_client)
    sample_chunk = TextChunk(
        chunk_id="C1",
        document_id="D1",
        document_title="Title",
        chunk_index=0,
        chunk_text="Some text",
    )

    with pytest.raises(EmbeddingProviderError, match="Embedding provider failed while indexing"):
        service.index_chunks([sample_chunk])


def test_embedding_failure_on_query_raises_domain_error():
    mock_client = MagicMock()
    mock_client.embed_query.side_effect = TimeoutError("OpenAI embedding request timed out")

    index = InMemoryVectorIndex()
    index.add(
        chunk=TextChunk(
            chunk_id="C1",
            document_id="D1",
            document_title="Title",
            chunk_index=0,
            chunk_text="Some text",
        ),
        vector=[0.1] * 16,
    )
    service = RetrievalService(embedding_client=mock_client, vector_index=index)

    with pytest.raises(EmbeddingProviderError, match="Embedding provider failed while processing query"):
        service.retrieve("what are the rules?", top_k=3)


# 7. Deterministic behavior using a fake embedding provider
def test_deterministic_behavior_using_fake_embedding_provider(
    sample_chunks: list[TextChunk],
):
    client1 = FakeEmbeddingClient(dimension=64)
    service1 = RetrievalService(embedding_client=client1)
    service1.index_chunks(sample_chunks)

    client2 = FakeEmbeddingClient(dimension=64)
    service2 = RetrievalService(embedding_client=client2)
    service2.index_chunks(sample_chunks)

    query = "cash reserve requirements and staged execution"
    results1 = service1.retrieve(query, top_k=3)
    results2 = service2.retrieve(query, top_k=3)

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.chunk_id == r2.chunk_id
        assert r1.document_id == r2.document_id
        assert r1.score == r2.score
        assert r1.chunk_text == r2.chunk_text


# 8. Top-k limiting and edge bounds
def test_top_k_limiting_bounds(populated_service: RetrievalService):
    results_top_1 = populated_service.retrieve("cash policy", top_k=1)
    assert len(results_top_1) == 1

    results_top_2 = populated_service.retrieve("cash policy", top_k=2)
    assert len(results_top_2) == 2

    # Requesting more results than available returns all available without error
    total_chunks = len(populated_service.vector_index)
    results_overflow = populated_service.retrieve("cash policy", top_k=100)
    assert len(results_overflow) == total_chunks


# 9. Vector math and index unit tests
def test_cosine_similarity_accuracy():
    # Identical vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1e-6) == 1.0

    # Orthogonal vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 1e-6) == 0.0

    # Opposite vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 1e-6) == -1.0

    # Zero magnitude vector handled safely without division by zero
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    # Dimension mismatch raises ValueError
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_in_memory_vector_index_add_batch_mismatch():
    index = InMemoryVectorIndex()
    chunk = TextChunk(
        chunk_id="C1",
        document_id="D1",
        document_title="Title",
        chunk_index=0,
        chunk_text="Text",
    )
    with pytest.raises(ValueError, match="Count mismatch"):
        index.add_batch([chunk], [[0.1], [0.2]])


def test_in_memory_vector_index_clear():
    index = InMemoryVectorIndex()
    chunk = TextChunk(
        chunk_id="C1",
        document_id="D1",
        document_title="Title",
        chunk_index=0,
        chunk_text="Text",
    )
    index.add(chunk, [0.5, 0.5])
    assert len(index) == 1
    index.clear()
    assert len(index) == 0


# 10. OpenAI embedding client factory safety
def test_get_openai_embedding_client_raises_cleanly_when_missing_key(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", None)

    with pytest.raises(ValueError, match="OpenAI API key is not configured"):
        get_openai_embedding_client()


# 11. Empty chunks indexing
def test_index_empty_chunks(fake_client: FakeEmbeddingClient):
    service = RetrievalService(embedding_client=fake_client)
    assert service.index_chunks([]) == 0
    assert len(service.vector_index) == 0


# 12. RetrievalResult model validation
def test_retrieval_result_validation():
    # Negative chunk index rejected
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk_id="C1",
            document_id="D1",
            document_title="Title",
            chunk_index=-1,
            chunk_text="Text",
            score=0.9,
        )

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk_id="C1",
            document_id="D1",
            document_title="Title",
            chunk_index=0,
            chunk_text="Text",
            score=0.9,
            unauthorized="extra",
        )
