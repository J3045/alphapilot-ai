import pytest
# pyrefly: ignore[missing-import]
from pydantic import ValidationError

from app.models.document import ResearchDocument, TextChunk
from app.services.document_ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocumentIngestionService,
    chunk_document,
    chunk_text,
    ingest_documents,
    validate_chunk_parameters,
)
from app.services.sample_documents import get_sample_documents


def sample_single_document() -> ResearchDocument:
    return ResearchDocument(
        document_id="TEST-DOC-001",
        title="Concentration Limits Test Document",
        content=(
            "Internal risk guidelines establish that no single equity position "
            "shall exceed 20.0 percent of total portfolio market value. "
            "Managers have three business days to restore compliance if a breach occurs."
        ),
        metadata={
            "source": "internal_risk_handbook.pdf",
            "section": "2.1",
            "synthetic": True,
        },
    )


# 1. One document ingested successfully
def test_one_document_ingested_successfully():
    doc = sample_single_document()
    chunks = ingest_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "TEST-DOC-001"
        assert chunk.document_title == "Concentration Limits Test Document"
        assert chunk.title == "Concentration Limits Test Document"
        assert len(chunk.chunk_text) > 0
        assert chunk.text == chunk.chunk_text
        assert chunk.metadata["source"] == "internal_risk_handbook.pdf"
        assert chunk.metadata["synthetic"] is True

    # Check indices are strictly 0-indexed and sequential
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# 2. Multiple documents
def test_multiple_documents_ingested_successfully():
    docs = get_sample_documents()
    assert len(docs) == 3

    chunks = ingest_documents(docs, chunk_size=300, chunk_overlap=50)

    # All three document IDs must be present in the output chunks
    doc_ids_in_chunks = {c.document_id for c in chunks}
    expected_doc_ids = {d.document_id for d in docs}
    assert doc_ids_in_chunks == expected_doc_ids

    # Document ordering must be preserved: doc1 chunks, then doc2, then doc3
    seen_ids: list[str] = []
    for chunk in chunks:
        if not seen_ids or seen_ids[-1] != chunk.document_id:
            seen_ids.append(chunk.document_id)
    assert seen_ids == [d.document_id for d in docs]


# 3. Chunk boundaries and overlap
def test_chunk_boundaries_and_overlap():
    text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    doc = ResearchDocument(
        document_id="BOUND-DOC",
        title="Boundary Verification",
        content=text,
    )

    chunk_size = 10
    chunk_overlap = 3
    # Step = 7
    # Chunk 0: text[0:10]   = '0123456789'
    # Chunk 1: text[7:17]   = '789ABCDEFG'  (overlap with 0 is '789')
    # Chunk 2: text[14:24]  = 'DEFG...    ' (overlap with 1 is 'DEFG'[:3] == 'DEF')
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) >= 3
    assert chunks[0].chunk_text == "0123456789"
    assert chunks[1].chunk_text == "789ABCDEFG"

    # Verify exact overlap characters between consecutive chunks
    for i in range(len(chunks) - 1):
        tail_of_current = chunks[i].chunk_text[-chunk_overlap:]
        head_of_next = chunks[i + 1].chunk_text[:chunk_overlap]
        assert tail_of_current == head_of_next


# 4. Metadata preservation
def test_metadata_preservation():
    metadata = {
        "source": "audit_policy_2026.pdf",
        "category": "risk_review",
        "author": "Chief Compliance Officer",
        "version": 3,
        "tags": ["governance", "synthetic"],
    }
    doc = ResearchDocument(
        document_id="META-DOC",
        title="Metadata Preservation Test",
        content="Short synthetic policy text for testing metadata retention.",
        metadata=metadata,
    )

    chunks = ingest_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata == metadata
        # Ensure deep copy: modifying chunk metadata does not mutate parent document
        chunk.metadata["extra_key"] = "test"
        assert "extra_key" not in doc.metadata


# 5. Deterministic ordering and repeated runs
def test_deterministic_ordering_and_repeated_runs():
    docs = get_sample_documents()

    run1 = ingest_documents(docs, chunk_size=250, chunk_overlap=50)
    run2 = ingest_documents(docs, chunk_size=250, chunk_overlap=50)

    assert len(run1) == len(run2)
    assert len(run1) > 0

    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.document_id == c2.document_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.chunk_text == c2.chunk_text
        assert c1.metadata == c2.metadata


# 6. Empty collection
def test_empty_collection():
    assert ingest_documents([]) == []
    service = DocumentIngestionService()
    assert service.ingest([]) == []


# 7. Empty or whitespace-only document content
def test_empty_or_whitespace_document_content():
    # Empty content
    empty_doc = ResearchDocument(
        document_id="EMPTY-DOC",
        title="Empty Document",
        content="",
    )
    assert chunk_document(empty_doc) == []
    assert ingest_documents([empty_doc]) == []

    # Whitespace-only content
    whitespace_doc = ResearchDocument(
        document_id="SPACE-DOC",
        title="Whitespace Document",
        content="   \n\t  \r\n   ",
    )
    assert chunk_document(whitespace_doc) == []
    assert ingest_documents([whitespace_doc]) == []

    # Mixed collection with empty doc and valid doc
    valid_doc = sample_single_document()
    mixed_result = ingest_documents([empty_doc, valid_doc, whitespace_doc])
    assert len(mixed_result) > 0
    # Every returned chunk must be non-empty and belong to valid_doc
    for chunk in mixed_result:
        assert chunk.document_id == valid_doc.document_id
        assert len(chunk.chunk_text.strip()) > 0


# 8. Invalid chunk size and overlap
def test_invalid_chunk_size_and_overlap():
    # Chunk size <= 0
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunk_parameters(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        validate_chunk_parameters(chunk_size=-10, chunk_overlap=0)

    # Chunk overlap < 0
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=-1)

    # Chunk overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        validate_chunk_parameters(chunk_size=100, chunk_overlap=150)

    # Service constructor validation
    with pytest.raises(ValueError):
        DocumentIngestionService(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError):
        DocumentIngestionService(chunk_size=100, chunk_overlap=100)


# 9. Long documents producing multiple chunks
def test_long_documents_producing_multiple_chunks():
    # Construct a 1,200 character document
    long_content = "AlphaPilot risk review section paragraph. " * 30
    assert len(long_content) > 1000

    doc = ResearchDocument(
        document_id="LONG-DOC",
        title="Extended Risk Guidelines",
        content=long_content,
    )

    chunk_size = 200
    chunk_overlap = 40
    chunks = ingest_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 5
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"LONG-DOC_chunk_{i}"
        assert len(chunk.chunk_text) <= chunk_size


# 10. No infinite loop when text is shorter than or equal to chunk size
def test_no_infinite_loop_when_text_shorter_than_or_equal_to_chunk_size():
    # Text shorter than chunk size
    short_doc = ResearchDocument(
        document_id="SHORT-DOC",
        title="Short Content",
        content="Short text.",
    )
    short_chunks = chunk_document(short_doc, chunk_size=100, chunk_overlap=20)
    assert len(short_chunks) == 1
    assert short_chunks[0].chunk_text == "Short text."
    assert short_chunks[0].chunk_index == 0

    # Text exactly equal to chunk size
    exact_text = "x" * 100
    exact_doc = ResearchDocument(
        document_id="EXACT-DOC",
        title="Exact Length Content",
        content=exact_text,
    )
    exact_chunks = chunk_document(exact_doc, chunk_size=100, chunk_overlap=20)
    assert len(exact_chunks) == 1
    assert exact_chunks[0].chunk_text == exact_text
    assert exact_chunks[0].chunk_index == 0


# 11. Model validation rejects invalid data predictably
def test_research_document_validation():
    # Empty or whitespace document_id rejected
    with pytest.raises(ValidationError):
        ResearchDocument(document_id="", title="Valid Title", content="Content")

    with pytest.raises(ValidationError):
        ResearchDocument(document_id="   ", title="Valid Title", content="Content")

    # Empty or whitespace title rejected
    with pytest.raises(ValidationError):
        ResearchDocument(document_id="DOC-1", title="", content="Content")

    with pytest.raises(ValidationError):
        ResearchDocument(document_id="DOC-1", title="   ", content="Content")

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        ResearchDocument(
            document_id="DOC-1",
            title="Valid Title",
            content="Content",
            unauthorized_field="malicious",
        )


def test_text_chunk_validation():
    # Negative chunk index rejected
    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="DOC-1_chunk_0",
            document_id="DOC-1",
            document_title="Title",
            chunk_index=-1,
            chunk_text="Valid chunk text",
        )

    # Empty or whitespace chunk text rejected
    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="DOC-1_chunk_0",
            document_id="DOC-1",
            document_title="Title",
            chunk_index=0,
            chunk_text="",
        )

    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="DOC-1_chunk_0",
            document_id="DOC-1",
            document_title="Title",
            chunk_index=0,
            chunk_text="   ",
        )

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        TextChunk(
            chunk_id="DOC-1_chunk_0",
            document_id="DOC-1",
            document_title="Title",
            chunk_index=0,
            chunk_text="Valid text",
            unknown_extra="forbidden",
        )


# 12. Service class wrapper
def test_document_ingestion_service_wrapper():
    service = DocumentIngestionService(chunk_size=300, chunk_overlap=50)
    doc = sample_single_document()

    chunks1 = service.ingest([doc])
    chunks2 = service.ingest_document(doc)

    assert len(chunks1) == len(chunks2)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
    assert chunks1[0].chunk_text == chunks2[0].chunk_text


# 13. Sample documents load and chunk properly
def test_sample_documents_validity():
    sample_docs = get_sample_documents()
    assert len(sample_docs) == 3

    for doc in sample_docs:
        assert doc.document_id.startswith("SYNTH-DOC-")
        assert "Synthetic" in doc.title
        assert doc.metadata.get("synthetic") is True
        assert len(doc.content) > 100

    chunks = ingest_documents(sample_docs)
    assert len(chunks) > len(sample_docs)  # Multi-chunk ingestion
    for chunk in chunks:
        assert chunk.metadata.get("synthetic") is True
