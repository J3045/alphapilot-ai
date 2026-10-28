# ADR 003: Modular Separation in the RAG Pipeline

## Status
Accepted

## Context
RAG systems frequently degrade into monolithic scripts where document ingestion, text splitting, embedding generation, vector indexing, similarity search, and answer synthesis are tangled together. This complicates unit testing, prevents offline evaluation, and blocks modular optimization.

## Decision
We architect the RAG pipeline into distinct, single-responsibility components:
1. **Document Ingestion (`app.services.document_ingestion`)**: Handles document loading, character chunking, and metadata preservation without touching embeddings or APIs.
2. **Embedding Client Protocol (`app.services.retrieval_service.EmbeddingClient`)**: Decouples vector generation using a Python `Protocol`, enabling swapping between `ChatOpenAI`, local models, or deterministic `FakeEmbeddingClient`.
3. **In-Memory Vector Index (`InMemoryVectorIndex`)**: Responsible solely for vector storage and cosine similarity scoring.
4. **Retrieval Service (`RetrievalService`)**: Bridges the query embedding and the index to return ranked `RetrievalResult` objects.
5. **Answer Generation (`RagService`)**: Constructs isolated prompts, enforces untrusted context boundaries, and invokes the LLM.

## Benefits
- **Zero-Cost Offline Evaluation**: Entire evaluation suites run in milliseconds using `FakeEmbeddingClient` without OpenAI API keys or charges.
- **Independent Optimization**: Chunk size, overlap, relevance threshold, and prompt wording can each be evaluated independently.
- **Defensive Boundaries**: Untrusted documents can be sanitized and tagged before reaching prompt synthesis.

## Trade-offs
- More files and boilerplate classes compared to an all-in-one helper script.

## Alternatives Considered
- *All-in-one RAG Framework (e.g., standard LangChain RetrievalQA)*: Masked inner token counts, hid prompt boundaries, and made deterministic offline evaluation difficult.

## Conditions for Revisit
If structured document types (PDFs with tables, Excel files, SEC EDGAR XBRL filings) are added, specialized ingestion parsers should be plugged in without rewriting the retrieval or generation services.
