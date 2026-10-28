# ADR 004: In-Memory Vector Index Rationale and Limitations

## Status
Accepted (with documented migration triggers)

## Context
A vector retrieval mechanism is required to match user questions against research policy chunks. Introducing external infrastructure (Pinecone, ChromaDB, Qdrant, Milvus, or PostgreSQL with pgvector) adds operational overhead, cloud credentials, network latency, and CI/CD complexity for an MVP with tens to hundreds of synthetic document chunks.

## Decision
We implemented a minimal, transparent `InMemoryVectorIndex` using Python lists and pure cosine similarity math (`app.services.retrieval_service`). Chunks and vectors reside in process heap memory.

## Benefits
- **Zero Infrastructure & Instant Startup**: Tests run in milliseconds without external Docker services, cloud accounts, or database migrations.
- **Complete Mathematical Transparency**: Cosine similarity is calculated directly without opaque C-extensions or proprietary similarity thresholds.
- **Portability**: The codebase runs consistently on developer workstations, CI pipelines, and cloud runners.

## Trade-offs & Limitations
- **Lack of Persistence**: The index is rebuilt in memory per instance lifecycle. Process restarts require re-indexing.
- **Linear Scan ($\mathcal{O}(N \cdot D)$)**: Searches every chunk sequentially. While fast for $N < 1,000$, search latency degrades linearly as dataset size increases.
- **Multi-Process Concurrency**: Multiple Uvicorn worker processes do not share index state, causing duplicated memory consumption.

## Alternatives Considered
- *Local Chroma / SQLite*: Adds binary SQLite compilation requirements and disk state management in tests.
- *Managed Cloud Vector DB*: Violates local offline development goals and introduces recurring costs.

## Conditions for Revisit
Migrate to PostgreSQL with `pgvector` or an ANN vector database when:
1. Document volume exceeds 10,000 chunks.
2. In-memory indexing at startup takes $> 5$ seconds.
3. Multi-worker or multi-container horizontal scaling requires shared persistence.
