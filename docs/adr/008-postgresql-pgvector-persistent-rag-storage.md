# ADR 008: Persistent RAG Vector Storage with PostgreSQL and pgvector

## Status
Accepted

## Context
Following Day 14, AlphaPilot AI executes LLM and embedding I/O asynchronously with per-worker bounded concurrency. However, the default `InMemoryVectorIndex` resides strictly in process RAM:
1. Document chunks and embeddings are lost on every process restart or deployment, requiring re-embedding via the OpenAI API.
2. Multiple Uvicorn worker processes (`--workers N`) maintain independent in-memory indexes, duplicating RAM and preventing shared dynamic document updates across workers or containers.
3. Relational document metadata (`document_id`, titles, JSON metadata, timestamps) lacked durable relational storage paired with vector similarity search.

## Decision
We introduced persistent vector storage backed by **PostgreSQL + `pgvector`**, integrated behind a clean dependency-injected repository and storage abstraction while keeping `InMemoryVectorIndex` intact for offline development and unit testing:

1. **Relational + Vector Persistence Layer (`app/db/`):**
   - Defined `ResearchDocumentRecord` (`research_documents`) and `DocumentChunkRecord` (`document_chunks`) using SQLAlchemy 2.0 async ORM (`asyncpg`) and `pgvector.sqlalchemy.Vector(embedding_dimension)`.
   - Configured `ON DELETE CASCADE` foreign key from chunks to parent documents, `UNIQUE (document_id, chunk_index)`, JSONB metadata columns, UTC timestamps, and an HNSW cosine-distance index (`vector_cosine_ops`) queried via parameterized `ORDER BY embedding <=> :query LIMIT :top_k`.
   - Managed engine creation, connection pooling (`pool_size`, `max_overflow`, `pool_timeout_seconds`, `pool_pre_ping`), `vector` extension verification, and clean FastAPI `lifespan` disposal inside `DatabaseManager`.
2. **Idempotent Repository & Dimension Integrity (`PostgresVectorRepository`):**
   - Encapsulated SQL, batch `ON CONFLICT` upserts, stale trailing chunk pruning, document metadata lookup, cascade deletion, and similarity ranking out of `RagService` and routes.
   - Enforced strict dimension and model compatibility checks (`EmbeddingDimensionMismatchError`) across `Settings`, `RetrievalService`, and stored chunk rows so incompatible embedding spaces are never mixed silently.
3. **Alembic Migrations & Local Docker Setup:**
   - Added async Alembic configuration (`alembic.ini`, `alembic/env.py`) reading `DATABASE_URL` dynamically (zero credentials committed) and initial migration `0001_create_pgvector_rag_tables.py` enabling `CREATE EXTENSION IF NOT EXISTS vector`.
   - Added `docker-compose.yml` using `pgvector/pgvector:pg16` with a persistent named volume.
4. **Backend Selection & Fail-Closed Guarantee:**
   - Added `RAG_STORAGE_BACKEND` (`in_memory` default for offline tests/evals, `postgres` for persistent deployments).
   - When `RAG_STORAGE_BACKEND=postgres`, `RagService` checks persisted chunk count on initialization (`acount() > 0`) to reuse stored vectors across restarts without re-embedding, and fails closed (`VectorStoreError` -> sanitized HTTP `503`) on DB errors instead of silently falling back to an empty `InMemoryVectorIndex`.
5. **Credential Redaction:**
   - Extended `redact_sensitive_text`, `JsonFormatter`, `Settings.get_redacted_database_url()`, and `sanitize_provider_error` to mask PostgreSQL URL passwords and DSN credentials before logging or returning error details.

## Benefits
- **Durability Across Restarts:** Chunk embeddings, text, and JSONB metadata survive application restarts and eliminate redundant cold-start embedding API charges.
- **Shared Multi-Worker State:** Multiple Uvicorn workers and application containers query a single consistent vector and metadata store.
- **Zero Impact on Offline Workflow:** Deterministic unit tests and the offline RAG evaluation runner continue to use `InMemoryVectorIndex` and `FakeEmbeddingClient` without needing a running PostgreSQL instance.
- **Operational Safety:** Parameterized SQL prevents injection, connection pooling bounds DB connections per worker, and fail-closed error handling avoids silent zero-evidence answers on DB outages.

## Trade-offs & Limitations
- **Approximate Nearest Neighbor (HNSW) Recall vs. Speed:** HNSW trades sub-millisecond/low-millisecond search latency for tiny approximate recall variance compared to exact brute-force scans (`InMemoryVectorIndex`).
- **Additional Infrastructure Dependency in Production:** Production deployments enabling `postgres` must provision PostgreSQL 16+ with the `pgvector` extension installed and run `alembic upgrade head` prior to serving traffic.
- **Schema Bound Dimension:** `document_chunks.embedding` fixes `VECTOR(dimension)` per migration (`1536` default for `text-embedding-3-small`); changing to a different dimension model requires an explicit schema migration and re-indexing.

## Alternatives Considered
- **Standalone Vector Databases (Qdrant, Pinecone, Chroma):** Adds a separate database cluster and eventual consistency/synchronization challenges between document metadata and vector payloads. `pgvector` covers MVP-to-growth corpora inside PostgreSQL with full ACID transactions.
- **SQLite + sqlite-vec:** Helpful for single-node local files, but unsuitable for concurrent multi-container write locking and connection pooling.
- **Raw `asyncpg` Without SQLAlchemy/Alembic:** Avoids ORM dependencies but pushes manual schema migration tracking, connection pool management, and raw SQL maintenance onto application code.

## Conditions for Revisit
Revisit dedicated vector databases or partitioned read replicas when:
1. Total indexed chunks scale into the tens of millions where vertical PostgreSQL memory/HNSW index sizing becomes a bottleneck.
2. High-write dynamic multi-tenant ingestion workloads require dedicated shard/namespace isolation or decoupled background indexing queues.

