# ADR 007: Asynchronous Provider Execution, Process-Local Index Lifecycle, and Bounded Concurrency

## Status
Accepted

## Context
Prior to Day 14, AlphaPilot AI exposed `async def` FastAPI routes (`/rag/ask`, `/agent/run`) that called synchronous LangGraph/OpenAI operations (`client.invoke`, `embed_query`, and `time.sleep` inside `execute_with_retry`). Under concurrent requests:
1. Synchronous network I/O and retry sleeps blocked the Uvicorn `asyncio` event loop thread, stalling fast endpoints like `/health` and `/portfolio/assess`.
2. Lazy in-memory vector index initialization (`RagService._ensure_initialized`) lacked synchronization across concurrent requests, risking duplicate embedding API calls and duplicate chunk insertions.
3. Unbounded concurrent LLM invocations risked memory/socket exhaustion and retry amplification during provider slowdowns.
4. Meanwhile, introducing Redis, PostgreSQL/pgvector, or Celery prematurely would violate MVP simplicity and add operational overhead without current dataset scale justifying it.

## Decision
We adopted a focused, single-node multi-worker-ready concurrency architecture:
1. **Native Async Execution with Sync Compatibility**:
   - Added async service workflows (`RetrievalService.aindex_chunks` / `aretrieve`, `RagService.answer_question_async`, `run_llm_portfolio_workflow_async`, and `execute_with_retry_async` using `asyncio.sleep`).
   - Prefer native `ChatOpenAI.ainvoke` and `OpenAIEmbeddings.aembed_query` / `aembed_documents` where available, falling back to `asyncio.to_thread` only for synchronous-only clients/mocks (explicitly guarding against `MagicMock` attribute traps via `inspect.iscoroutinefunction`).
   - Preserved synchronous `RagService.answer_question` and `run_llm_portfolio_workflow` so deterministic unit tests and the offline evaluation runner require zero event-loop changes.
2. **Concurrency-Safe Process-Local Index Lifecycle**:
   - Guarded `RagService` initialization with `threading.Lock` (sync) and per-event-loop `asyncio.Lock` (async), and protected `InMemoryVectorIndex` writes and brief snapshot reads with `threading.RLock`.
   - Registered process-local singletons cleanly in FastAPI `lifespan`, with opt-in `RAG_PREWARM_ON_STARTUP` (default `false` so offline tests and `/health` remain dependency-free).
3. **Application-Level Bounded Concurrency (`LLMConcurrencyLimiter`)**:
   - Implemented `LLMConcurrencyLimiter` using per-loop `asyncio.Semaphore` (`MAX_CONCURRENT_LLM_REQUESTS`, default `10`) and bounded wait (`LLM_QUEUE_TIMEOUT_SECONDS`, default `5.0`) capped by `WorkflowDeadline.remaining_seconds()`.
   - Acquired slots strictly around individual provider invocations (never during backoff sleeps, validation, or deterministic tool execution) and classified `ConcurrencyLimitExceededError` as non-retryable (`error_category="backpressure"`) to prevent retry amplification.

## Benefits
- **Non-Blocking Concurrency**: Concurrent I/O-bound LLM/embedding requests overlap on the event loop without starving `/health` or `/portfolio/assess`.
- **Single Cold-Start Build**: Concurrent requests on a cold worker await a single initialization rather than duplicating embedding spend.
- **Controlled Backpressure**: Excess load fails fast with sanitized HTTP `503` / `failed` responses rather than building unbounded queues.
- **Zero External Infrastructure Overhead**: Preserves instant local setup and deterministic offline tests.

## Trade-offs & Limitations
- **Process-Local Memory**: Each Uvicorn worker (`--workers N`) maintains an independent `InMemoryVectorIndex`, `LLMConcurrencyLimiter`, and `_RECORDED_EVENTS` buffer. Peak per-node concurrency is `workers * max_concurrent_llm_requests`, and dynamic index updates are not shared across workers.
- **Local vs. Provider Quotas**: `LLMConcurrencyLimiter` caps concurrent in-flight requests per worker, not OpenAI account-wide RPM/TPM quotas.

## Alternatives Considered
- *Wrapping all sync methods blindly in `asyncio.to_thread`*: Wastes thread-pool workers on network sockets when `ChatOpenAI` and `OpenAIEmbeddings` already support native `asyncio` I/O (`ainvoke`, `aembed_query`), and does not eliminate `time.sleep` thread pinning.
- *Adding Redis + PostgreSQL/pgvector + Celery now*: Premature for a static 3-document MVP corpus; documented with concrete migration triggers in `docs/architecture.md`.

## Conditions for Revisit
Revisit external shared storage and distributed rate limiting when:
1. Research documents require runtime user uploads shared across multiple workers/containers.
2. Corpus size exceeds ~10,000–50,000 chunks or startup indexing exceeds 5 seconds.
3. Multi-instance horizontal scaling requires coordinated organization-wide OpenAI RPM/TPM throttling or centralized telemetry aggregation.

