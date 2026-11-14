# AlphaPilot AI — System Architecture & Design Specification

## 1. System Overview

**AlphaPilot AI** is an agentic portfolio intelligence and research platform designed to combine:
1. **Deterministic Financial Analysis**: Mathematical precision for portfolio risk assessments, concentration limits, and minimum cash buffer calculations.
2. **LLM-Based Research & Reasoning**: Semantic search and grounded question-answering over investment mandates, compliance guidelines, and policy research documents.

### Why Separate Deterministic Analysis from LLM Reasoning?
Large Language Models (LLMs) are probabilistic token-prediction engines. In financial domains, calculating weights, summing balances, or enforcing regulatory concentration limits with an LLM leads to non-deterministic rounding errors, calculation drift, or potential hallucinations. 

Conversely, rigid programmatic logic cannot answer qualitative policy questions such as *"Under what circumstances can a portfolio breach its cash buffer?"*

AlphaPilot AI addresses this by establishing a strict architectural boundary:
- **Math & Risk Rules**: Executed exclusively by deterministic Python functions validated via Pydantic (`app.services.risk_engine` and `app.tools.portfolio_tools`).
- **Research & Reasoning**: Handled by a controlled LangGraph agent and a grounded RAG pipeline (`app.services.rag_service`), where documents are treated strictly as **untrusted passive evidence** rather than executable instructions.

---

## 2. Component Architecture & System Flow

```mermaid
flowchart TD
    subgraph ClientBoundary["Client & External Network Boundary"]
        UserClient["API Client / Web UI"]
    end

    subgraph FastAPILayer["FastAPI Gateway & Input Validation Boundary"]
        FastAPIApp["FastAPI Application (app/main.py)"]
        ReqValidation["Pydantic Input Validation (extra='forbid', bounds, finite checks)"]
        FastAPIApp --> ReqValidation
    end

    subgraph AgentOrchestration["Agent Orchestration (LangGraph)"]
        PortfolioGraph["StateGraph (app/agents/portfolio_graph.py)"]
        CallModelNode["call_model Node"]
        ToolRouter{"Tool Requested?"}
        ExecToolNode["execute_tool Node"]
        FinalizeNode["finalize_text Node"]
    end

    subgraph ControlledTools["Controlled Tool Layer & Risk Engine"]
        PermGate["Permission Gate (portfolio:read)"]
        ToolSchema["Pydantic Tool Input Validation"]
        RiskEngine["Deterministic Risk Engine (app/services/risk_engine.py)"]
    end

    subgraph RAGPipeline["RAG Service Pipeline (app/services/rag_service.py)"]
        Embedder["Embedding Client (Protocol / OpenAI / Fake)"]
        VectorIdx["In-Memory Vector Index (Cosine Similarity)"]
        RelevanceFilter["Relevance Filter (score >= min_relevance_score)"]
        PromptBuilder["Prompt Builder (XML Delimited Untrusted Context)"]
    end

    subgraph ProviderBoundary["External LLM Provider Boundary (OpenAI API)"]
        ChatModel["ChatOpenAI (gpt-4o-mini)"]
        ResilienceLayer["Resilience Layer (Retry + Exponential Backoff + Deadline)"]
    end

    subgraph ObservabilityLayer["Observability & Telemetry (app/observability.py)"]
        TelemetryTracker["track_llm_operation Context Manager"]
        JsonLogger["JsonFormatter with Secret Redaction (sk-*, Bearer)"]
    end

    UserClient -->|"POST /portfolio/assess"| FastAPIApp
    UserClient -->|"POST /agent/run"| FastAPIApp
    UserClient -->|"POST /rag/ask"| FastAPIApp

    ReqValidation -->|"/agent/run"| PortfolioGraph
    ReqValidation -->|"/portfolio/assess"| RiskEngine
    ReqValidation -->|"/rag/ask"| RAGPipeline

    PortfolioGraph --> CallModelNode
    CallModelNode --> ResilienceLayer
    ResilienceLayer --> ChatModel
    ChatModel --> ToolRouter
    ToolRouter -->|"Yes"| ExecToolNode
    ToolRouter -->|"No"| FinalizeNode

    ExecToolNode --> PermGate
    PermGate --> ToolSchema
    ToolSchema --> RiskEngine
    RiskEngine -->|"Assessment Result"| ExecToolNode
    ExecToolNode -->|"Loop (Iteration Bounded)"| CallModelNode

    RAGPipeline --> Embedder
    Embedder --> VectorIdx
    VectorIdx --> RelevanceFilter
    RelevanceFilter --> PromptBuilder
    PromptBuilder --> ResilienceLayer

    ResilienceLayer -.-> TelemetryTracker
    TelemetryTracker -.-> JsonLogger
```

---

## 3. RAG Query Execution Sequence

The following sequence details how AlphaPilot answers policy research questions while enforcing untrusted data boundaries, source attribution, and timeouts.

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Client
    participant API as FastAPI (/rag/ask)
    participant Service as RagService
    participant Index as InMemoryVectorIndex
    participant Embed as EmbeddingClient
    participant LLM as LLM Provider (ChatOpenAI)
    participant Obs as Observability & Logger

    Client->>API: POST /rag/ask {question, top_k}
    API->>API: Validate RagAskRequest (length <= 2000, 1 <= top_k <= 10)
    alt Validation Failure
        API-->>Client: 422 Unprocessable Entity
    end

    API->>Service: answer_question(question, top_k)
    Service->>Service: Initialize monotonic WorkflowDeadline

    Service->>Index: Ensure index built (sample documents)
    Service->>Embed: embed_query(question)
    Embed-->>Service: query_vector [float * 64/1536]

    Service->>Index: search(query_vector, top_k)
    Index-->>Service: raw_results [(chunk, cosine_similarity)]

    Service->>Service: Filter results by min_relevance_score (>= 0.15)
    alt No chunks exceed threshold
        Service->>Obs: record_insufficient_evidence_without_llm()
        Service-->>API: RagAnswerResult(status="insufficient_evidence", sources=[])
        API-->>Client: 200 OK (Insufficient evidence standard refusal)
    end

    Service->>Service: Build isolated prompt (System prompt + <retrieved_evidence> + question)
    Service->>Obs: track_llm_operation(operation="rag_answer")
    Service->>LLM: invoke(messages) with execute_with_retry & deadline
    alt Provider Timeout or Deadline Exceeded
        LLM-->>Service: TimeoutError / DeadlineExceeded
        Service-->>API: RagAnswerResult(status="failed", error="Request to model provider timed out.")
        API-->>Client: 503 Service Unavailable
    else Success
        LLM-->>Service: AIMessage(answer_content)
        Service->>Obs: record_success(token_usage, cost)
        Service-->>API: RagAnswerResult(status="answered", answer, sources)
        API-->>Client: 200 OK (RagAskResponse with grounded sources)
    end
```

---

## 4. Agent Tool-Calling Sequence

The LLM agent does not directly execute Python code or access shell functions. It interacts exclusively through validated schema proposals mediated by LangGraph and application validation gates.

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Client
    participant API as FastAPI (/agent/run)
    participant Graph as LangGraph (LLMPortfolioState)
    participant LLM as ChatOpenAI (Model Node)
    participant Gate as Tool Permission Gate
    participant Schema as Pydantic Input Validator
    participant Risk as assess_portfolio_risk (Deterministic)

    Client->>API: POST /agent/run {query, portfolio}
    API->>API: Validate AgentRunRequest (extra='forbid', length bounds)
    API->>Graph: run_llm_portfolio_workflow(query)

    Graph->>Graph: Initialize state (iteration_count=0, max_iterations=2, deadline)
    Graph->>LLM: call_model(messages)
    LLM-->>Graph: AIMessage with tool_calls=[assess_portfolio_risk(args)]

    Graph->>Graph: route_after_model -> execute_tool_node
    Graph->>Gate: check_tool_permission("assess_portfolio_risk")
    alt Tool not in allowlist or permission missing
        Gate-->>Graph: PermissionError / ValueError
        Graph-->>API: state(status="failed", error="Controlled tool execution rejected...")
        API-->>Client: 200 OK {status: "failed", error: "..."}
    end

    Graph->>Schema: PortfolioAssessmentInput.model_validate(tool_args)
    alt Schema validation fails (e.g. NaN, negative cash, unexpected fields)
        Schema-->>Graph: ValidationError
        Graph-->>API: state(status="failed", error="Malformed tool arguments...")
        API-->>Client: 200 OK {status: "failed", error: "..."}
    end

    Graph->>Risk: assess_portfolio_risk(portfolio)
    Risk-->>Graph: dict(passed=True/False, violations, concentrations)
    Graph->>Graph: Append ToolMessage to state, iteration_count += 1

    Graph->>LLM: call_model([UserMessage, AIMessage, ToolMessage])
    LLM-->>Graph: AIMessage(final_response="Portfolio DEMO-001 passed risk assessment...")
    Graph->>Graph: route_after_model -> finalize_text_node
    Graph-->>API: state(status="completed", final_response, assessment_result)
    API-->>Client: 200 OK (AgentRunResponse)
```

---

## 5. Trust Boundaries & Security Enforcements

| Boundary | Data Flow | Trust Level | Enforced Controls |
| :--- | :--- | :--- | :--- |
| **Client to FastAPI** | HTTP Request Body (`/portfolio/assess`, `/agent/run`, `/rag/ask`) | **Untrusted** | Pydantic v2 schemas, `extra="forbid"`, `min_length=1`, `max_length=2000`, `allow_inf_nan=False`, finite float validation. |
| **FastAPI to Services** | Validated Pydantic Models & Dicts | **Validated / Internal** | Typed function signatures, explicit parameter passing, immutable state dictionaries. |
| **RAG Retrieval to LLM** | Ingested Document Excerpts | **Untrusted Passive Data** | Tagged as untrusted; delimited in prompt; system prompt explicitly forbids following instructions embedded in documents. |
| **LLM Output to System** | Tool call name & arguments | **Untrusted Model Generation** | Tool name allowlist, explicit permission check (`portfolio:read`), strict Pydantic model validation (`PortfolioAssessmentInput`). |
| **System to LLM Provider** | Prompts, Embeddings queries | **Sanitized Outgoing Payload** | API key stored in `SecretStr`, omitted from logs and client responses; timeout and deadline bounded. |
| **Internal Errors to Logs** | Exceptions, Telemetry events | **Diagnostic Internal** | `JsonFormatter` regex masking of `sk-*`, `Bearer *`, auth tokens, passwords, and secrets. |

---

## 6. Scalability and Performance Analysis

### 6.1 Asymptotic Complexity Analysis
- **In-Memory Vector Search**: Linear scan $\mathcal{O}(N \cdot D)$, where $N$ is total chunks indexed and $D$ is embedding dimension (e.g., $D = 64$ for tests, $D = 1536$ for OpenAI). For MVP ($N \approx 50$ chunks), retrieval takes $< 2\text{ ms}$. At $N = 100,000$ chunks, $\mathcal{O}(N \cdot D)$ linear scan would degrade request latency ($> 300\text{ ms}$) and require migrating to an approximate nearest neighbors (ANN) index (e.g., HNSW or IVF).
- **Ingestion & Chunking**: $\mathcal{O}(C)$ where $C$ is document character length. Overlapping chunk sliding window is single-pass $\mathcal{O}(C)$.
- **Risk Engine**: $\mathcal{O}(H)$ where $H$ is the number of holdings in a portfolio ($H \le 500$). The mathematical calculation takes $< 0.1\text{ ms}$.

### 6.2 Latency Characteristics
- **Network Latency to Model Provider**: Typically $500\text{ ms} - 2500\text{ ms}$ per call. Because agent workflows require up to 2 LLM invocations (tool selection + synthesis), end-to-end latency ranges from $1.5\text{ s} - 5.0\text{ s}$.
- **Async Event-Loop Execution (Day 14)**: FastAPI routes (`/agent/run`, `/rag/ask`) now invoke native async provider workflows (`ChatOpenAI.ainvoke`, `OpenAIEmbeddings.aembed_query`, `OpenAIEmbeddings.aembed_documents`, LangGraph `ainvoke`) with non-blocking `execute_with_retry_async` (`asyncio.sleep`) and `LLMConcurrencyLimiter` (`asyncio.Semaphore`). Sync-only clients/mocks safely fall back to `asyncio.to_thread`. Synchronous entrypoints (`RagService.answer_question`, `run_llm_portfolio_workflow`) remain intact for the deterministic offline evaluation runner and unit tests.

### 6.3 Shared Mutable State & Multi-Instance Constraints
- `InMemoryVectorIndex` stores chunk vectors inside Python heap memory on a single process.
- **Limitation**: Running multiple uvicorn worker processes (`--workers 4`) or multiple container replicas results in duplicate index memory allocations and independent state. Updating sample documents in one worker does not reflect in others.

---

## 7. Storage and Caching Evolution

```mermaid
graph LR
    subgraph Phase1["Phase 1: Current In-Memory MVP"]
        M1[InMemoryVectorIndex]
        M2[Singleton Process Memory]
    end

    subgraph Phase2["Phase 2: Hybrid Persistent Index"]
        P1[PostgreSQL with pgvector]
        P2[Shared Multi-Instance State]
    end

    subgraph Phase3["Phase 3: Production Enterprise Scale"]
        E1[Managed Vector DB / Pinecone / Qdrant]
        E2[Redis Distributed Embedding Cache]
        E3[S3 / Object Store for Raw PDFs]
    end

    Phase1 -->|"Trigger: Dataset > 50,000 chunks or multi-process replicas"| Phase2
    Phase2 -->|"Trigger: Multi-tenant tenant isolation, > 1M documents"| Phase3
```

### Caching Considerations
1. **Query Embedding Cache**: A hash map keying `SHA256(query_text) -> vector` avoids paying provider API costs for repeated user questions.
2. **Answer Cache**: Caching synthesized answers must be avoided or keyed with strict document revision IDs. Otherwise, if compliance policies change, stale answers could be served.

---

## 8. Prioritized Future Architectural Improvements

1. **Persistent Storage (pgvector)**: Migrate from process-local `InMemoryVectorIndex` to PostgreSQL with `pgvector` for cross-worker and cross-container index persistence.
2. **Semantic Query Embedding Cache**: Store deterministic query vectors in Redis with a 24-hour TTL to reduce OpenAI embedding API spend across workers.
3. **Background Document Ingestion Queue**: Offload large multi-document ingestion jobs (e.g., 500 PDFs) to an external queue and worker pool with job-status tracking.
4. **Hierarchical Document Ingestion**: Support PDF/HTML document parsers with token-aware chunk boundaries.

---

## Distributed Computing and Concurrency

> **Updated Scope Note (Day 15)**: Persistent vector storage using **PostgreSQL + `pgvector`** (`RAG_STORAGE_BACKEND=postgres`) is now implemented via SQLAlchemy asyncpg connection pooling and Alembic migrations alongside the default offline **`in_memory`** backend (`InMemoryVectorIndex`). Shared Redis caching, centralized OpenTelemetry export, and external background job queues documented in Sections 8–9 remain proposed future extensions and are not deployed in this repository.

### 1. Current Execution Model by Service

| Component / Endpoint | Execution Model | Rationale |
| :--- | :--- | :--- |
| `GET /health` | Non-blocking `async def` | Returns lightweight process health metadata in microseconds. |
| `POST /portfolio/assess` | `async def` calling inline deterministic Python (`assess_portfolio_risk`) | Portfolio math over $\le 500$ holdings takes $< 0.1\text{ ms}$; multiprocessing IPC would add far more latency than the calculation itself. |
| `POST /agent/run` | `async def` calling `await run_llm_portfolio_workflow_async` | LangGraph `ainvoke()` executes `call_model` asynchronously via `ChatOpenAI.ainvoke` (or `asyncio.to_thread` fallback) while keeping deterministic `execute_tool_node` inline. |
| `POST /rag/ask` | `async def` calling `await rag_service.answer_question_async` | Asynchronous index warm-up (`aembed_documents`), asynchronous query embedding (`aembed_query`), fast in-memory snapshot search, and `ChatOpenAI.ainvoke`. |
| `app.evaluation.runner` | Synchronous (`RagService.answer_question`) | Deterministic offline evaluation runs sequentially without event-loop or external provider overhead. |

### 2. FastAPI Request Concurrency Audit (Before vs. After Day 14)

| Operation | Pre-Day 14 Execution | Potential Issue | Implemented Day 14 Solution |
| :--- | :--- | :--- | :--- |
| `/rag/ask` LLM `invoke()` | Synchronous inside `async def` route | Blocked the Uvicorn event loop thread for $500\text{--}2500\text{ ms}$, stalling `/health` and `/portfolio/assess` | Async `answer_question_async` calling `ChatOpenAI.ainvoke` (with `asyncio.to_thread` fallback for sync mocks) |
| `/rag/ask` `embed_query()` | Synchronous inside `async def` route | Blocked event loop during embedding network round-trip | Async `RetrievalService.aretrieve` calling `OpenAIEmbeddings.aembed_query` |
| Cold-start `_ensure_initialized` | Unguarded sync check-then-act | Concurrent burst requests could trigger duplicate chunking, duplicate `embed_documents` billing, and duplicate index entries | Double-checked locking with `threading.Lock` (sync) and loop-scoped `asyncio.Lock` (async), plus atomic `InMemoryVectorIndex` snapshot reads |
| Retry backoff `time.sleep()` | Synchronous `time.sleep` inside `execute_with_retry` | Froze entire event loop for $0.5\text{--}2.0\text{ s}$ per retry | `execute_with_retry_async` using `await asyncio.sleep(delay)` integrated with `WorkflowDeadline` |
| `/agent/run` graph execution | Synchronous `graph.invoke()` inside `async def` route | Blocked event loop across up to 2 LLM turns and retries | `run_llm_portfolio_workflow_async` using LangGraph `ainvoke` and async `call_model_node_async` |
| Unbounded concurrent LLM calls | No in-flight cap | Burst traffic could exhaust sockets/memory or amplify 429 retry storms | `LLMConcurrencyLimiter` (`asyncio.Semaphore`) with bounded `llm_queue_timeout_seconds` capped by `WorkflowDeadline` |

### 3. Why Event-Loop Blocking Matters (Timeline Example)

Suppose **User A** sends a slow RAG question ($1.5\text{ s}$ LLM latency), **User B** sends `/portfolio/assess` ($< 0.1\text{ ms}$ CPU), and **User C** sends another RAG question ($1.5\text{ s}$):
- **With blocking sync calls inside `async def`:** `0.0s -> 1.5s` User A holds the single event loop thread; User B and User C wait behind User A until `1.5s`. Total time $\approx 3.0\text{ s}$, and User B's sub-millisecond portfolio check suffers $1.5\text{ s}$ tail latency.
- **With `await` async I/O:** At `0.0s`, User A suspends on `await client.ainvoke()`, immediately yielding the event loop. User B completes `/portfolio/assess` at `0.001s`. User C starts its `await client.ainvoke()` at `0.002s`. Both A and C overlap waiting on network I/O and finish around `1.5s`.

### 4. Thread, Process, and Container Boundaries

- **Single process:** One Python interpreter owns one heap (`InMemoryVectorIndex`, `_DEFAULT_RAG_SERVICE`, `_DEFAULT_LLM_LIMITER`, `_RECORDED_EVENTS`).
- **Multiple threads (`asyncio.to_thread` / `ThreadPoolExecutor`):** Threads share the same process heap. In CPython, the Global Interpreter Lock (GIL) serializes Python bytecode execution across threads for CPU-bound work, while I/O-bound syscalls release the GIL. Shared structures (`InMemoryVectorIndex`, `_RECORDED_EVENTS`, `PricingRegistry`, `_ensure_initialized`) therefore use explicit locks (`threading.Lock` / `threading.RLock`).
- **Multiple Uvicorn worker processes (`--workers N`):** Each worker is a separate OS process with **independent heap memory** and its own `asyncio` event loop. Indexing a document or incrementing an in-memory counter in Worker 1 is invisible to Worker 2.
- **Multiple containers / pods:** Each container runs its own worker processes; shared state across containers requires external services (PostgreSQL, Redis, object storage).

### 5. Concurrency Safety & State Isolation

- **Per-request state:** `LLMPortfolioState`, `WorkflowDeadline`, `RagAnswerResult`, and `TelemetryTracker` are instantiated fresh per request/operation so User A's portfolio holdings or query text never leak into User B's state or telemetry measurements.
- **Read-mostly vector index:** `InMemoryVectorIndex` protects writes (`add`, `add_batch`, `clear`) with an `RLock`, stores vectors as immutable tuples, and takes a brief lock only to copy a tuple snapshot during `search()`, allowing concurrent searches to score cosine similarity without blocking each other.
- **Shared telemetry sink:** `_RECORDED_EVENTS` and `PricingRegistry` guard mutations and copies with `threading.Lock` / `threading.RLock`.

### 6. Rate Limiting, Backpressure & Bounded Concurrency

- **Application-level concurrency limit (`LLMConcurrencyLimiter`) vs. Provider account limits:**
  - `LLMConcurrencyLimiter` (`asyncio.Semaphore` per event loop, configured via `MAX_CONCURRENT_LLM_REQUESTS`, default `10`) bounds how many simultaneous in-flight LLM calls *this worker process* allows, protecting local memory, connection pools, and downstream stability.
  - This is **separate from** OpenAI's account-level RPM/TPM (Requests/Tokens Per Minute) quotas, which apply globally across all workers, apps, and users sharing the API key.
- **Bounded waiting & deadline integration:** A request waits at most `min(llm_queue_timeout_seconds, deadline.remaining_seconds())` for a slot. If no slot opens in time, it fails fast with `ConcurrencyLimitExceededError` (HTTP `503` on `/rag/ask`, `status="failed"` on `/agent/run`, telemetry `error_category="backpressure"`).
- **No retry amplification:** `is_retryable_error(ConcurrencyLimitExceededError)` returns `False`, and slots are acquired *per individual attempt* (never held during exponential backoff `asyncio.sleep`) so backing-off requests cannot starve active requests or amplify a thundering herd.

### 7. Multi-Worker Uvicorn Operation

- **Development (Windows-friendly, single worker with optional reload):**
  ```powershell
  .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
- **Multi-worker mode (reload MUST be disabled):**
  ```powershell
  .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
  ```
  - **Why `--reload` and `--workers` are not interchangeable:** `--reload` spawns a file-watching supervisor process designed for local code editing; combining `--reload` with `--workers > 1` is unsupported/rejected by Uvicorn and obscures process boundaries.
  - **Worker sizing considerations:** Choose worker count based on CPU cores, per-worker memory footprint (each worker holds its own Python interpreter + `InMemoryVectorIndex`), OpenAI account RPM/TPM limits (`workers * max_concurrent_llm_requests` sets peak concurrent LLM calls per node), and workload mix (I/O-heavy LLM routes benefit from async concurrency within each worker rather than excessive processes).

### 8. Background Processing Architecture (Proposed Future Design)

When a user uploads **500 research documents** (thousands of chunks and embedding API calls taking tens of seconds or minutes), synchronous HTTP request-response handling risks client/proxy timeouts, worker starvation, and lost progress if a connection drops. For large ingestion workloads, a proposed background job pipeline decouples submission from processing:

1. FastAPI receives `POST /documents/ingest` with batch payload or file references.
2. API validates metadata/payload limits and rejects malformed input immediately (`422`).
3. API creates a persistent job record (`status="queued"`, `job_id`) in a shared database (proposed: PostgreSQL).
4. API enqueues the `job_id` onto a durable broker/queue (proposed: Redis / RabbitMQ / SQS) and immediately returns `202 Accepted` with `job_id`.
5. An independent background worker process pulls the job from the queue.
6. The worker chunks documents, generates embeddings (respecting provider rate limits), and persists documents + vectors into shared storage (proposed: PostgreSQL + `pgvector` or managed vector DB).
7. The worker updates job progress and final status (`completed` / `failed`) in the job store.
8. The client polls `GET /jobs/{job_id}` (or receives a webhook/SSE event) to inspect completion status.

```mermaid
flowchart LR
    Client["Client"] -->|"1. POST /documents/ingest"| API["FastAPI API (Proposed Endpoint)"]
    API -->|"2. Create Job (queued)"| JobDB[("Proposed: PostgreSQL Job Store")]
    API -->|"3. Enqueue job_id"| Queue["Proposed: Job Queue (Redis / Broker)"]
    API -->|"4. 202 Accepted (job_id)"| Client
    Queue -->|"5. Consume Job"| Worker["Proposed: Background Ingestion Worker"]
    Worker -->|"6. Embed Chunks"| OpenAI["External Embedding Provider"]
    Worker -->|"7. Upsert Vectors"| VectorDB[("Proposed: pgvector / Vector DB")]
    Worker -->|"8. Update Status"| JobDB
```

### 9. Scalability Roadmap & Capacity Planning (Proposed Multi-Instance Architecture)

```mermaid
flowchart TD
    Client["API Clients"] --> LB["Proposed: Load Balancer / Reverse Proxy"]
    LB --> API1["FastAPI Instance 1 (Stateless Workers)"]
    LB --> API2["FastAPI Instance 2 (Stateless Workers)"]

    subgraph Services["Application Layer (Per Worker)"]
        RAG["RAG & Agent Async Orchestration + Semaphore"]
    end

    API1 --> RAG
    API2 --> RAG

    RAG -->|"Proposed: Shared Embedding Cache"| Cache[("Proposed: Redis Cache")]
    RAG -->|"Proposed: Shared Vector Search"| VectorStore[("Proposed: PostgreSQL + pgvector / Vector DB")]
    RAG -->|"Proposed: Async Job Enqueue"| JobQueue["Proposed: Background Job Queue"]
    JobQueue --> BgWorkers["Proposed: Ingestion Workers"]
    BgWorkers --> VectorStore
    RAG -->|"HTTPS + Bounded Concurrency + Retries"| LLM["External LLM Provider (OpenAI API)"]
    BgWorkers -->|"Embeddings API"| LLM
```

#### Migration Triggers

| Current limitation | Future solution | When to introduce |
| :--- | :--- | :--- |
| Process-local `InMemoryVectorIndex` (default offline mode) | **Implemented on Day 15**: Switch `RAG_STORAGE_BACKEND=postgres` (`PostgresVectorRepository`) | Already available when persistence across restarts or multiple workers is required; managed vector DB if corpus reaches millions of vectors with heavy metadata filtering |
| Repeated identical query embeddings hit OpenAI every request | Shared embedding cache (`Redis` keyed by normalized query hash + model ID + TTL) | Repeated FAQ/policy queries account for measurable embedding cost or latency |
| In-memory `_RECORDED_EVENTS` telemetry sink per process | Centralized OpenTelemetry / structured log sink + metrics collector | Running $> 1$ worker or container where aggregate latency/cost dashboards and alerting are required |
| Process-local `LLMConcurrencyLimiter` cannot coordinate global provider quotas across replicas | Distributed token-bucket rate limiter (`Redis`) or API gateway quota controller | Horizontal scaling across multiple instances risks exceeding shared OpenAI organization RPM/TPM limits |
| Synchronous startup/request ingestion only over static sample docs | Asynchronous job queue + dedicated ingestion worker pool | Users upload dynamic document batches (e.g., 50–500+ PDFs) where processing exceeds HTTP timeout budgets |

### 10. Key Failure Scenarios & Mitigations

- **Event loop starvation:** Avoided on `/rag/ask` and `/agent/run` via native async calls (`ainvoke`, `aembed_query`) and `asyncio.to_thread` fallback; remaining risk if someone introduces blocking I/O inside an `async def` route.
- **Cold-start thundering herd on vector index:** Prevented per worker via double-checked `asyncio.Lock` / `threading.Lock` inside `_ensure_initialized_async()` / `_ensure_initialized()`; optional `RAG_PREWARM_ON_STARTUP=true` warms index before serving traffic.
- **Queue explosion & retry amplification under slow OpenAI responses:** Prevented by `LLMConcurrencyLimiter` bounded `llm_queue_timeout_seconds`, `WorkflowDeadline`, non-retryable `ConcurrencyLimitExceededError`, and `max_retries=0` on `ChatOpenAI`.
- **Multi-worker memory exhaustion:** When running `in_memory`, each Uvicorn worker duplicates `InMemoryVectorIndex`. Switching to `RAG_STORAGE_BACKEND=postgres` externalizes chunk vectors to shared PostgreSQL storage.

---

## Persistent Vector Storage with PostgreSQL + pgvector (Day 15)

### 1. Concepts: PostgreSQL vs. `pgvector` vs. Relational Storage vs. Vector Similarity Search

- **PostgreSQL (Relational Database):** A general-purpose transactional SQL database storing structured rows, primary/foreign keys, JSONB metadata, timestamps, and ACID integrity guarantees.
- **`pgvector` (PostgreSQL Extension):** Adds a native `VECTOR(n)` data type, distance operators (`<=>` cosine distance, `<->` L2 distance, `<#>` inner product), and approximate nearest-neighbor vector indexes (such as HNSW and IVFFlat) directly inside PostgreSQL.
- **Relational Storage vs. Vector Similarity Search:**
  - *Relational queries* answer exact structured questions (`WHERE document_id = 'SYNTH-DOC-001'`, `ON DELETE CASCADE`, `UNIQUE (document_id, chunk_index)`) using B-tree indexes.
  - *Vector similarity search* answers semantic proximity questions (`ORDER BY embedding <=> $1 LIMIT k`) by ranking stored embeddings by geometric closeness in $n$-dimensional space, enabling retrieval even when exact keywords differ.
  - Storing both side-by-side in PostgreSQL lets AlphaPilot delete or update documents transactionally, preserve source metadata atomically with chunk vectors, and avoid operating two separate databases for MVP/scale-up deployments.

### 2. Storage Architecture & Backend Selection

```mermaid
flowchart TD
    RagSvc["RagService (app/services/rag_service.py)"] -->|"backend == in_memory"| InMemRet["RetrievalService + InMemoryVectorIndex"]
    RagSvc -->|"backend == postgres"| PgRet["RetrievalService + AsyncVectorStore"]
    PgRet --> Repo["PostgresVectorRepository (app/db/repository.py)"]
    Repo --> DbMgr["DatabaseManager asyncpg Pool (app/db/session.py)"]
    DbMgr -->|"SQL + <=> HNSW Query"| PG[("PostgreSQL 16 + pgvector")]
```

- **Dependency-Injected Storage Abstraction:** `RetrievalService` accepts either `InMemoryVectorIndex` (default for deterministic unit tests and offline evaluation) or an `AsyncVectorStore` implementation (`PostgresVectorRepository`).
- **No Silent Fallback Policy:** When `RAG_STORAGE_BACKEND=postgres`, missing `DATABASE_URL`, missing `pgvector` extension, dimension/model mismatch, or PostgreSQL connection failures raise explicit `VectorStoreError` / `VectorStoreConnectionError` / `EmbeddingDimensionMismatchError` errors (surfaced on `/rag/ask` as sanitized HTTP `503` without exposing credentials), rather than silently dropping to an empty `InMemoryVectorIndex`.
- **Restart Persistence Reuse:** On startup or first request, `RagService._ensure_initialized_async()` verifies connectivity and checks `acount()`. If chunks already exist in PostgreSQL (`count > 0`), AlphaPilot reuses the persisted vectors directly without calling OpenAI embeddings again.

### 3. Database Schema & HNSW Indexing

Defined in [`app/db/models.py`](../app/db/models.py) and migrated via [`alembic/versions/0001_create_pgvector_rag_tables.py`](../alembic/versions/0001_create_pgvector_rag_tables.py):

1. **`research_documents`**
   - `document_id` (`VARCHAR(128)`, PK)
   - `title` (`TEXT`, NOT NULL)
   - `content` (`TEXT`, NOT NULL)
   - `metadata` (`JSONB`, NOT NULL)
   - `created_at`, `updated_at` (`TIMESTAMPTZ`, server-managed UTC timestamps)
2. **`document_chunks`**
   - `chunk_id` (`VARCHAR(192)`, PK)
   - `document_id` (`VARCHAR(128)`, FK -> `research_documents.document_id` `ON DELETE CASCADE`, B-tree indexed)
   - `document_title` (`TEXT`, NOT NULL)
   - `chunk_index` (`INTEGER`, NOT NULL, with `UNIQUE (document_id, chunk_index)`)
   - `chunk_text` (`TEXT`, NOT NULL)
   - `metadata` (`JSONB`, NOT NULL)
   - `embedding_model` (`VARCHAR(128)`, NOT NULL, B-tree indexed)
   - `embedding_dimension` (`INTEGER`, NOT NULL)
   - `embedding` (`VECTOR(1536)`, NOT NULL — aligned with `EMBEDDING_DIMENSION=1536` for `text-embedding-3-small`)
   - `created_at`, `updated_at` (`TIMESTAMPTZ`)
   - **HNSW Vector Index (`ix_document_chunks_embedding_hnsw`):** Built with `USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`, queried cleanly via `ORDER BY embedding <=> :query ASC LIMIT :top_k`.

- **Dimension & Model Guardrails:** Both `Settings` and `PostgresVectorRepository` validate that stored and queried vectors match the configured model and dimension, preventing silent mixing of incompatible embedding spaces.
- **Credential Hygiene:** `DATABASE_URL` uses `SecretStr`, `alembic.ini` stores no credentials, and [`app/logging_config.py`](../app/logging_config.py) masks passwords embedded in `postgres://` / `postgresql+asyncpg://` URLs and DSN strings.

### 4. Configuration Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RAG_STORAGE_BACKEND` | `in_memory` | Set to `postgres` to persist RAG chunks and vectors in PostgreSQL + `pgvector`. |
| `DATABASE_URL` | `None` (required when `postgres`) | Async connection URL, e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/alphapilot`. |
| `EMBEDDING_DIMENSION` | `1536` | Must match `OPENAI_EMBEDDING_MODEL` (`1536` for `text-embedding-3-small`, `3072` for `text-embedding-3-large`). |
| `DB_POOL_SIZE` | `5` | SQLAlchemy asyncpg base pool connections per worker process. |
| `DB_MAX_OVERFLOW` | `10` | Maximum overflow connections above `DB_POOL_SIZE`. |
| `DB_POOL_TIMEOUT_SECONDS` | `10.0` | Maximum seconds to wait for a pooled DB connection before raising timeout error. |
| `DB_ECHO` | `false` | Enables verbose SQLAlchemy query logging (keep `false` in production). |

### 5. Setup, Migration & Test Commands (Windows PowerShell)

```powershell
# 1. Install dependencies (including sqlalchemy[asyncio], asyncpg, pgvector, alembic)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Start local PostgreSQL 16 + pgvector via Docker Compose
docker compose up -d

# 3. Configure environment for PostgreSQL backend
$env:RAG_STORAGE_BACKEND = "postgres"
$env:DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/alphapilot"

# 4. Run Alembic database migrations (creates pgvector extension, tables, constraints, and HNSW index)
.venv\Scripts\alembic.exe upgrade head

# 5. Run unit & mocked async tests
.venv\Scripts\python.exe -m pytest tests\test_persistence_sync_unit.py tests\test_pgvector_mocked_async.py

# 6. Run live PostgreSQL + pgvector integration tests
$env:ALPHAPILOT_RUN_PGVECTOR_TESTS = "1"
.venv\Scripts\python.exe -m pytest tests\test_pgvector_integration.py
```

---

## Broker Integration, Credential Security & Immutable Snapshots (Days 16–21)

AlphaPilot AI connects to brokerage accounts through a read-only broker abstraction ([`app/integrations/brokers/base.py`](../app/integrations/brokers/base.py)), initially implementing **Zerodha Kite Connect v3** ([`app/integrations/brokers/zerodha.py`](../app/integrations/brokers/zerodha.py)), while separating broker protocols from canonical normalization ([`app/integrations/normalizers/portfolio_normalizer.py`](../app/integrations/normalizers/portfolio_normalizer.py)), authenticated AES-256-GCM encryption ([`app/integrations/security/encryption.py`](../app/integrations/security/encryption.py)), multi-user ownership ([`app/auth/`](../app/auth/)), immutable PostgreSQL snapshot history ([`alembic/versions/0002_add_broker_and_portfolio_snapshots.py`](../alembic/versions/0002_add_broker_and_portfolio_snapshots.py)), and deterministic risk + LangGraph workflows ([`app/agents/portfolio_graph.py`](../app/agents/portfolio_graph.py)).

```mermaid
flowchart LR
    User["Authenticated User (Bearer HS256 JWT)"] -->|"POST /brokers/zerodha/connect"| BrokerSvc["BrokerConnectionService"]
    BrokerSvc -->|"Signed CSRF State + redirect_params"| KiteLogin["Zerodha Kite Login v3"]
    User -->|"GET /brokers/zerodha/callback"| BrokerSvc
    BrokerSvc -->|"SHA-256 checksum exchange"| KiteAPI["Kite Connect REST API"]
    BrokerSvc -->|"AES-256-GCM (AAD: user_id + broker)"| PgDB[("PostgreSQL Shared DatabaseManager")]
    User -->|"POST /portfolios/{id}/refresh (Manual Only)"| IngestionSvc["PortfolioIngestionService"]
    IngestionSvc -->|"Holdings + Positions + Equity Margins"| KiteAPI
    IngestionSvc --> Normalizer["ZerodhaPortfolioNormalizer"]
    Normalizer -->|"Atomic TX: snapshot + holdings (Numeric)"| PgDB
    User -->|"GET /risk & POST /analyze"| Intelligence["Risk Engine + LangGraph (Sanitized Provenance Only)"]
    PgDB -->|"Latest Completed Snapshot"| Intelligence
```

### Key Architectural Guarantees
1. **Strictly Read-Only Broker Scope:** Zero trading, order placement, or transfer methods exist anywhere in `BrokerAdapter` or `ZerodhaBrokerAdapter`.
2. **Manual-Only Refresh:** Snapshots are created exclusively upon explicit `POST /portfolios/{portfolio_id}/refresh` calls; there are no polling workers, cron tasks, or external queues.
3. **Single Database Manager & Connection Pool:** Both RAG `pgvector` tables and broker/portfolio snapshot tables reuse [`DatabaseManager`](../app/db/session.py) over the same PostgreSQL connection pool.
4. **Secret Isolation:** Access tokens are encrypted with AES-256-GCM + AAD ownership binding and retired-key rotation support; zero plaintext tokens, ciphertexts, checksums, or `request_token` parameters ever enter logs, HTTP responses, or `LLMPortfolioState`.
5. **Financial Integrity:** `average_purchase_price` is never substituted as current market `last_price`, partial/malformed broker payloads are never marked `"completed"`, and missing market valuations or sector metadata are explicitly reported as limitations.

For complete setup steps, API schemas, environment variables, security controls, and test commands, see **[Broker Integration Guide](broker-integration.md)** and **[ADR 009](adr/009-broker-integration-and-snapshot-architecture.md)**.


