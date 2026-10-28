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
- **Synchronous Execution**: Currently, LangGraph and RAG workflows execute synchronously on thread workers. While FastAPI endpoints are `async`, internal operations call synchronous clients (`execute_with_retry`). Under high concurrency, worker threads will saturate without an async event loop migration (`ChatOpenAI.ainvoke`).

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

1. **Async Migration**: Replace synchronous `invoke()` calls with `ainvoke()` and `asyncio.gather()` to maximize FastAPI async event loop concurrency.
2. **Persistent Storage (pgvector)**: Migrate from in-memory index to PostgreSQL with `pgvector` for cross-process index persistence.
3. **Semantic Query Embedding Cache**: Store deterministic query vectors in Redis with a 24-hour TTL to reduce OpenAI embedding API spend.
4. **Hierarchical Document Ingestion**: Support PDF/HTML document parsers with token-aware chunk boundaries.
