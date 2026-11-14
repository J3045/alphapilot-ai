"""Synchronous, thread-safety, and multi-process isolation tests for Day 14.

These tests verify configuration bounds, error classification, thread-safe index and
RAG initialization, independent portfolio state under parallel thread execution,
concurrency-safe telemetry collection, and OS process memory isolation without
requiring an asyncio OS-socket event loop.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
import pytest
from pydantic import ValidationError

from app.agents.portfolio_graph import run_llm_portfolio_workflow
from app.config import Settings
from app.models.document import ResearchDocument, TextChunk
from app.models.portfolio import Holding, Portfolio
from app.observability import (
    categorize_error,
    clear_recorded_events,
    get_recorded_events,
    track_llm_operation,
)
from app.services.rag_service import RagService
from app.services.resilience import (
    ConcurrencyLimitExceededError,
    LLMConcurrencyLimiter,
    is_retryable_error,
    sanitize_provider_error,
    supports_async_method,
)
from app.services.retrieval_service import FakeEmbeddingClient, InMemoryVectorIndex
from app.services.risk_engine import assess_portfolio_risk


@pytest.fixture(autouse=True)
def reset_telemetry():
    clear_recorded_events()
    yield
    clear_recorded_events()


# ---------------------------------------------------------------------------
# 1. Configuration & Primitive Validation
# ---------------------------------------------------------------------------

def test_concurrency_settings_validation_and_defaults():
    cfg = Settings()
    assert cfg.max_concurrent_llm_requests == 10
    assert cfg.llm_queue_timeout_seconds == 5.0
    assert cfg.rag_prewarm_on_startup is False

    custom = Settings(
        max_concurrent_llm_requests=25,
        llm_queue_timeout_seconds=2.5,
        rag_prewarm_on_startup=True,
    )
    assert custom.max_concurrent_llm_requests == 25
    assert custom.llm_queue_timeout_seconds == 2.5
    assert custom.rag_prewarm_on_startup is True

    with pytest.raises(ValidationError):
        Settings(max_concurrent_llm_requests=0)

    with pytest.raises(ValidationError):
        Settings(llm_queue_timeout_seconds=0.0)


def test_limiter_constructor_validation():
    limiter = LLMConcurrencyLimiter(max_concurrent=4, queue_timeout_seconds=1.5)
    assert limiter.max_concurrent == 4
    assert limiter.queue_timeout_seconds == 1.5

    with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
        LLMConcurrencyLimiter(max_concurrent=0)

    with pytest.raises(ValueError, match="queue_timeout_seconds must be > 0"):
        LLMConcurrencyLimiter(queue_timeout_seconds=0.0)


def test_backpressure_error_classification_and_sanitization():
    exc = ConcurrencyLimitExceededError("Model execution capacity saturated (2 concurrent requests).")
    assert is_retryable_error(exc) is False
    assert categorize_error(exc) == "backpressure"
    assert sanitize_provider_error(exc) == (
        "Model execution capacity temporarily saturated. Please try again shortly."
    )


def test_supports_async_method_guards_against_magic_mock():
    sync_mock = MagicMock()
    assert supports_async_method(sync_mock, "ainvoke") is False

    class SyncOnlyClient:
        def invoke(self, messages):
            return messages

    class AsyncClient:
        async def ainvoke(self, messages):
            return messages

    assert supports_async_method(SyncOnlyClient(), "ainvoke") is False
    assert supports_async_method(AsyncClient(), "ainvoke") is True

    async_mock_holder = MagicMock()
    async_mock_holder.ainvoke = AsyncMock(return_value="ok")
    assert supports_async_method(async_mock_holder, "ainvoke") is True


# ---------------------------------------------------------------------------
# 2. Thread-Safe Index Snapshot & Concurrent Initialization
# ---------------------------------------------------------------------------

def test_in_memory_vector_index_concurrent_reads_and_immutability():
    index = InMemoryVectorIndex()
    vec = [1.0, 0.0, 0.0]
    chunk = TextChunk(
        chunk_id="DOC-1_chunk_0",
        document_id="DOC-1",
        document_title="Test Policy",
        chunk_index=0,
        chunk_text="Minimum cash reserve floor is 5.0%.",
    )
    index.add(chunk, vec)

    # Mutating caller's original list must not mutate the index's stored vector
    vec[0] = 0.0
    vec[1] = 1.0

    def run_search(idx: int) -> list[tuple[TextChunk, float]]:
        return index.search([1.0, 0.0, 0.0], top_k=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_search, range(24)))

    assert len(results) == 24
    for res in results:
        assert len(res) == 1
        assert res[0][0].chunk_id == "DOC-1_chunk_0"
        assert res[0][1] == pytest.approx(1.0, abs=1e-6)


def test_rag_service_concurrent_thread_initialization_runs_once():
    embed_calls = 0
    base_embedder = FakeEmbeddingClient(dimension=64)

    class CountingEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            nonlocal embed_calls
            embed_calls += 1
            return base_embedder.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return base_embedder.embed_query(text)

    docs = [
        ResearchDocument(
            document_id="DOC-INIT-1",
            title="Init Policy",
            content="Concentration limit is 40.0% and minimum cash is 5.0%.",
            metadata={"synthetic": True},
        )
    ]
    service = RagService(
        embedding_client=CountingEmbedder(),
        llm_client=MagicMock(),
        documents=docs,
    )

    def init_worker(_: int) -> int:
        service._ensure_initialized()
        assert service.retrieval_service is not None
        return len(service.retrieval_service.vector_index)

    with ThreadPoolExecutor(max_workers=10) as pool:
        sizes = list(pool.map(init_worker, range(10)))

    assert embed_calls == 1
    assert all(size == 1 for size in sizes)


# ---------------------------------------------------------------------------
# 3. Independent Request State & Telemetry Thread Safety
# ---------------------------------------------------------------------------

def test_concurrent_portfolio_assessments_produce_independent_results():
    def assess_case(idx: int) -> tuple[int, dict]:
        portfolio = Portfolio(
            portfolio_id=f"USER-{idx}",
            cash=float(1000 * (idx + 1)),
            holdings=[
                Holding(symbol=f"SYM{idx}", quantity=10.0, price=100.0),
            ],
        )
        return idx, assess_portfolio_risk(portfolio)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(assess_case, range(20)))

    for idx, result in outcomes:
        assert result["portfolio_id"] == f"USER-{idx}"
        assert result["total_value"] == pytest.approx(1000.0 * (idx + 1) + 1000.0)
        assert result["concentrations"][0]["symbol"] == f"SYM{idx}"


def test_concurrent_agent_workflows_isolate_request_state():
    class EchoLLM:
        def invoke(self, messages):
            user_text = messages[-1].content
            return AIMessage(
                content=f"Handled: {user_text}",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

    def run_case(idx: int) -> tuple[int, dict]:
        query = f"Assess portfolio request #{idx}"
        state = run_llm_portfolio_workflow(user_query=query, llm_client=EchoLLM())
        return idx, state

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run_case, range(12)))

    for idx, state in results:
        assert state["status"] == "completed"
        assert state["user_query"] == f"Assess portfolio request #{idx}"
        assert state["final_response"] == f"Handled: Assess portfolio request #{idx}"

    events = get_recorded_events()
    assert len(events) == 12
    assert all(e.outcome == "success" and e.total_tokens == 15 for e in events)


def test_concurrent_telemetry_recording_does_not_lose_or_corrupt_events():
    def record_one(idx: int) -> None:
        with track_llm_operation(operation=f"op_{idx}", model="gpt-4o-mini") as tracker:
            tracker.record_success(
                response=AIMessage(
                    content="ok",
                    usage_metadata={"input_tokens": 100 + idx, "output_tokens": 20, "total_tokens": 120 + idx},
                )
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record_one, range(25)))

    events = get_recorded_events()
    assert len(events) == 25
    recorded_ops = {e.operation for e in events}
    assert recorded_ops == {f"op_{i}" for i in range(25)}


# ---------------------------------------------------------------------------
# 4. Multi-Process Memory Isolation Diagnostic
# ---------------------------------------------------------------------------

def test_separate_processes_maintain_independent_vector_index_memory():
    """Prove two Python worker processes do not share in-memory vector index state.

    Simulates multi-worker Uvicorn behavior (--workers 2): Process A indexes a custom
    chunk into its singleton RagService, while Process B observes a completely separate
    process memory space with its own PID and unshared index.
    """
    script_worker_a = (
        "import json, os; "
        "from app.models.document import TextChunk; "
        "from app.services.rag_service import get_default_rag_service; "
        "from app.services.retrieval_service import FakeEmbeddingClient, RetrievalService; "
        "svc = get_default_rag_service(); "
        "svc.retrieval_service = RetrievalService(embedding_client=FakeEmbeddingClient(dimension=8)); "
        "chunk = TextChunk(chunk_id='PROC-A_0', document_id='PROC-A', document_title='Worker A Doc', "
        "chunk_index=0, chunk_text='Process local state'); "
        "svc.retrieval_service.vector_index.add(chunk, [1.0] * 8); "
        "print(json.dumps({'pid': os.getpid(), 'svc_id': id(svc), 'index_len': len(svc.retrieval_service.vector_index)}))"
    )
    script_worker_b = (
        "import json, os; "
        "from app.services.rag_service import get_default_rag_service; "
        "svc = get_default_rag_service(); "
        "idx_len = len(svc.retrieval_service.vector_index) if svc.retrieval_service is not None else 0; "
        "print(json.dumps({'pid': os.getpid(), 'svc_id': id(svc), 'index_len': idx_len}))"
    )

    proc_a = subprocess.run(
        [sys.executable, "-c", script_worker_a],
        capture_output=True,
        text=True,
        check=True,
    )
    proc_b = subprocess.run(
        [sys.executable, "-c", script_worker_b],
        capture_output=True,
        text=True,
        check=True,
    )

    data_a = json.loads(proc_a.stdout.strip().splitlines()[-1])
    data_b = json.loads(proc_b.stdout.strip().splitlines()[-1])

    assert data_a["pid"] != data_b["pid"]
    assert data_a["index_len"] == 1
    assert data_b["index_len"] == 0

