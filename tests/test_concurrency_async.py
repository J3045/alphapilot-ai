"""Asynchronous concurrency, backpressure, and FastAPI lifecycle tests for Day 14.

NOTE: These tests exercise `asyncio.run(...)`, `asyncio.gather(...)`, and `httpx.AsyncClient`
over ASGI. In Windows environments with strict loopback socket sandboxing where
`asyncio.new_event_loop()` is blocked, run this suite outside the restricted sandbox:
    .venv\\Scripts\\python.exe -m pytest -v tests/test_concurrency_async.py
"""

import asyncio
from unittest.mock import MagicMock

import httpx
from langchain_core.messages import AIMessage
import pytest

from app.agents.portfolio_graph import run_llm_portfolio_workflow_async
from app.main import app
from app.models.document import ResearchDocument
from app.observability import clear_recorded_events, get_recorded_events
from app.services.rag_service import RagService
from app.services.resilience import (
    ConcurrencyLimitExceededError,
    LLMConcurrencyLimiter,
    WorkflowDeadline,
    WorkflowDeadlineExceededError,
    execute_with_retry_async,
)
from app.services.retrieval_service import FakeEmbeddingClient
from tests.test_resilience import SyntheticRateLimitError


@pytest.fixture(autouse=True)
def reset_telemetry():
    clear_recorded_events()
    yield
    clear_recorded_events()


class FakeAsyncRAGLLM:
    """Deterministic async LLM tracking concurrency and preserving question isolation."""

    def __init__(self, delay_seconds: float = 0.03) -> None:
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_seconds)
            user_text = messages[-1].content
            question = user_text.split("--- QUESTION ---")[-1].strip()
            return AIMessage(
                content=f"Grounded answer for: {question} [SYNTH-DOC-002]",
                usage_metadata={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
            )
        finally:
            self.active -= 1


class FakeAsyncAgentLLM:
    """Deterministic async tool-calling LLM for portfolio agent concurrency tests."""

    def __init__(self, delay_seconds: float = 0.02) -> None:
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, messages):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_seconds)
            last_msg = messages[-1]
            # If tool output already exists, synthesize final answer
            if getattr(last_msg, "type", None) == "tool":
                return AIMessage(
                    content=f"Final summary: {last_msg.content}",
                    usage_metadata={"input_tokens": 60, "output_tokens": 20, "total_tokens": 80},
                )

            # Extract portfolio ID marker from user query to prove isolation
            query = str(last_msg.content)
            pid = query.split("PORTFOLIO:")[-1].strip() if "PORTFOLIO:" in query else "DEFAULT-P"
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{pid}",
                        "name": "assess_portfolio_risk",
                        "args": {
                            "portfolio": {
                                "portfolio_id": pid,
                                "cash": 2000.0,
                                "holdings": [
                                    {"symbol": "ALFA", "quantity": 10.0, "price": 100.0},
                                    {"symbol": "BETA", "quantity": 5.0, "price": 200.0},
                                ],
                            }
                        },
                    }
                ],
            )
        finally:
            self.active -= 1


# ---------------------------------------------------------------------------
# 1 & 3 & 9. Concurrent RAG Requests, Isolation & Source Attribution
# ---------------------------------------------------------------------------

def test_concurrent_rag_requests_preserve_isolation_and_sources():
    async def scenario():
        fake_llm = FakeAsyncRAGLLM(delay_seconds=0.04)
        service = RagService(
            llm_client=fake_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
            concurrency_limiter=LLMConcurrencyLimiter(max_concurrent=10, queue_timeout_seconds=2.0),
        )
        questions = [
            "What is the minimum cash reserve requirement?",
            "What is the single security concentration ceiling?",
            "How does risk escalation work?",
            "When must cash deployment be staged?",
            "What is the sector exposure boundary?",
        ]
        results = await asyncio.gather(
            *(service.answer_question_async(q, top_k=2) for q in questions)
        )
        return fake_llm, questions, results

    fake_llm, questions, results = asyncio.run(scenario())
    assert fake_llm.max_active > 1
    assert fake_llm.calls == len(questions)

    for q, res in zip(questions, results):
        assert res.status == "answered"
        assert q in res.answer
        assert len(res.sources) > 0
        assert res.error is None

    events = get_recorded_events()
    assert len(events) == len(questions)
    assert all(e.outcome == "success" and e.total_tokens == 75 for e in events)


# ---------------------------------------------------------------------------
# 2 & 3. Concurrent Portfolio Agent Workflows & Independent State
# ---------------------------------------------------------------------------

def test_concurrent_agent_workflows_preserve_independent_portfolios():
    async def scenario():
        fake_llm = FakeAsyncAgentLLM(delay_seconds=0.03)
        limiter = LLMConcurrencyLimiter(max_concurrent=8, queue_timeout_seconds=2.0)
        pids = [f"CONC-{i:03d}" for i in range(6)]
        states = await asyncio.gather(
            *(
                run_llm_portfolio_workflow_async(
                    user_query=f"Assess PORTFOLIO:{pid}",
                    llm_client=fake_llm,
                    concurrency_limiter=limiter,
                )
                for pid in pids
            )
        )
        return fake_llm, pids, states

    fake_llm, pids, states = asyncio.run(scenario())
    assert fake_llm.max_active > 1

    for pid, state in zip(pids, states):
        assert state["status"] == "completed"
        assert state["assessment_result"] is not None
        assert state["assessment_result"]["portfolio_id"] == pid
        assert pid in (state["final_response"] or "")


# ---------------------------------------------------------------------------
# 4. Safe Async Index Initialization (No Duplicate Indexing Under Burst)
# ---------------------------------------------------------------------------

def test_concurrent_async_rag_cold_start_initializes_index_once():
    async def scenario():
        base_embedder = FakeEmbeddingClient(dimension=64)
        embed_doc_calls = 0

        class SlowCountingAsyncEmbedder:
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                nonlocal embed_doc_calls
                embed_doc_calls += 1
                await asyncio.sleep(0.03)
                return await base_embedder.aembed_documents(texts)

            async def aembed_query(self, text: str) -> list[float]:
                return await base_embedder.aembed_query(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return base_embedder.embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:
                return base_embedder.embed_query(text)

        docs = [
            ResearchDocument(
                document_id="COLD-001",
                title="Cold Start Policy",
                content="Minimum cash reserve requirement is 5.0% for all accounts.",
                metadata={"synthetic": True},
            )
        ]
        service = RagService(
            llm_client=FakeAsyncRAGLLM(delay_seconds=0.01),
            embedding_client=SlowCountingAsyncEmbedder(),
            documents=docs,
            concurrency_limiter=LLMConcurrencyLimiter(max_concurrent=10),
        )

        results = await asyncio.gather(
            *(service.answer_question_async("What is the minimum cash reserve?", top_k=1) for _ in range(8))
        )
        return embed_doc_calls, len(service.retrieval_service.vector_index), results

    embed_doc_calls, index_size, results = asyncio.run(scenario())
    assert embed_doc_calls == 1
    assert index_size == 1
    assert all(r.status == "answered" for r in results)


# ---------------------------------------------------------------------------
# 5. Sync-Only Client Fallback via asyncio.to_thread
# ---------------------------------------------------------------------------

def test_async_service_falls_back_to_thread_for_sync_only_clients():
    async def scenario():
        sync_llm = MagicMock()
        sync_llm.invoke.return_value = AIMessage(
            content="Minimum cash is 5.0% per [SYNTH-DOC-002].",
            usage_metadata={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        )
        service = RagService(
            llm_client=sync_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
            concurrency_limiter=LLMConcurrencyLimiter(max_concurrent=4),
        )
        res = await service.answer_question_async("What is the cash reserve requirement?", top_k=2)
        return sync_llm, res

    sync_llm, res = asyncio.run(scenario())
    assert res.status == "answered"
    assert sync_llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# 6. Async Retry & Workflow Deadline Compatibility
# ---------------------------------------------------------------------------

def test_execute_with_retry_async_recovers_and_respects_deadline():
    async def scenario():
        attempts = 0
        sleeps: list[float] = []

        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise SyntheticRateLimitError("429 Too Many Requests")
            return "recovered"

        async def fake_sleep(delay: float):
            sleeps.append(delay)

        deadline = WorkflowDeadline(timeout_seconds=10.0)
        val = await execute_with_retry_async(
            fn=flaky,
            max_retries=2,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=1.0,
            sleep_fn=fake_sleep,
            rng=lambda _low, high: high,
            deadline=deadline,
        )
        return attempts, sleeps, val

    attempts, sleeps, val = asyncio.run(scenario())
    assert val == "recovered"
    assert attempts == 2
    assert sleeps == [pytest.approx(0.25)]


def test_async_workflow_respects_pre_expired_deadline():
    async def scenario():
        expired = WorkflowDeadline(timeout_seconds=0.0)
        fake_llm = FakeAsyncRAGLLM()
        service = RagService(
            llm_client=fake_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
        )
        # Ensure expired deadline triggers controlled failed status without invoking LLM
        res = await service.answer_question_async(
            "What is the cash reserve?",
            workflow_timeout_seconds=0.0,
        )
        return fake_llm, res

    fake_llm, res = asyncio.run(scenario())
    assert res.status == "failed"
    assert "deadline exceeded" in (res.error or "").lower()
    assert fake_llm.calls == 0


# ---------------------------------------------------------------------------
# 7 & 8. Bounded Concurrency Enforcement & Backpressure Saturation
# ---------------------------------------------------------------------------

def test_concurrency_limiter_caps_in_flight_llm_calls():
    async def scenario():
        fake_llm = FakeAsyncRAGLLM(delay_seconds=0.04)
        limiter = LLMConcurrencyLimiter(max_concurrent=2, queue_timeout_seconds=2.0)
        service = RagService(
            llm_client=fake_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
            concurrency_limiter=limiter,
        )
        results = await asyncio.gather(
            *(service.answer_question_async("What is the cash reserve?", top_k=1) for _ in range(6))
        )
        return fake_llm, results

    fake_llm, results = asyncio.run(scenario())
    assert all(r.status == "answered" for r in results)
    assert fake_llm.max_active <= 2


def test_backpressure_rejects_excess_requests_when_queue_budget_exhausted():
    async def scenario():
        fake_llm = FakeAsyncRAGLLM(delay_seconds=0.25)
        limiter = LLMConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.04)
        service = RagService(
            llm_client=fake_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
            concurrency_limiter=limiter,
        )
        # Pre-build index so both requests compete directly on the LLM semaphore
        await service._ensure_initialized_async()

        results = await asyncio.gather(
            service.answer_question_async("What is the cash reserve?", top_k=1),
            service.answer_question_async("What is the concentration ceiling?", top_k=1),
        )
        return results

    results = asyncio.run(scenario())
    statuses = sorted(r.status for r in results)
    assert statuses == ["answered", "failed"]
    failed = next(r for r in results if r.status == "failed")
    assert "capacity temporarily saturated" in (failed.error or "").lower()

    events = get_recorded_events()
    assert any(e.error_category == "backpressure" for e in events)


def test_concurrency_limiter_integrates_with_workflow_deadline():
    async def scenario():
        limiter = LLMConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=5.0)
        short_deadline = WorkflowDeadline(timeout_seconds=0.03)

        async def hold_slot():
            async with limiter.acquire():
                await asyncio.sleep(0.15)

        holder = asyncio.create_task(hold_slot())
        await asyncio.sleep(0.005)
        try:
            with pytest.raises((WorkflowDeadlineExceededError, ConcurrencyLimitExceededError)):
                async with limiter.acquire(deadline=short_deadline):
                    pass
        finally:
            await holder

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 10. Non-Blocking FastAPI ASGI Endpoint Concurrency
# ---------------------------------------------------------------------------

def test_fastapi_endpoints_remain_responsive_during_slow_rag_call(monkeypatch):
    async def scenario():
        slow_llm = FakeAsyncRAGLLM(delay_seconds=0.20)
        rag_service = RagService(
            llm_client=slow_llm,
            embedding_client=FakeEmbeddingClient(dimension=64),
        )
        await rag_service._ensure_initialized_async()
        monkeypatch.setattr("app.main.get_default_rag_service", lambda: rag_service)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rag_task = asyncio.create_task(
                client.post("/rag/ask", json={"question": "What is the minimum cash reserve?", "top_k": 2})
            )
            await asyncio.sleep(0.01)

            health_resp = await client.get("/health")
            assess_resp = await client.post(
                "/portfolio/assess",
                json={
                    "portfolio_id": "FAST-001",
                    "cash": 2000.0,
                    "holdings": [
                        {"symbol": "ALFA", "quantity": 10.0, "price": 100.0},
                        {"symbol": "BETA", "quantity": 5.0, "price": 200.0},
                    ],
                },
            )
            rag_resp = await rag_task
            return health_resp, assess_resp, rag_resp

    health_resp, assess_resp, rag_resp = asyncio.run(scenario())
    assert health_resp.status_code == 200
    assert assess_resp.status_code == 200
    assert rag_resp.status_code == 200
    assert rag_resp.json()["status"] == "answered"

