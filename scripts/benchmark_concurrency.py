"""Controlled synthetic concurrency benchmark for AlphaPilot AI.

Measures sequential vs. concurrent async RAG execution using a deterministic fake LLM
with fixed simulated network I/O delay (no OpenAI credentials or API billing).
Does NOT represent production OpenAI throughput, RPM/TPM limits, or real latency.

Run outside sandboxed loopback environments:
    .venv\\Scripts\\python.exe -m scripts.benchmark_concurrency
"""

import asyncio
import statistics
import time

from langchain_core.messages import AIMessage

from app.observability import clear_recorded_events
from app.services.rag_service import RagService
from app.services.resilience import LLMConcurrencyLimiter
from app.services.retrieval_service import FakeEmbeddingClient


class SimulatedAsyncLLM:
    """Fake async LLM simulating external network I/O latency via asyncio.sleep."""

    def __init__(self, latency_seconds: float = 0.05) -> None:
        self.latency_seconds = latency_seconds

    async def ainvoke(self, messages):
        await asyncio.sleep(self.latency_seconds)
        return AIMessage(
            content="Minimum cash reserve is 5.0% per [SYNTH-DOC-002].",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


async def run_benchmark(
    num_requests: int = 20,
    simulated_llm_latency_seconds: float = 0.05,
    max_concurrent: int = 10,
) -> None:
    fake_llm = SimulatedAsyncLLM(latency_seconds=simulated_llm_latency_seconds)
    limiter = LLMConcurrencyLimiter(max_concurrent=max_concurrent, queue_timeout_seconds=10.0)
    service = RagService(
        llm_client=fake_llm,
        embedding_client=FakeEmbeddingClient(dimension=64),
        concurrency_limiter=limiter,
    )
    await service._ensure_initialized_async()
    question = "What is the minimum cash reserve requirement?"

    async def timed_request() -> tuple[float, bool]:
        start = time.perf_counter()
        res = await service.answer_question_async(question=question, top_k=2)
        elapsed = time.perf_counter() - start
        return elapsed, (res.status != "answered")

    # 1. Sequential baseline
    clear_recorded_events()
    seq_start = time.perf_counter()
    seq_results = [await timed_request() for _ in range(num_requests)]
    seq_total = time.perf_counter() - seq_start

    # 2. Concurrent execution
    clear_recorded_events()
    conc_start = time.perf_counter()
    conc_results = await asyncio.gather(*(timed_request() for _ in range(num_requests)))
    conc_total = time.perf_counter() - conc_start

    def summarize(label: str, total: float, results: list[tuple[float, bool]]) -> None:
        latencies_ms = [lat * 1000.0 for lat, _ in results]
        failures = sum(1 for _, failed in results if failed)
        print(f"\n[{label}]")
        print(f"  Requests:             {len(results)}")
        print(f"  Total execution time: {total:.4f}s")
        print(f"  Avg latency:          {statistics.mean(latencies_ms):.2f} ms")
        print(f"  Max latency:          {max(latencies_ms):.2f} ms")
        print(f"  Failure count:        {failures}")
        print(f"  Concurrency cap:      {max_concurrent}")

    print("=" * 72)
    print("ALPHAPILOT AI — SYNTHETIC LOCAL CONCURRENCY BENCHMARK")
    print(f"Simulated LLM delay={simulated_llm_latency_seconds}s | FakeEmbeddingClient")
    print("=" * 72)
    summarize("SEQUENTIAL", seq_total, seq_results)
    summarize("CONCURRENT (asyncio.gather)", conc_total, conc_results)
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

