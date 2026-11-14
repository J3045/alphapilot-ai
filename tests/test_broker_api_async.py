"""Full ASGI HTTP endpoint & async LangGraph integration tests for Days 20–21.

NOTE: Exercises `asyncio.run(...)` and `httpx.AsyncClient(transport=ASGITransport(app=app))`.
In Windows sandboxes where loopback `socket.socketpair()` / `asyncio.new_event_loop()` is blocked,
run outside the restricted sandbox:
    .venv\\Scripts\\python.exe -m pytest -v tests/test_broker_api_async.py
"""

import asyncio
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from langchain_core.messages import AIMessage

from app.auth.service import AuthService
from app.db.broker_repository import InMemoryBrokerPersistenceRepository
from app.integrations.brokers.base import BrokerRegistry
from app.main import app
from tests.test_broker_api_and_e2e_intelligence import (
    _DeterministicEnvelopeEncryptor,
    _make_mock_zerodha_adapter,
)


def test_fastapi_asgi_full_broker_and_analyze_roundtrip():
    async def _run() -> None:
        repo = InMemoryBrokerPersistenceRepository()
        auth = AuthService(secret_key="asgi-test-signing-key-32-chars-ok!!")
        scenario: dict[str, Any] = {}
        registry = BrokerRegistry()
        registry.register(_make_mock_zerodha_adapter(scenario))

        class _FakeAsyncLLM:
            async def ainvoke(self, messages):
                return AIMessage(
                    content="ASGI snapshot analysis complete: RELIANCE concentration 75.0%, sector missing.",
                    usage_metadata={"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
                )

        app.state.broker_repository = repo
        app.state.auth_service = auth
        app.state.broker_registry = registry
        app.state.token_encryption_service = _DeterministicEnvelopeEncryptor()
        app.state.agent_llm_client = _FakeAsyncLLM()

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                reg = await client.post(
                    "/auth/register",
                    json={"email": "asgi@example.com", "password": "Password123!"},
                )
                assert reg.status_code == 201
                headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

                # Unauthenticated access rejected
                unauth = await client.get("/brokers/connections")
                assert unauth.status_code == 401

                connect_resp = await client.post("/brokers/zerodha/connect", json={}, headers=headers)
                assert connect_resp.status_code == 200
                parsed = urlparse(connect_resp.json()["authorization_url"])
                state = parse_qs(parse_qs(parsed.query)["redirect_params"][0])["state"][0]

                cb = await client.get(
                    "/brokers/zerodha/callback",
                    params={"request_token": "req-asgi-1", "state": state, "status": "success"},
                    headers=headers,
                )
                assert cb.status_code == 200
                conn_id = cb.json()["connection_id"]

                pf = await client.post(
                    "/portfolios",
                    json={
                        "name": "ASGI Portfolio",
                        "broker_name": "zerodha",
                        "broker_connection_id": conn_id,
                    },
                    headers=headers,
                )
                assert pf.status_code == 201
                pid = pf.json()["portfolio_id"]

                ref = await client.post(
                    f"/portfolios/{pid}/refresh",
                    json={"idempotency_key": "asgi-1"},
                    headers=headers,
                )
                assert ref.status_code == 200
                assert ref.json()["status"] == "completed"

                analyzed = await client.post(
                    f"/portfolios/{pid}/analyze",
                    json={"query": "Evaluate concentration risk on my synced portfolio"},
                    headers=headers,
                )
                assert analyzed.status_code == 200
                body = analyzed.json()
                assert body["status"] == "completed"
                assert body["risk_assessment"]["total_value"] == 40000.0
                assert "sector" in " ".join(body["limitations"]).lower()
                assert "kite-live-access-secret-999" not in str(body)
        finally:
            app.state.broker_repository = None
            app.state.auth_service = None
            app.state.broker_registry = None
            app.state.token_encryption_service = None
            app.state.agent_llm_client = None

    asyncio.run(_run())

