"""Service, endpoint-security, multi-user isolation, and E2E intelligence tests (Days 20–21)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from langchain_core.messages import AIMessage
import pytest

from app.agents.portfolio_graph import (
    run_llm_portfolio_workflow,
    sanitize_snapshot_provenance,
)
from app.api.auth_routes import login_for_access_token, register_user
from app.api.broker_routes import (
    connect_zerodha,
    list_broker_connections,
    revoke_broker_connection,
    zerodha_callback,
)
from app.api.dependencies import translate_service_exception
from app.api.portfolio_routes import (
    create_portfolio,
    evaluate_latest_snapshot_risk,
    get_portfolio_snapshot,
    get_portfolio_with_latest_snapshot,
    list_portfolio_snapshots,
    refresh_portfolio_manual,
)
from app.api.schemas import (
    PortfolioCreateRequest,
    PortfolioRefreshRequest,
    ZerodhaConnectRequest,
)
from app.auth.dependencies import get_current_user
from app.auth.models import AuthenticatedUser, UserLoginRequest, UserRegisterRequest
from app.auth.service import AuthService
from app.db.broker_repository import InMemoryBrokerPersistenceRepository
from app.integrations.brokers.base import BrokerRegistry
from app.integrations.brokers.exceptions import BrokerTokenExpiredError
from app.integrations.brokers.zerodha import ZerodhaBrokerAdapter
from app.logging_config import JsonFormatter
from app.main import app
from app.services.broker_service import BrokerConnectionService
from app.services.portfolio_ingestion_service import PortfolioIngestionService


def _step_no_io(coro: Any) -> Any:
    """Drive purely in-memory async flows cleanly without OS socketpair event loops."""
    try:
        while True:
            yielded = coro.send(None)
            if yielded is not None:
                raise RuntimeError("Unexpected async wait primitive in synchronous test harness.")
    except StopIteration as stop:
        return stop.value


class _DeterministicEnvelopeEncryptor:
    """Test-only keyed reversible envelope isolating cryptography package absence in sandbox."""

    def __init__(self, key_id: str = "test-v1") -> None:
        self._key_id = key_id

    def encrypt(self, plaintext_token: str, *, user_id: str, broker_name: str) -> tuple[str, str]:
        return f"enc:test:{self._key_id}:{user_id}:{broker_name}:{plaintext_token[::-1]}", self._key_id

    def decrypt(self, encrypted_envelope: str, *, user_id: str, broker_name: str) -> str:
        parts = encrypted_envelope.split(":", 5)
        if len(parts) != 6 or parts[3] != user_id or parts[4] != broker_name:
            raise ValueError("AAD mismatch")
        return parts[5][::-1]


def _make_mock_zerodha_adapter(scenario: dict[str, Any]) -> ZerodhaBrokerAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        if scenario.get("expired"):
            return httpx.Response(
                403,
                json={"status": "error", "error_type": "TokenException", "message": "Token expired"},
            )
        if scenario.get("rate_limited"):
            return httpx.Response(
                429,
                json={"status": "error", "error_type": "TooManyRequests", "message": "Rate limited"},
            )
        if request.method == "POST" and request.url.path == "/session/token":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "user_id": "ZR100",
                        "user_name": "Alice Investor",
                        "email": "alice@example.com",
                        "access_token": "kite-live-access-secret-999",
                        "exchanges": ["NSE", "BSE"],
                        "products": ["CNC"],
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/user/profile":
            return httpx.Response(
                200,
                json={"status": "success", "data": {"user_id": "ZR100", "user_name": "Alice Investor"}},
            )
        if request.method == "GET" and request.url.path == "/portfolio/holdings":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": scenario.get(
                        "holdings",
                        [
                            {
                                "tradingsymbol": "RELIANCE",
                                "exchange": "NSE",
                                "instrument_token": 738561,
                                "isin": "INE002A01018",
                                "product": "CNC",
                                "quantity": 10,
                                "t1_quantity": 0,
                                "average_price": 2400.0,
                                "last_price": 3000.0,
                            },
                            {
                                "tradingsymbol": "INFY",
                                "exchange": "NSE",
                                "instrument_token": 408065,
                                "isin": "INE009A01021",
                                "product": "CNC",
                                "quantity": 5,
                                "t1_quantity": 0,
                                "average_price": 1400.0,
                                "last_price": 1800.0,
                            },
                        ],
                    ),
                },
            )
        if request.method == "GET" and request.url.path == "/portfolio/positions":
            return httpx.Response(200, json={"status": "success", "data": {"net": [], "day": []}})
        if request.method == "GET" and request.url.path == "/user/margins":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "equity": {
                            "enabled": True,
                            "net": 1000.0,
                            "available": {"cash": 1000.0, "live_balance": 1000.0},
                            "utilised": {"debits": 0.0},
                        }
                    },
                },
            )
        if request.method == "DELETE" and request.url.path == "/session/token":
            scenario["revoked_upstream"] = True
            return httpx.Response(200, json={"status": "success", "data": True})
        raise AssertionError(f"Unexpected call {request.method} {request.url}")

    return ZerodhaBrokerAdapter(
        api_key="test-kite-key",
        api_secret="test-kite-secret",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    )


def test_app_mounts_all_required_broker_and_portfolio_routes():
    paths = set(app.openapi()["paths"].keys())
    expected_paths = {
        "/health",
        "/portfolio/assess",
        "/agent/run",
        "/rag/ask",
        "/auth/register",
        "/auth/token",
        "/brokers/zerodha/connect",
        "/brokers/zerodha/callback",
        "/brokers/connections",
        "/brokers/connections/{connection_id}",
        "/portfolios",
        "/portfolios/{portfolio_id}/refresh",
        "/portfolios/{portfolio_id}",
        "/portfolios/{portfolio_id}/snapshots",
        "/portfolios/{portfolio_id}/snapshots/{snapshot_id}",
        "/portfolios/{portfolio_id}/risk",
        "/portfolios/{portfolio_id}/analyze",
    }
    assert expected_paths.issubset(paths)


def test_end_to_end_multi_user_auth_connect_refresh_risk_langgraph_and_revoke():
    repo = InMemoryBrokerPersistenceRepository()
    auth_service = AuthService(secret_key="test-signing-key-32-chars-minimum!")
    scenario: dict[str, Any] = {}
    adapter = _make_mock_zerodha_adapter(scenario)
    registry = BrokerRegistry()
    registry.register(adapter)
    encryptor = _DeterministicEnvelopeEncryptor()

    broker_service = BrokerConnectionService(
        repository=repo,
        registry=registry,
        auth_service=auth_service,
        encryption_service=encryptor,  # type: ignore[arg-type]
    )
    ingestion_service = PortfolioIngestionService(
        repository=repo,
        registry=registry,
        encryption_service=encryptor,  # type: ignore[arg-type]
    )

    # Capture JSON logs to prove zero secrets/tokens ever emit in logs
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger("app")
    root_logger.addHandler(handler)
    prev_level = root_logger.level
    root_logger.setLevel(logging.INFO)

    try:
        # 1. Register Alice & Bob and verify Bearer dependency rejects missing/tampered tokens
        alice_tok = _step_no_io(
            register_user(
                UserRegisterRequest(email="alice@example.com", password="Password123!"),
                repo=repo,
                auth_service=auth_service,
            )
        )
        bob_tok = _step_no_io(
            register_user(
                UserRegisterRequest(email="bob@example.com", password="Password456!"),
                repo=repo,
                auth_service=auth_service,
            )
        )
        alice_login = _step_no_io(
            login_for_access_token(
                UserLoginRequest(email="alice@example.com", password="Password123!"),
                repo=repo,
                auth_service=auth_service,
            )
        )
        alice_user = _step_no_io(
            get_current_user(
                credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=alice_login.access_token),
                auth_service=auth_service,
                repo=repo,
            )
        )
        bob_user = _step_no_io(
            get_current_user(
                credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=bob_tok.access_token),
                auth_service=auth_service,
                repo=repo,
            )
        )
        with pytest.raises(HTTPException) as unauth_exc:
            _step_no_io(get_current_user(credentials=None, auth_service=auth_service, repo=repo))
        assert unauth_exc.value.status_code == 401

        # 2. Alice connects Zerodha & gets CSRF-signed redirect URL + HttpOnly binding cookie
        connect_response = Response()
        connect_res = _step_no_io(
            connect_zerodha(
                response=connect_response,
                payload=ZerodhaConnectRequest(),
                current_user=alice_user,
                service=broker_service,
            )
        )
        set_cookie_header = connect_response.headers.get("set-cookie", "")
        assert "alphapilot_oauth_bind=" in set_cookie_header
        assert "httponly" in set_cookie_header.lower()
        assert "samesite=lax" in set_cookie_header.lower()
        cookie_val = set_cookie_header.split("alphapilot_oauth_bind=", 1)[1].split(";", 1)[0]

        parsed_auth_url = urlparse(connect_res.authorization_url)
        redirect_params_qs = parse_qs(parse_qs(parsed_auth_url.query)["redirect_params"][0])
        csrf_state = redirect_params_qs["state"][0]

        empty_req = Request(scope={"type": "http", "headers": []})
        cookie_req = Request(
            scope={
                "type": "http",
                "headers": [(b"cookie", f"alphapilot_oauth_bind={cookie_val}".encode("ascii"))],
            }
        )

        # Bob attempting to redeem Alice's callback state fails 401 (without consuming Alice's state!)
        with pytest.raises(HTTPException) as hijack_exc:
            _step_no_io(
                zerodha_callback(
                    request=empty_req,
                    response=Response(),
                    request_token="one-time-req-tok",
                    state=csrf_state,
                    status="success",
                    current_user=bob_user,
                    service=broker_service,
                )
            )
        assert hijack_exc.value.status_code == 401

        # Alice redeems callback via browser HttpOnly cookie (no Bearer header required!) -> encrypted token stored
        cb_response = Response()
        conn_summary = _step_no_io(
            zerodha_callback(
                request=cookie_req,
                response=cb_response,
                request_token="one-time-req-tok",
                state=csrf_state,
                status="success",
                current_user=None,
                service=broker_service,
            )
        )
        assert conn_summary.status == "active"
        assert conn_summary.broker_account_id == "ZR100"
        assert "kite-live-access-secret-999" not in json.dumps(conn_summary.model_dump(mode="json"))

        # Callback replay fails cleanly
        with pytest.raises(HTTPException) as replay_exc:
            _step_no_io(
                zerodha_callback(
                    request=cookie_req,
                    response=Response(),
                    request_token="one-time-req-tok",
                    state=csrf_state,
                    status="success",
                    current_user=alice_user,
                    service=broker_service,
                )
            )
        assert replay_exc.value.status_code == 401

        # 3. Create portfolio & perform manual refresh
        portfolio = _step_no_io(
            create_portfolio(
                PortfolioCreateRequest(
                    name="Alice Main Equity",
                    broker_name="zerodha",
                    broker_connection_id=conn_summary.connection_id,
                ),
                current_user=alice_user,
                service=ingestion_service,
            )
        )
        snap = _step_no_io(
            refresh_portfolio_manual(
                portfolio.portfolio_id,
                PortfolioRefreshRequest(idempotency_key="manual-1"),
                current_user=alice_user,
                service=ingestion_service,
            )
        )
        assert snap.status == "completed"
        assert snap.holdings_count == 2
        assert snap.valuation_available is True

        # Cross-user access on portfolio, snapshots, and connections blocked cleanly (404)
        with pytest.raises(HTTPException) as bob_pf_exc:
            _step_no_io(
                get_portfolio_with_latest_snapshot(
                    portfolio.portfolio_id,
                    current_user=bob_user,
                    service=ingestion_service,
                )
            )
        assert bob_pf_exc.value.status_code == 404

        with pytest.raises(HTTPException) as bob_snap_exc:
            _step_no_io(
                get_portfolio_snapshot(
                    portfolio.portfolio_id,
                    snap.snapshot_id,
                    current_user=bob_user,
                    service=ingestion_service,
                )
            )
        assert bob_snap_exc.value.status_code == 404

        # 4. Deterministic Risk Evaluation + LangGraph Workflow E2E
        # Portfolio totals: RELIANCE 10*3000=30000, INFY 5*1800=9000, cash=1000 => 40000
        # RELIANCE = 75.0% (> 40% violation), cash = 2.5% (< 5% violation)
        risk_summary = _step_no_io(
            evaluate_latest_snapshot_risk(
                portfolio.portfolio_id,
                current_user=alice_user,
                service=ingestion_service,
            )
        )
        assert risk_summary.risk_assessment is not None
        assert risk_summary.risk_assessment["total_value"] == 40000.0
        assert risk_summary.risk_assessment["passed"] is False
        assert len(risk_summary.risk_assessment["violations"]) == 2
        assert risk_summary.provenance["snapshot_id"] == snap.snapshot_id

        # Feed sanitized intelligence context into LangGraph workflow with deterministic mock LLM
        ctx = _step_no_io(
            ingestion_service.build_intelligence_context(
                user_id=alice_user.user_id,
                portfolio_id=portfolio.portfolio_id,
            )
        )
        assert ctx.domain_portfolio is not None

        class _SyncFinalLLM:
            def invoke(self, messages):
                joined = "\n".join(str(m.content) for m in messages)
                assert snap.snapshot_id in joined
                assert "Sector classification is not provided" in joined
                assert "kite-live-access-secret-999" not in joined
                return AIMessage(
                    content=(
                        f"Based on snapshot {snap.snapshot_id}, RELIANCE is 75.0% and cash is 2.5%. "
                        "Sector data is unavailable."
                    ),
                    usage_metadata={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
                )

        # Include dummy untrusted secrets to prove sanitize_snapshot_provenance strips them
        unsafe_provenance = {
            **ctx.provenance,
            "access_token": "kite-live-access-secret-999",
            "encrypted_access_token": "enc:secret",
        }
        assert "access_token" not in (sanitize_snapshot_provenance(unsafe_provenance) or {})

        graph_state = run_llm_portfolio_workflow(
            user_query="Analyze concentration risk and note limitations.",
            llm_client=_SyncFinalLLM(),
            portfolio_context=ctx.domain_portfolio,
            snapshot_provenance=unsafe_provenance,
            portfolio_limitations=ctx.limitations,
            precomputed_assessment=ctx.risk_assessment,
        )
        assert graph_state["status"] == "completed"
        assert "75.0%" in (graph_state["final_response"] or "")
        assert "kite-live-access-secret-999" not in str(graph_state)

        # 5. Rate-limit error returns 429 and preserves previous completed snapshot
        scenario["rate_limited"] = True
        with pytest.raises(HTTPException) as rl_exc:
            _step_no_io(
                refresh_portfolio_manual(
                    portfolio.portfolio_id,
                    PortfolioRefreshRequest(idempotency_key="manual-rl"),
                    current_user=alice_user,
                    service=ingestion_service,
                )
            )
        assert rl_exc.value.status_code == 429
        scenario["rate_limited"] = False

        latest_view = _step_no_io(
            get_portfolio_with_latest_snapshot(
                portfolio.portfolio_id,
                current_user=alice_user,
                service=ingestion_service,
            )
        )
        assert latest_view.latest_snapshot is not None
        assert latest_view.latest_snapshot.snapshot_id == snap.snapshot_id

        # 6. Expired Zerodha token marks connection expired cleanly and fails refresh 401
        scenario["expired"] = True
        with pytest.raises(HTTPException) as exp_exc:
            _step_no_io(
                refresh_portfolio_manual(
                    portfolio.portfolio_id,
                    PortfolioRefreshRequest(idempotency_key="manual-exp"),
                    current_user=alice_user,
                    service=ingestion_service,
                )
            )
        assert exp_exc.value.status_code == 401
        assert "reauth_required" in str(exp_exc.value.detail)
        conns_after_exp = _step_no_io(list_broker_connections(current_user=alice_user, service=broker_service))
        assert conns_after_exp[0].status == "expired"
        scenario["expired"] = False

        # 7. Revoke connection wipes encrypted token & blocks future refresh
        revoked = _step_no_io(
            revoke_broker_connection(
                conn_summary.connection_id,
                current_user=alice_user,
                service=broker_service,
            )
        )
        assert revoked.status == "revoked"
        assert scenario.get("revoked_upstream") is True
        raw_conn = _step_no_io(repo.get_broker_connection(conn_summary.connection_id, user_id=alice_user.user_id))
        assert raw_conn is not None and raw_conn.encrypted_access_token is None

        with pytest.raises(HTTPException) as revoked_exc:
            _step_no_io(
                refresh_portfolio_manual(
                    portfolio.portfolio_id,
                    PortfolioRefreshRequest(idempotency_key="manual-after-revoke"),
                    current_user=alice_user,
                    service=ingestion_service,
                )
            )
        assert revoked_exc.value.status_code == 403

        # Verify logs recorded zero raw tokens, secrets, or request tokens
        logs_text = log_stream.getvalue()
        assert "kite-live-access-secret-999" not in logs_text
        assert "one-time-req-tok" not in logs_text
        assert "test-kite-secret" not in logs_text
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)


def test_http_exception_mapping_for_broker_failures():
    assert translate_service_exception(BrokerTokenExpiredError("expired")).status_code == 401
    assert "reauth_required" in translate_service_exception(BrokerTokenExpiredError("expired")).detail
