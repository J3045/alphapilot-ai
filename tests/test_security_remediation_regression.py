"""Targeted regression suite for Days 16-21 security and architecture remediations.

Designed to run hermetically in offline sandboxes via synchronous stepping for async
service/route coroutines while exercising real FastAPI handlers, Starlette requests,
in-memory repositories, httpx MockTransport, and deterministic encryption stubs.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import hmac
from http.cookies import SimpleCookie
import io
import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
import httpx
import pytest
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import RedirectResponse

from app.agents.portfolio_graph import run_portfolio_workflow
from app.api.auth_routes import register_user
from app.api.broker_routes import OAUTH_BINDING_COOKIE_NAME, connect_zerodha, zerodha_callback
from app.api.portfolio_routes import _to_snapshot_response, refresh_portfolio_manual
from app.api.schemas import PortfolioRefreshRequest
from app.auth.dependencies import get_current_user, get_optional_current_user
from app.auth.models import AuthenticatedUser, UserRegisterRequest
from app.auth.service import AuthError, AuthService
from app.config import settings
from app.db.broker_repository import InMemoryBrokerPersistenceRepository
from app.integrations.brokers.base import BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConfigurationError,
    BrokerRateLimitError,
    BrokerResponseValidationError,
    BrokerTimeoutError,
)
from app.integrations.brokers.models import POSITIONS_EXCLUSION_LIMITATION
from app.integrations.brokers.zerodha import ZerodhaBrokerAdapter
from app.integrations.normalizers.portfolio_normalizer import ZerodhaPortfolioNormalizer
from app.integrations.security.encryption import (
    TokenEncryptionError,
    TokenEncryptionService,
    parse_encryption_key_ring,
)
from app.logging_config import JsonFormatter, redact_sensitive_text
from app.models.portfolio import Holding, Portfolio
from app.services.broker_service import BrokerConnectionService
from app.services.portfolio_ingestion_service import PortfolioIngestionService
from app.services.resilience import WorkflowDeadline, workflow_deadline_scope
from app.services.risk_engine import assess_portfolio_risk
from tests.pg_test_utils import require_isolated_pg_test_url


def _step_no_io(coro: Any) -> Any:
    """Step an in-memory coroutine to completion without starting an asyncio event loop."""
    try:
        while True:
            yielded = coro.send(None)
            if yielded is not None:
                raise RuntimeError("Unexpected async wait primitive inside synchronous test harness.")
    except StopIteration as stop:
        return stop.value


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _mint_raw_hs256_jwt(payload: dict[str, Any], secret: str) -> str:
    header_b64 = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig_b64 = _b64url(hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest())
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _make_starlette_request(cookie_header: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie_header:
        headers.append((b"cookie", cookie_header.encode("latin-1")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/brokers/zerodha/callback",
            "headers": headers,
        }
    )


class _StubEnvelopeEncryptor:
    """Deterministic key-versioned envelope stub when cryptography is absent in sandbox."""

    def __init__(self, active_key_id: str = "v1") -> None:
        self.active_key_id = active_key_id
        self._seq = 0

    def extract_key_id(self, encrypted_envelope: str) -> str | None:
        parts = encrypted_envelope.split(":")
        return parts[2] if len(parts) == 5 and parts[0] == "enc" else None

    def needs_rotation(self, encrypted_envelope: str) -> bool:
        kid = self.extract_key_id(encrypted_envelope)
        return kid is None or kid != self.active_key_id

    def encrypt(self, plaintext: str, *, user_id: str, broker_name: str) -> tuple[str, str]:
        self._seq += 1
        token_b64 = _b64url(f"{user_id}|{broker_name}|{plaintext}".encode("utf-8"))
        return f"enc:v1:{self.active_key_id}:nonce{self._seq}:{token_b64}", self.active_key_id

    def decrypt(self, encrypted_envelope: str, *, user_id: str, broker_name: str) -> str:
        parts = encrypted_envelope.split(":")
        if len(parts) != 5 or parts[0] != "enc":
            raise TokenEncryptionError("Malformed envelope")
        raw = base64.urlsafe_b64decode(parts[4] + "=" * ((-len(parts[4])) % 4)).decode("utf-8")
        uid, bname, secret = raw.split("|", 2)
        if uid != user_id or bname != broker_name:
            raise TokenEncryptionError("AAD mismatch")
        return secret


# ---------------------------------------------------------------------------
# Priority 1A: JWT bool claims + DB-backed active/deleted enforcement
# ---------------------------------------------------------------------------


def test_jwt_rejects_boolean_temporal_claims_and_enforces_db_user_state():
    secret = "test-super-secret-signing-key-32-bytes!!"
    auth = AuthService(secret_key=secret)
    repo = InMemoryBrokerPersistenceRepository()

    token_resp = _step_no_io(
        register_user(
            UserRegisterRequest(email="remedy@example.com", password="Password123!"),
            repo=repo,
            auth_service=auth,
        )
    )
    user_id = token_resp.user_id
    now_ts = int(datetime.now(timezone.utc).timestamp())

    # Boolean exp / iat / nbf must all be rejected cleanly
    for bad_claims in (
        {"sub": user_id, "email": "remedy@example.com", "token_use": "access", "exp": True, "iat": now_ts},
        {"sub": user_id, "email": "remedy@example.com", "token_use": "access", "exp": False, "iat": now_ts},
        {"sub": user_id, "email": "remedy@example.com", "token_use": "access", "exp": now_ts + 300, "iat": True},
        {"sub": user_id, "email": "remedy@example.com", "token_use": "access", "exp": now_ts + 300, "nbf": False},
    ):
        forged_bool_jwt = _mint_raw_hs256_jwt(bad_claims, secret)
        with pytest.raises(AuthError):
            auth.verify_access_token(forged_bool_jwt)

    # Forged is_active=True claim must NEVER grant active status without DB check
    forged_active_jwt = _mint_raw_hs256_jwt(
        {
            "sub": user_id,
            "email": "remedy@example.com",
            "token_use": "access",
            "is_active": True,
            "iat": now_ts,
            "exp": now_ts + 300,
        },
        secret,
    )
    verified = auth.verify_access_token(forged_active_jwt)
    assert verified.is_active is False

    # Active DB user succeeds
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=forged_active_jwt)
    active_user = _step_no_io(get_current_user(credentials=creds, auth_service=auth, repo=repo))
    assert active_user.user_id == user_id
    assert active_user.is_active is True

    # Deactivated DB user fails closed with 403 even with valid JWT carrying is_active=True
    deactivated = _step_no_io(repo.set_user_active(user_id, is_active=False))
    assert deactivated is not None and deactivated.is_active is False
    with pytest.raises(HTTPException) as inactive_exc:
        _step_no_io(get_current_user(credentials=creds, auth_service=auth, repo=repo))
    assert inactive_exc.value.status_code == 403

    # Deleted DB user fails closed with 401
    assert _step_no_io(repo.delete_user(user_id)) is True
    with pytest.raises(HTTPException) as deleted_exc:
        _step_no_io(get_current_user(credentials=creds, auth_service=auth, repo=repo))
    assert deleted_exc.value.status_code == 401


def test_get_optional_current_user_fails_closed_when_bearer_present():
    auth = AuthService(secret_key="test-super-secret-signing-key-32-bytes!!")
    repo = InMemoryBrokerPersistenceRepository()

    # Absent credentials returns None
    assert _step_no_io(get_optional_current_user(credentials=None, auth_service=auth, repo=repo)) is None

    # Present malformed credentials MUST raise 401, never silently degrade to None
    bad_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tampered.jwt.value")
    with pytest.raises(HTTPException) as exc:
        _step_no_io(get_optional_current_user(credentials=bad_creds, auth_service=auth, repo=repo))
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Priority 1B & 1C: Persistent OAuth state, cookie binding & optional redirect
# ---------------------------------------------------------------------------


def _build_zerodha_stack(
    *,
    scenario: dict[str, Any],
    encryptor: Any | None = None,
    sleep_fn: Any | None = None,
    max_retries: int = 0,
    max_backoff_seconds: float = 2.0,
) -> tuple[
    InMemoryBrokerPersistenceRepository,
    AuthService,
    ZerodhaBrokerAdapter,
    BrokerConnectionService,
    PortfolioIngestionService,
    Any,
]:
    repo = InMemoryBrokerPersistenceRepository()
    auth = AuthService(secret_key="test-super-secret-signing-key-32-bytes!!")

    def handler(request: httpx.Request) -> httpx.Response:
        scenario["http_calls"] = scenario.get("http_calls", 0) + 1
        if request.method == "POST" and request.url.path == "/session/token":
            if scenario.get("fail_exchange"):
                return httpx.Response(401, json={"status": "error", "error_type": "TokenException", "message": "bad"})
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "user_id": scenario.get("broker_account_id", "ZX1001"),
                        "user_name": "Alice Investor",
                        "email": "alice@example.com",
                        "access_token": scenario.get("access_token", "kite-access-tok-secret"),
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/user/profile":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "user_id": scenario.get("profile_user_id", scenario.get("broker_account_id", "ZX1001")),
                        "user_name": "Alice Investor",
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/portfolio/holdings":
            if scenario.get("fail_holdings_error"):
                return httpx.Response(502, text="Bad Gateway")
            return httpx.Response(
                200,
                json={"status": "success", "data": scenario.get("holdings", [])},
            )
        if request.method == "GET" and request.url.path == "/portfolio/positions":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "net": scenario.get("net_positions", []),
                        "day": scenario.get("day_positions", []),
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/user/margins":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "equity": {
                            "enabled": True,
                            "net": scenario.get("cash", 50000.0),
                            "available": {"cash": scenario.get("cash", 50000.0)},
                        }
                    },
                },
            )
        raise AssertionError(f"Unexpected HTTP call: {request.method} {request.url}")

    adapter = ZerodhaBrokerAdapter(
        api_key="kite_api_key",
        api_secret="kite_api_secret_val",
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=max_backoff_seconds,
        sleep_fn=sleep_fn,
    )
    registry = BrokerRegistry()
    registry.register(adapter)
    enc = encryptor if encryptor is not None else _StubEnvelopeEncryptor()
    broker_service = BrokerConnectionService(
        repository=repo,
        registry=registry,
        auth_service=auth,
        encryption_service=enc,
    )
    ingestion_service = PortfolioIngestionService(
        repository=repo,
        registry=registry,
        encryption_service=enc,
    )
    return repo, auth, adapter, broker_service, ingestion_service, enc


def _extract_state_from_auth_url(auth_url: str) -> str:
    outer_q = parse_qs(urlparse(auth_url).query)
    inner_q = parse_qs(outer_q["redirect_params"][0])
    return inner_q["state"][0]


def _parse_oauth_cookie_header(response: Response) -> tuple[str, SimpleCookie]:
    cookies = SimpleCookie()
    set_cookie_values = response.headers.getlist("set-cookie")
    assert set_cookie_values, "Expected Set-Cookie header on connect response"
    for header in set_cookie_values:
        cookies.load(header)
    assert OAUTH_BINDING_COOKIE_NAME in cookies
    return cookies[OAUTH_BINDING_COOKIE_NAME].value, cookies


def test_browser_oauth_cookie_flow_json_default_and_optional_redirect(monkeypatch):
    scenario = {"broker_account_id": "ZX1001"}
    repo, auth, _, broker_service, _, _ = _build_zerodha_stack(scenario=scenario)

    tok = _step_no_io(
        register_user(
            UserRegisterRequest(email="alice@example.com", password="Password123!"),
            repo=repo,
            auth_service=auth,
        )
    )
    alice = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

    # 1. Default JSON callback mode (zerodha_post_callback_redirect_uri = None)
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    connect_resp = Response()
    auth_req = _step_no_io(
        connect_zerodha(
            response=connect_resp,
            payload=None,
            current_user=alice,
            service=broker_service,
        )
    )
    cookie_val, parsed = _parse_oauth_cookie_header(connect_resp)
    morsel = parsed[OAUTH_BINDING_COOKIE_NAME]
    assert bool(morsel["httponly"]) is True
    assert str(morsel["samesite"]).lower() == "lax"
    assert morsel["path"] == "/brokers/zerodha/callback"
    state_token = _extract_state_from_auth_url(auth_req.authorization_url)

    # Simulate browser redirect callback with ONLY HttpOnly cookie (no Bearer token!)
    cb_resp = Response()
    summary = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
            response=cb_resp,
            request_token="one-time-req-tok-1",
            state=state_token,
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    assert summary.status == "active"
    assert summary.broker_name == "zerodha"
    assert "max-age=0" in cb_resp.headers.get("set-cookie", "").lower()

    # Replaying exact same state + cookie fails 401 across workers
    with pytest.raises(HTTPException) as replay_exc:
        _step_no_io(
            zerodha_callback(
                request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
                response=Response(),
                request_token="one-time-req-tok-1",
                state=state_token,
                status="success",
                current_user=None,
                service=broker_service,
            )
        )
    assert replay_exc.value.status_code == 401

    # 2. Optional 302 frontend redirect mode (secret-free URL)
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", "https://ui.alphapilot.test/brokers")
    connect_resp2 = Response()
    auth_req2 = _step_no_io(
        connect_zerodha(response=connect_resp2, payload=None, current_user=alice, service=broker_service)
    )
    cookie_val2, _ = _parse_oauth_cookie_header(connect_resp2)
    state2 = _extract_state_from_auth_url(auth_req2.authorization_url)

    redir = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val2}"),
            response=Response(),
            request_token="one-time-req-tok-2",
            state=state2,
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    assert isinstance(redir, RedirectResponse)
    assert redir.status_code == 302
    loc = redir.headers["location"]
    assert loc.startswith("https://ui.alphapilot.test/brokers?broker=zerodha&status=connected")
    assert "kite-access-tok-secret" not in loc
    assert "one-time-req-tok-2" not in loc
    assert state2 not in loc
    assert "max-age=0" in redir.headers.get("set-cookie", "").lower()


def test_oauth_callback_wrong_binding_fails_without_burning_victim_state(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    scenario = {"broker_account_id": "ZX1002"}
    repo, auth, _, broker_service, _, _ = _build_zerodha_stack(scenario=scenario)

    alice_tok = _step_no_io(
        register_user(UserRegisterRequest(email="alice2@example.com", password="Password123!"), repo=repo, auth_service=auth)
    )
    bob_tok = _step_no_io(
        register_user(UserRegisterRequest(email="bob2@example.com", password="Password456!"), repo=repo, auth_service=auth)
    )
    alice = AuthenticatedUser(user_id=alice_tok.user_id, email=alice_tok.email, is_active=True)
    bob = AuthenticatedUser(user_id=bob_tok.user_id, email=bob_tok.email, is_active=True)

    resp = Response()
    auth_req = _step_no_io(connect_zerodha(response=resp, payload=None, current_user=alice, service=broker_service))
    alice_cookie, _ = _parse_oauth_cookie_header(resp)
    alice_state = _extract_state_from_auth_url(auth_req.authorization_url)

    # Unbound callback (no cookie, no bearer) -> 401
    with pytest.raises(HTTPException) as exc1:
        _step_no_io(
            zerodha_callback(
                request=_make_starlette_request(),
                response=Response(),
                request_token="req-1",
                state=alice_state,
                status="success",
                current_user=None,
                service=broker_service,
            )
        )
    assert exc1.value.status_code == 401

    # Wrong cookie -> 401
    with pytest.raises(HTTPException) as exc2:
        _step_no_io(
            zerodha_callback(
                request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}=forged-wrong-secret"),
                response=Response(),
                request_token="req-1",
                state=alice_state,
                status="success",
                current_user=None,
                service=broker_service,
            )
        )
    assert exc2.value.status_code == 401

    # Cross-user Bearer fallback -> 401
    with pytest.raises(HTTPException) as exc3:
        _step_no_io(
            zerodha_callback(
                request=_make_starlette_request(),
                response=Response(),
                request_token="req-1",
                state=alice_state,
                status="success",
                current_user=bob,
                service=broker_service,
            )
        )
    assert exc3.value.status_code == 401

    # Victim's original state + cookie MUST STILL succeed!
    summary = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={alice_cookie}"),
            response=Response(),
            request_token="req-alice-valid",
            state=alice_state,
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    assert summary.status == "active"


def test_zerodha_build_authorization_url_validates_explicit_redirect_uri(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_redirect_uri", "https://api.alphapilot.test/api/v1/brokers/zerodha/callback")
    adapter = ZerodhaBrokerAdapter(api_key="kite_key", api_secret="kite_sec")

    allowed = adapter.build_authorization_url(
        "state-ok",
        redirect_uri="https://api.alphapilot.test/api/v1/brokers/zerodha/callback",
    )
    assert "state-ok" in allowed

    with pytest.raises(BrokerAuthenticationError):
        adapter.build_authorization_url("state-bad", redirect_uri="https://evil.example/steal")


# ---------------------------------------------------------------------------
# Priority 2: Key-ring strictness, kid validation & CAS re-encryption on read
# ---------------------------------------------------------------------------


def test_encryption_keyring_strict_validation_and_envelope_limits():
    valid_b64 = base64.urlsafe_b64encode(bytes([7]) * 32).decode("ascii")

    # Malformed base64 characters rejected instead of silently ignored
    with pytest.raises(BrokerConfigurationError):
        parse_encryption_key_ring(f"v1:{'!' * 44}")
    with pytest.raises(BrokerConfigurationError):
        parse_encryption_key_ring(f"v1:{valid_b64[:10]} {valid_b64[10:]}")

    # Kid delimiter ':', blank kid, and >64-char kid rejected
    with pytest.raises(BrokerConfigurationError):
        parse_encryption_key_ring(f'{{"bad:kid": "{valid_b64}"}}')
    with pytest.raises(BrokerConfigurationError):
        TokenEncryptionService(keys={"bad:kid": bytes([7]) * 32}, active_key_id="bad:kid")
    with pytest.raises(BrokerConfigurationError):
        TokenEncryptionService(keys={"": bytes([7]) * 32}, active_key_id="")
    with pytest.raises(BrokerConfigurationError):
        TokenEncryptionService(keys={"k" * 65: bytes([7]) * 32}, active_key_id="k" * 65)

    # Short (<16 bytes) ciphertext & non-12-byte nonce & bad base64 rejected in decrypt
    svc = TokenEncryptionService(keys={"v1": bytes([7]) * 32}, active_key_id="v1")
    nonce_12 = base64.urlsafe_b64encode(bytes(12)).decode("ascii")
    short_ct = base64.urlsafe_b64encode(bytes(15)).decode("ascii")
    with pytest.raises(TokenEncryptionError):
        svc.decrypt(f"enc:v1:v1:{nonce_12}:not!valid!b64", user_id="u1", broker_name="zerodha")
    with pytest.raises(TokenEncryptionError):
        svc.decrypt(f"enc:v1:v1:{base64.urlsafe_b64encode(bytes(8)).decode('ascii')}:{short_ct}", user_id="u1", broker_name="zerodha")
    with pytest.raises(TokenEncryptionError):
        svc.decrypt(f"enc:v1:v1:{nonce_12}:{short_ct}", user_id="u1", broker_name="zerodha")


def test_retired_key_auto_reencrypts_on_read_via_cas_and_guards_stale_writes(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    encryptor_v1 = _StubEnvelopeEncryptor(active_key_id="v1")
    scenario = {"broker_account_id": "ZX2001", "holdings": [], "cash": 10000.0}
    repo, auth, adapter, broker_service, _, _ = _build_zerodha_stack(
        scenario=scenario,
        encryptor=encryptor_v1,
    )

    tok = _step_no_io(
        register_user(UserRegisterRequest(email="rot@example.com", password="Password123!"), repo=repo, auth_service=auth)
    )
    user = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

    resp = Response()
    auth_req = _step_no_io(connect_zerodha(response=resp, payload=None, current_user=user, service=broker_service))
    cookie_val, _ = _parse_oauth_cookie_header(resp)
    state_tok = _extract_state_from_auth_url(auth_req.authorization_url)
    conn_summary = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
            response=Response(),
            request_token="req-rot-1",
            state=state_tok,
            status="success",
            current_user=None,
            service=broker_service,
        )
    )

    stored_v1 = _step_no_io(repo.get_broker_connection(conn_summary.connection_id, user_id=user.user_id))
    assert stored_v1 is not None
    assert stored_v1.encryption_key_id == "v1"
    v1_ciphertext = stored_v1.encrypted_access_token
    assert v1_ciphertext is not None and ":v1:" in v1_ciphertext

    # Verify CAS refuses update when expected_encrypted_token does not match
    assert (
        _step_no_io(
            repo.rotate_broker_connection_token(
                conn_summary.connection_id,
                user_id=user.user_id,
                expected_encrypted_token="enc:v1:v1:wrong:wrong",
                new_encrypted_token="should-not-write",
                new_encryption_key_id="v2",
            )
        )
        is None
    )

    # Upgrade active key to v2 and trigger read via PortfolioIngestionService
    encryptor_v2 = _StubEnvelopeEncryptor(active_key_id="v2")
    registry = BrokerRegistry()
    registry.register(adapter)
    ingestion_v2 = PortfolioIngestionService(
        repository=repo,
        registry=registry,
        encryption_service=encryptor_v2,  # type: ignore[arg-type]
    )
    pf = _step_no_io(
        ingestion_v2.create_portfolio(
            user_id=user.user_id,
            name="Rotation Portfolio",
            broker_name="zerodha",
            broker_connection_id=conn_summary.connection_id,
        )
    )
    snap = _step_no_io(
        ingestion_v2.refresh_portfolio(
            user_id=user.user_id,
            portfolio_id=pf.portfolio_id,
            idempotency_key="rot-snap-1",
        )
    )
    assert snap.status == "completed"

    stored_v2 = _step_no_io(repo.get_broker_connection(conn_summary.connection_id, user_id=user.user_id))
    assert stored_v2 is not None
    assert stored_v2.encryption_key_id == "v2"
    assert stored_v2.encrypted_access_token != v1_ciphertext
    assert encryptor_v2.decrypt(
        stored_v2.encrypted_access_token or "",
        user_id=user.user_id,
        broker_name="zerodha",
    ) == "kite-access-tok-secret"


# ---------------------------------------------------------------------------
# Priority 3A/3B/3C/3D: Typed batch parsing, 32-char symbols, ISIN, positions
# ---------------------------------------------------------------------------


def test_zerodha_batch_parsing_keeps_healthy_rows_forces_partial_and_scrubs_secrets():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/portfolio/holdings":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": [
                        {
                            "tradingsymbol": "M&MFIN",
                            "exchange": "NSE",
                            "isin": "INE774D01024",
                            "quantity": 10,
                            "t1_quantity": 0,
                            "average_price": 250.0,
                            "last_price": 300.0,
                        },
                        {
                            "tradingsymbol": "BADQTY",
                            "exchange": "NSE",
                            "quantity": -4,
                            "average_price": 100.0,
                            "last_price": 120.0,
                            "access_token": "leak-must-not-surface",
                        },
                    ],
                },
            )
        if request.url.path == "/portfolio/positions":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "net": [
                            {
                                "tradingsymbol": "INFY",
                                "exchange": "NSE",
                                "product": "MIS",
                                "quantity": -5,
                                "average_price": 1800.0,
                                "last_price": 1790.0,
                                "pnl": 50.0,
                            },
                            {
                                "tradingsymbol": "CORRUPT",
                                "exchange": "NSE",
                                "product": "MIS",
                                "quantity": "not-a-number",
                                "average_price": 10.0,
                            },
                        ],
                        "day": [],
                    },
                },
            )
        raise AssertionError(f"Unexpected path {request.url.path}")

    adapter = ZerodhaBrokerAdapter(
        api_key="k",
        api_secret="s",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    )
    h_batch = _step_no_io(adapter.get_holdings_batch("tok"))
    p_batch = _step_no_io(adapter.get_positions_batch("tok"))
    assert len(h_batch.items) == 1 and h_batch.items[0].symbol == "M&MFIN"
    assert len(h_batch.malformed_records) == 1
    assert len(p_batch.items) == 1 and p_batch.items[0].quantity == Decimal("-5")
    assert len(p_batch.malformed_records) == 1

    normalizer = ZerodhaPortfolioNormalizer()
    res = normalizer.normalize(
        profile=adapter.build_profile("AB1234", "Investor")
        if hasattr(adapter, "build_profile")
        else __import__("app.integrations.brokers.models", fromlist=["BrokerProfile"]).BrokerProfile(
            broker_name="zerodha",
            broker_account_id="AB1234",
        ),
        raw_holdings=h_batch.items,
        raw_positions=p_batch.items,
        pre_malformed_holdings=h_batch.malformed_records,
        pre_malformed_positions=p_batch.malformed_records,
        balance=__import__("app.integrations.brokers.models", fromlist=["BrokerBalance"]).BrokerBalance(
            available_cash=Decimal("1000"),
            currency="INR",
        ),
        partial_policy="mark_partial",
    )
    assert res.status == "partial"
    assert res.valuation_available is False
    assert len(res.holdings) == 1
    assert len(res.positions) == 1
    assert len(res.malformed_records) == 2
    assert "leak-must-not-surface" not in json.dumps(res.malformed_records)

    with pytest.raises(BrokerResponseValidationError):
        normalizer.normalize(
            profile=__import__("app.integrations.brokers.models", fromlist=["BrokerProfile"]).BrokerProfile(
                broker_name="zerodha",
                broker_account_id="AB1234",
            ),
            raw_holdings=h_batch.items,
            pre_malformed_holdings=h_batch.malformed_records,
            balance=__import__("app.integrations.brokers.models", fromlist=["BrokerBalance"]).BrokerBalance(
                available_cash=Decimal("1000"),
                currency="INR",
            ),
            partial_policy="fail",
        )


def test_indian_ticker_symbols_up_to_32_chars_and_injection_rejection():
    for valid in ("M&M", "M&MFIN", "J&KBANK", "MCDOWELL-N", "NIFTY100BEES", "SGBDEC2512-GB", "A" * 32):
        h = Holding(symbol=valid, quantity=1.0, price=100.0)
        assert h.symbol == valid

    for invalid in ("A" * 33, "", "   ", "INFY<script>", "TCS;DROP TABLE", "HDFC BANK"):
        with pytest.raises(ValidationError):
            Holding(symbol=invalid, quantity=1.0, price=100.0)


def test_isin_cross_exchange_aggregation_and_unverified_duplicate_handling():
    from app.integrations.brokers.models import BrokerBalance, BrokerHoldingItem, BrokerProfile

    normalizer = ZerodhaPortfolioNormalizer()
    profile = BrokerProfile(broker_name="zerodha", broker_account_id="ZX300")
    balance = BrokerBalance(available_cash=Decimal("0"), currency="INR")

    # Same verified ISIN across NSE & BSE -> aggregated so 100% single-issuer concentration is caught!
    same_isin = normalizer.normalize(
        profile=profile,
        raw_holdings=[
            BrokerHoldingItem(
                symbol="RELIANCE",
                exchange="NSE",
                isin="INE002A01018",
                quantity=Decimal("4"),
                average_price=Decimal("2400"),
                last_price=Decimal("2500"),
            ),
            BrokerHoldingItem(
                symbol="RELIANCE",
                exchange="BSE",
                isin="INE002A01018",
                quantity=Decimal("6"),
                average_price=Decimal("2450"),
                last_price=Decimal("2600"),
            ),
        ],
        balance=balance,
    )
    domain_agg, _ = same_isin.to_domain_portfolio("P-ISIN-AGG")
    assert domain_agg is not None
    assert len(domain_agg.holdings) == 1
    assert domain_agg.holdings[0].instrument_id == "isin:INE002A01018"
    assert domain_agg.holdings[0].quantity == pytest.approx(10.0)
    assert domain_agg.holdings[0].price == pytest.approx(2560.0)
    risk = assess_portfolio_risk(domain_agg)
    assert risk["passed"] is False
    assert len(risk["violations"]) >= 1

    # Duplicate tradingsymbol across NSE/BSE WITHOUT ISIN -> kept distinct with explicit limitation
    no_isin = normalizer.normalize(
        profile=profile,
        raw_holdings=[
            BrokerHoldingItem(
                symbol="SBIN",
                exchange="NSE",
                isin=None,
                quantity=Decimal("5"),
                average_price=Decimal("700"),
                last_price=Decimal("750"),
            ),
            BrokerHoldingItem(
                symbol="SBIN",
                exchange="BSE",
                isin=None,
                quantity=Decimal("5"),
                average_price=Decimal("700"),
                last_price=Decimal("760"),
            ),
        ],
        balance=balance,
    )
    domain_unverified, notes = no_isin.to_domain_portfolio("P-NO-ISIN")
    assert domain_unverified is not None
    assert len(domain_unverified.holdings) == 2
    assert {h.instrument_id for h in domain_unverified.holdings} == {
        "unverified:NSE:SBIN:0",
        "unverified:BSE:SBIN:1",
    }
    assert any("ISIN" in note for note in notes)

    # Sanity check direct RiskEngine ISIN grouping even when holdings come separately
    direct_pf = Portfolio(
        portfolio_id="P-DIRECT-ISIN",
        holdings=[
            Holding(symbol="TCS", quantity=5.0, price=2000.0, instrument_id="isin:INE467B01029"),
            Holding(symbol="TCS", quantity=5.0, price=2000.0, instrument_id="isin:INE467B01029"),
        ],
        cash=10000.0,
    )
    direct_risk = assess_portfolio_risk(direct_pf)
    assert direct_risk["passed"] is False
    assert len(direct_risk["concentrations"]) == 1
    assert direct_risk["concentrations"][0]["weight_pct"] == pytest.approx(20000.0 / 30000.0 * 100, rel=1e-4)
    assert any("TCS exceeds" in v for v in direct_risk["violations"])


def test_open_positions_persisted_separately_and_excluded_from_holdings_concentration(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    scenario: dict[str, Any] = {
        "broker_account_id": "ZX4001",
        "cash": 20000.0,
        "holdings": [
            {
                "tradingsymbol": "M&M",
                "exchange": "NSE",
                "isin": "INE101A01026",
                "product": "CNC",
                "quantity": 10,
                "t1_quantity": 0,
                "average_price": 2000.0,
                "last_price": 2500.0,
            }
        ],
        "net_positions": [
            {
                "tradingsymbol": "TCS",
                "exchange": "NSE",
                "product": "MIS",
                "quantity": -5,
                "average_price": 4000.0,
                "last_price": 4050.0,
                "pnl": -250.0,
                "unrealised": -250.0,
                "realised": 0.0,
            }
        ],
    }
    repo, auth, _, broker_service, ingestion_service, _ = _build_zerodha_stack(scenario=scenario)
    tok = _step_no_io(
        register_user(UserRegisterRequest(email="pos@example.com", password="Password123!"), repo=repo, auth_service=auth)
    )
    user = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

    c_resp = Response()
    auth_req = _step_no_io(connect_zerodha(response=c_resp, payload=None, current_user=user, service=broker_service))
    cookie_val, _ = _parse_oauth_cookie_header(c_resp)
    conn = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
            response=Response(),
            request_token="req-pos-1",
            state=_extract_state_from_auth_url(auth_req.authorization_url),
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    pf = _step_no_io(
        ingestion_service.create_portfolio(
            user_id=user.user_id,
            name="Positions Portfolio",
            broker_name="zerodha",
            broker_connection_id=conn.connection_id,
        )
    )
    snap = _step_no_io(
        ingestion_service.refresh_portfolio(
            user_id=user.user_id,
            portfolio_id=pf.portfolio_id,
            idempotency_key="pos-snap-1",
        )
    )
    assert len(snap.positions) == 1
    assert snap.positions[0].symbol == "TCS"
    assert snap.positions[0].quantity == Decimal("-5")

    api_resp = _to_snapshot_response(snap)
    assert api_resp.positions_count == 1
    assert api_resp.positions[0].quantity == -5.0

    ctx = _step_no_io(
        ingestion_service.build_intelligence_context(
            user_id=user.user_id,
            portfolio_id=pf.portfolio_id,
        )
    )
    assert POSITIONS_EXCLUSION_LIMITATION in ctx.limitations
    holding_symbols = [h.symbol for h in ctx.domain_portfolio.holdings]
    assert holding_symbols == ["M&M"]
    assert "TCS" not in holding_symbols


def test_refresh_rejects_broker_account_id_mismatch_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    scenario: dict[str, Any] = {
        "broker_account_id": "ZX5001",
        "holdings": [],
        "cash": 10000.0,
    }
    repo, auth, _, broker_service, ingestion_service, _ = _build_zerodha_stack(scenario=scenario)
    tok = _step_no_io(
        register_user(UserRegisterRequest(email="acct@example.com", password="Password123!"), repo=repo, auth_service=auth)
    )
    user = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

    c_resp = Response()
    auth_req = _step_no_io(connect_zerodha(response=c_resp, payload=None, current_user=user, service=broker_service))
    cookie_val, _ = _parse_oauth_cookie_header(c_resp)
    conn = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
            response=Response(),
            request_token="req-acct-1",
            state=_extract_state_from_auth_url(auth_req.authorization_url),
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    pf = _step_no_io(
        ingestion_service.create_portfolio(
            user_id=user.user_id,
            name="Mismatch Check",
            broker_name="zerodha",
            broker_connection_id=conn.connection_id,
        )
    )
    # Profile suddenly returns a different Zerodha user_id
    scenario["profile_user_id"] = "DIFF9999"
    with pytest.raises(BrokerAuthenticationError):
        _step_no_io(
            ingestion_service.refresh_portfolio(
                user_id=user.user_id,
                portfolio_id=pf.portfolio_id,
                idempotency_key="mismatch-1",
            )
        )
    snaps = _step_no_io(repo.list_portfolio_snapshots(portfolio_id=pf.portfolio_id, user_id=user.user_id))
    assert len(snaps) == 1
    assert snaps[0].status == "failed"
    assert snaps[0].holdings == []
    _, latest_completed = _step_no_io(
        ingestion_service.get_latest_portfolio_with_snapshot(
            user_id=user.user_id,
            portfolio_id=pf.portfolio_id,
        )
    )
    assert latest_completed is None


# ---------------------------------------------------------------------------
# Priority 4: Pre-HTTP idempotency & retryability after failed snapshot
# ---------------------------------------------------------------------------


def test_idempotency_precheck_skips_broker_http_and_failed_key_can_retry(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    scenario: dict[str, Any] = {
        "broker_account_id": "ZX6001",
        "cash": 15000.0,
        "holdings": [
            {
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "isin": "INE009A01021",
                "quantity": 5,
                "t1_quantity": 0,
                "average_price": 1500.0,
                "last_price": 1800.0,
            }
        ],
    }
    repo, auth, _, broker_service, ingestion_service, _ = _build_zerodha_stack(scenario=scenario)
    tok = _step_no_io(
        register_user(UserRegisterRequest(email="idem@example.com", password="Password123!"), repo=repo, auth_service=auth)
    )
    user = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

    c_resp = Response()
    auth_req = _step_no_io(connect_zerodha(response=c_resp, payload=None, current_user=user, service=broker_service))
    cookie_val, _ = _parse_oauth_cookie_header(c_resp)
    conn = _step_no_io(
        zerodha_callback(
            request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
            response=Response(),
            request_token="req-idem-1",
            state=_extract_state_from_auth_url(auth_req.authorization_url),
            status="success",
            current_user=None,
            service=broker_service,
        )
    )
    pf = _step_no_io(
        ingestion_service.create_portfolio(
            user_id=user.user_id,
            name="Idempotency Portfolio",
            broker_name="zerodha",
            broker_connection_id=conn.connection_id,
        )
    )

    calls_before_first = scenario.get("http_calls", 0)
    snap1 = _step_no_io(
        refresh_portfolio_manual(
            pf.portfolio_id,
            PortfolioRefreshRequest(idempotency_key="idem-ok-1"),
            current_user=user,
            service=ingestion_service,
        )
    )
    calls_after_first = scenario.get("http_calls", 0)
    assert calls_after_first > calls_before_first

    # Second call with same idempotency_key returns existing snapshot WITHOUT hitting broker HTTP
    snap1_dup = _step_no_io(
        refresh_portfolio_manual(
            pf.portfolio_id,
            PortfolioRefreshRequest(idempotency_key="idem-ok-1"),
            current_user=user,
            service=ingestion_service,
        )
    )
    assert snap1_dup.snapshot_id == snap1.snapshot_id
    assert scenario.get("http_calls", 0) == calls_after_first

    # Failed refresh with new key records requested_idempotency_key but frees idempotency_key for retry
    scenario["fail_holdings_error"] = True
    with pytest.raises(HTTPException):
        _step_no_io(
            refresh_portfolio_manual(
                pf.portfolio_id,
                PortfolioRefreshRequest(idempotency_key="idem-retry-1"),
                current_user=user,
                service=ingestion_service,
            )
        )

    snaps = _step_no_io(repo.list_portfolio_snapshots(portfolio_id=pf.portfolio_id, user_id=user.user_id))
    failed_snaps = [s for s in snaps if s.status == "failed"]
    assert len(failed_snaps) == 1
    assert failed_snaps[0].idempotency_key is None
    assert failed_snaps[0].metadata.get("requested_idempotency_key") == "idem-retry-1"

    # Retrying with the exact same idempotency_key after fixing upstream succeeds!
    scenario["fail_holdings_error"] = False
    recovered = _step_no_io(
        refresh_portfolio_manual(
            pf.portfolio_id,
            PortfolioRefreshRequest(idempotency_key="idem-retry-1"),
            current_user=user,
            service=ingestion_service,
        )
    )
    assert recovered.status == "completed"
    assert recovered.snapshot_id != failed_snaps[0].snapshot_id


# ---------------------------------------------------------------------------
# Priority 5 & 6: Provenance/log privacy + Zerodha resilience + PG guard
# ---------------------------------------------------------------------------


def test_provenance_and_logs_exclude_broker_account_id_and_scrub_oauth_cookies(monkeypatch):
    monkeypatch.setattr(settings, "zerodha_post_callback_redirect_uri", None)
    secret_account_id = "ZX9876PRIV"
    scenario: dict[str, Any] = {
        "broker_account_id": secret_account_id,
        "cash": 25000.0,
        "holdings": [
            {
                "tradingsymbol": "INFY",
                "exchange": "NSE",
                "isin": "INE009A01021",
                "quantity": 10,
                "t1_quantity": 0,
                "average_price": 1500.0,
                "last_price": 1800.0,
            }
        ],
    }
    repo, auth, _, broker_service, ingestion_service, _ = _build_zerodha_stack(scenario=scenario)

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(JsonFormatter())
    app_logger = logging.getLogger("app")
    app_logger.addHandler(handler)
    prev_level = app_logger.level
    app_logger.setLevel(logging.INFO)
    try:
        tok = _step_no_io(
            register_user(UserRegisterRequest(email="priv@example.com", password="Password123!"), repo=repo, auth_service=auth)
        )
        user = AuthenticatedUser(user_id=tok.user_id, email=tok.email, is_active=True)

        c_resp = Response()
        auth_req = _step_no_io(connect_zerodha(response=c_resp, payload=None, current_user=user, service=broker_service))
        cookie_val, _ = _parse_oauth_cookie_header(c_resp)
        conn = _step_no_io(
            zerodha_callback(
                request=_make_starlette_request(f"{OAUTH_BINDING_COOKIE_NAME}={cookie_val}"),
                response=Response(),
                request_token="req-priv-1",
                state=_extract_state_from_auth_url(auth_req.authorization_url),
                status="success",
                current_user=None,
                service=broker_service,
            )
        )
        pf = _step_no_io(
            ingestion_service.create_portfolio(
                user_id=user.user_id,
                name="Privacy Portfolio",
                broker_name="zerodha",
                broker_connection_id=conn.connection_id,
            )
        )
        _step_no_io(
            ingestion_service.refresh_portfolio(
                user_id=user.user_id,
                portfolio_id=pf.portfolio_id,
                idempotency_key="priv-1",
            )
        )
        ctx = _step_no_io(
            ingestion_service.build_intelligence_context(
                user_id=user.user_id,
                portfolio_id=pf.portfolio_id,
            )
        )
        assert "broker_account_id" not in ctx.provenance
        assert secret_account_id not in json.dumps(ctx.provenance)

        # Verify downstream LangGraph workflow payload does not carry raw broker_account_id
        llm_input_payload = ctx.domain_portfolio.model_dump(exclude_none=True)
        assert secret_account_id not in json.dumps(llm_input_payload)
        wf_state = run_portfolio_workflow({"portfolio": llm_input_payload})
        assert wf_state["status"] == "completed"
        assert secret_account_id not in json.dumps(wf_state)

        logs_out = log_stream.getvalue()
        assert secret_account_id not in logs_out
        assert cookie_val not in logs_out
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(prev_level)

    # Direct OAuth cookie / state / broker_account_id redaction checks
    scrubbed = redact_sensitive_text(
        "Cookie: alphapilot_oauth_bind=secretbind123; Path=/ "
        "Set-Cookie: alphapilot_oauth_bind=secretbind456 "
        "oauth_state=signed_s_val broker_account_id=ZX9876PRIV"
    )
    assert "secretbind123" not in scrubbed
    assert "secretbind456" not in scrubbed
    assert "signed_s_val" not in scrubbed
    assert "ZX9876PRIV" not in scrubbed


def test_zerodha_adapter_resilience_retry_429_deadline_and_fail_fast_errors():
    recorded_sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)

    # 1. Transient 502 recovers on retry with recorded exponential delay
    attempts_502 = {"n": 0}

    def handler_502(req: httpx.Request) -> httpx.Response:
        attempts_502["n"] += 1
        if attempts_502["n"] == 1:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={"status": "success", "data": []})

    adapter_502 = ZerodhaBrokerAdapter(
        api_key="k",
        api_secret="s",
        transport=httpx.MockTransport(handler_502),
        max_retries=2,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=2.0,
        sleep_fn=fake_sleep,
        rng=__import__("random").Random(42).uniform,
    )
    res_502 = _step_no_io(adapter_502.get_holdings_batch("tok"))
    assert res_502.items == []
    assert attempts_502["n"] == 2
    assert len(recorded_sleeps) == 1

    # 2. Bounded 429 Retry-After honored and succeeds; excessive Retry-After fails immediately
    recorded_sleeps.clear()
    attempts_429 = {"n": 0}

    def handler_429_ok(req: httpx.Request) -> httpx.Response:
        attempts_429["n"] += 1
        if attempts_429["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "1"},
                json={"status": "error", "error_type": "TooManyRequests", "message": "slow down"},
            )
        return httpx.Response(200, json={"status": "success", "data": []})

    adapter_429_ok = ZerodhaBrokerAdapter(
        api_key="k",
        api_secret="s",
        transport=httpx.MockTransport(handler_429_ok),
        max_retries=1,
        initial_backoff_seconds=0.05,
        max_backoff_seconds=2.0,
        sleep_fn=fake_sleep,
    )
    _step_no_io(adapter_429_ok.get_holdings_batch("tok"))
    assert attempts_429["n"] == 2
    assert recorded_sleeps == [1.0]

    recorded_sleeps.clear()
    attempts_429_big = {"n": 0}

    def handler_429_big(req: httpx.Request) -> httpx.Response:
        attempts_429_big["n"] += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "120"},
            json={"status": "error", "error_type": "TooManyRequests", "message": "too long"},
        )

    adapter_429_big = ZerodhaBrokerAdapter(
        api_key="k",
        api_secret="s",
        transport=httpx.MockTransport(handler_429_big),
        max_retries=2,
        initial_backoff_seconds=0.05,
        max_backoff_seconds=2.0,
        sleep_fn=fake_sleep,
    )
    with pytest.raises(BrokerRateLimitError):
        _step_no_io(adapter_429_big.get_holdings_batch("tok"))
    assert attempts_429_big["n"] == 1
    assert recorded_sleeps == []

    # 3. Non-retryable 401/403/400 fail on attempt 1 without retries
    for code, etype in ((401, "TokenException"), (403, "PermissionException"), (400, "InputException")):
        count = {"n": 0}

        def handler_fail(req: httpx.Request, c=code, et=etype, cnt=count) -> httpx.Response:
            cnt["n"] += 1
            return httpx.Response(c, json={"status": "error", "error_type": et, "message": "nope"})

        adapter_fail = ZerodhaBrokerAdapter(
            api_key="k",
            api_secret="s",
            transport=httpx.MockTransport(handler_fail),
            max_retries=3,
            initial_backoff_seconds=0.05,
            max_backoff_seconds=2.0,
            sleep_fn=fake_sleep,
        )
        with pytest.raises(Exception):
            _step_no_io(adapter_fail.get_holdings_batch("tok"))
        assert count["n"] == 1

    # 4. Expired WorkflowDeadline raises BrokerTimeoutError
    clock = [100.0]
    deadline = WorkflowDeadline(timeout_seconds=1.0, time_fn=lambda: clock[0])
    clock[0] = 105.0
    adapter_dl = ZerodhaBrokerAdapter(
        api_key="k",
        api_secret="s",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "success", "data": []})),
        max_retries=1,
    )
    with workflow_deadline_scope(deadline):
        with pytest.raises(BrokerTimeoutError):
            _step_no_io(adapter_dl.get_holdings_batch("tok", deadline=deadline))


def test_pg_test_isolation_guard_enforces_opt_in_and_blocks_working_db(monkeypatch):
    monkeypatch.delenv("ALPHAPILOT_RUN_PG_TESTS", raising=False)
    monkeypatch.delenv("ALPHAPILOT_RUN_PGVECTOR_TESTS", raising=False)
    monkeypatch.delenv("ALPHAPILOT_TEST_DATABASE_URL", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_isolated_pg_test_url()

    monkeypatch.setenv("ALPHAPILOT_RUN_PG_TESTS", "1")
    with pytest.raises(pytest.skip.Exception):
        require_isolated_pg_test_url()

    # Refuses working 'alphapilot' database by name
    monkeypatch.setenv(
        "ALPHAPILOT_TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:pw@localhost:5432/alphapilot",
    )
    with pytest.raises(RuntimeError, match="alphapilot"):
        require_isolated_pg_test_url()

    # Refuses matching runtime DATABASE_URL even with different db name
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:pw@localhost:5432/other_live")
    monkeypatch.setenv(
        "ALPHAPILOT_TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:pw@localhost:5432/other_live",
    )
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        require_isolated_pg_test_url()

    # Accepts clean dedicated test DB URL
    isolated = "postgresql+asyncpg://postgres:pw@localhost:5432/alphapilot_test"
    monkeypatch.setenv("ALPHAPILOT_TEST_DATABASE_URL", isolated)
    assert require_isolated_pg_test_url() == isolated
