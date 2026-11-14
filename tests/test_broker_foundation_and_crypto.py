"""Unit tests for Day 16–17 broker foundation, redaction, AuthService, and TokenEncryptionService."""

import base64
from datetime import datetime, timedelta, timezone
import logging
import os

import pytest

from app.auth.service import AuthConfigurationError, AuthError, AuthService, hash_oauth_token
from app.db.broker_repository import InMemoryBrokerPersistenceRepository, OAuthStateRejectedError
from app.logging_config import JsonFormatter, redact_sensitive_text
from app.integrations.brokers.base import BrokerAdapter, BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerConfigurationError,
    BrokerError,
    BrokerTokenExpiredError,
    BrokerUnsupportedOperationError,
)
from app.integrations.brokers.zerodha import ZerodhaBrokerAdapter
from app.integrations.security.encryption import (
    TokenEncryptionError,
    TokenEncryptionService,
    parse_encryption_key_ring,
)


def _b64_key(seed_byte: int) -> str:
    return base64.urlsafe_b64encode(bytes([seed_byte]) * 32).decode("ascii")


def test_broker_adapter_exposes_read_only_surface_without_trading_methods():
    forbidden = ("place_order", "modify_order", "cancel_order", "buy", "sell", "transfer_funds")
    adapter_attrs = set(dir(BrokerAdapter)) | set(dir(ZerodhaBrokerAdapter))
    for attr in forbidden:
        assert attr not in adapter_attrs


def test_broker_registry_resolves_and_rejects_unknown_brokers():
    registry = BrokerRegistry()
    adapter = ZerodhaBrokerAdapter(api_key="kite_key_test", api_secret="kite_secret_test")
    registry.register(adapter)

    assert registry.list_brokers() == ["zerodha"]
    assert registry.get("ZERODHA") is adapter

    with pytest.raises(BrokerUnsupportedOperationError):
        registry.get("upstox")


def test_broker_error_and_json_logger_redact_kite_secrets():
    err = BrokerError(
        "Failed request_token=secretreq123 Authorization: token kite123:tok999 api_secret=topsecret",
        broker_name="zerodha",
    )
    assert "secretreq123" not in str(err)
    assert "tok999" not in str(err)
    assert "topsecret" not in str(err)

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="broker_test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=45,
        msg="Token failure Authorization: token mykey:mytok456 request_token=abc999",
        args=(),
        exc_info=None,
    )
    record.checksum = "deadbeefchecksum"
    record.encrypted_access_token = "enc:v1:v1:nonce:cipher"
    formatted = formatter.format(record)
    assert "mytok456" not in formatted
    assert "abc999" not in formatted
    assert "deadbeefchecksum" not in formatted
    assert "enc:v1:v1:nonce:cipher" not in formatted

    cleaned = redact_sensitive_text(
        "https://callback.local/?status=success&request_token=one_time_tok&state=signed_state"
    )
    assert "one_time_tok" not in cleaned
    assert "signed_state" not in cleaned


def test_parse_encryption_key_ring_formats_and_validation():
    k1 = _b64_key(1)
    k2 = _b64_key(2)
    parsed_csv = parse_encryption_key_ring(f"v1:{k1}, v2:{k2}")
    assert set(parsed_csv.keys()) == {"v1", "v2"}
    assert len(parsed_csv["v1"]) == 32

    parsed_json = parse_encryption_key_ring(f'{{"v1": "{k1}"}}')
    assert set(parsed_json.keys()) == {"v1"}

    short_key = base64.urlsafe_b64encode(b"too-short").decode("ascii")
    with pytest.raises(BrokerConfigurationError):
        parse_encryption_key_ring(f"v1:{short_key}")

    with pytest.raises(BrokerConfigurationError):
        TokenEncryptionService(keys={"v1": os.urandom(32), "v2": os.urandom(32)})


def test_token_encryption_envelope_metadata_and_fail_closed_guards():
    service = TokenEncryptionService(
        keys={"v1": bytes([1]) * 32, "v2": bytes([2]) * 32},
        active_key_id="v2",
    )
    assert service.active_key_id == "v2"
    assert service.extract_key_id("enc:v1:v1:nonce:ct") == "v1"
    assert service.needs_rotation("enc:v1:v1:nonce:ct") is True
    assert service.needs_rotation("enc:v1:v2:nonce:ct") is False

    with pytest.raises(TokenEncryptionError):
        service.encrypt("", user_id="u1", broker_name="zerodha")
    with pytest.raises(TokenEncryptionError):
        service.decrypt("invalid-envelope", user_id="u1", broker_name="zerodha")
    with pytest.raises(TokenEncryptionError):
        service.decrypt("enc:v1:unknown_kid:aaaa:bbbb", user_id="u1", broker_name="zerodha")


def test_token_encryption_aes_gcm_roundtrip_aad_and_rotation():
    pytest.importorskip("cryptography")

    key_v1 = bytes([11]) * 32
    key_v2 = bytes([22]) * 32
    service_v1 = TokenEncryptionService(keys={"v1": key_v1}, active_key_id="v1")

    ciphertext_v1, kid = service_v1.encrypt(
        "kite-access-secret-token",
        user_id="user-alice",
        broker_name="zerodha",
    )
    assert kid == "v1"
    assert "kite-access-secret-token" not in ciphertext_v1
    assert service_v1.decrypt(ciphertext_v1, user_id="user-alice", broker_name="zerodha") == (
        "kite-access-secret-token"
    )

    # AAD binding stops cross-user or cross-broker ciphertext reuse even with valid key
    with pytest.raises(TokenEncryptionError):
        service_v1.decrypt(ciphertext_v1, user_id="user-bob", broker_name="zerodha")
    with pytest.raises(TokenEncryptionError):
        service_v1.decrypt(ciphertext_v1, user_id="user-alice", broker_name="upstox")

    # Tampering causes GCM authentication tag failure
    tampered = ciphertext_v1[:-2] + ("AA" if ciphertext_v1[-2:] != "AA" else "BB")
    with pytest.raises(TokenEncryptionError):
        service_v1.decrypt(tampered, user_id="user-alice", broker_name="zerodha")

    # Key rotation: v2 active service can still decrypt v1 and rotate cleanly
    service_v2 = TokenEncryptionService(keys={"v1": key_v1, "v2": key_v2}, active_key_id="v2")
    assert service_v2.needs_rotation(ciphertext_v1) is True
    assert service_v2.decrypt(ciphertext_v1, user_id="user-alice", broker_name="zerodha") == (
        "kite-access-secret-token"
    )
    ciphertext_v2, new_kid = service_v2.rotate(
        ciphertext_v1,
        user_id="user-alice",
        broker_name="zerodha",
    )
    assert new_kid == "v2"
    assert service_v2.needs_rotation(ciphertext_v2) is False
    assert service_v2.decrypt(ciphertext_v2, user_id="user-alice", broker_name="zerodha") == (
        "kite-access-secret-token"
    )


def test_auth_service_passwords_jwt_and_oauth_csrf_state():
    with pytest.raises(AuthConfigurationError):
        AuthService("short")

    auth = AuthService(secret_key="test-super-secret-signing-key-32-bytes")
    pw_hash = auth.hash_password("StrongPass123!")
    assert "StrongPass123!" not in pw_hash
    assert auth.verify_password("StrongPass123!", pw_hash) is True
    assert auth.verify_password("WrongPass123!", pw_hash) is False

    token, expires_at = auth.issue_access_token(user_id="u-1", email="alice@example.com")
    assert expires_at > datetime.now(timezone.utc)
    principal = auth.verify_access_token(token)
    assert principal.user_id == "u-1"
    assert principal.email == "alice@example.com"

    # Tampered signature rejected
    tampered_token = token[:-3] + ("xyz" if token[-3:] != "xyz" else "abc")
    with pytest.raises(AuthError):
        auth.verify_access_token(tampered_token)

    # Expired token rejected
    expired_token, _ = auth.issue_access_token(
        user_id="u-1",
        email="alice@example.com",
        ttl_seconds=-5,
    )
    with pytest.raises(AuthError, match="expired"):
        auth.verify_access_token(expired_token)

    # OAuth CSRF state binds user, broker, and enforces single-use replay rejection via persistent storage
    state, claims, state_exp = auth.create_oauth_state_with_claims(user_id="u-1", broker_name="zerodha")
    with pytest.raises(AuthError, match="user binding"):
        auth.verify_oauth_state_token(
            state,
            expected_user_id="u-2",
            expected_broker_name="zerodha",
        )

    repo = InMemoryBrokerPersistenceRepository()

    def _step(c):
        try:
            c.send(None)
        except StopIteration as stop:
            return stop.value
        raise AssertionError("Unexpected suspension")

    _step(repo.create_user(email="u1@example.com", hashed_password="h1", user_id="u-1"))
    binding_hash = hash_oauth_token("binding-secret-xyz")
    _step(
        repo.create_oauth_state_record(
            nonce_hash=hash_oauth_token(claims.nonce),
            user_id="u-1",
            broker_name="zerodha",
            client_binding_hash=binding_hash,
            expires_at=state_exp,
        )
    )

    # Separate AuthService worker (same shared signing secret) verifies cryptographic signature and consumes persistent record once
    worker_b_auth = AuthService(secret_key="test-super-secret-signing-key-32-bytes")
    verified_claims = worker_b_auth.verify_oauth_state_token(
        state,
        expected_user_id="u-1",
        expected_broker_name="zerodha",
    )
    assert verified_claims.user_id == "u-1"
    assert verified_claims.broker_name == "zerodha"
    consumed = _step(
        repo.consume_oauth_state_record(
            nonce_hash=hash_oauth_token(verified_claims.nonce),
            broker_name="zerodha",
            expected_user_id="u-1",
            client_binding_hash=binding_hash,
        )
    )
    assert consumed.user_id == "u-1"

    # Replay across a third worker is rejected cleanly via persistent state record
    worker_c_auth = AuthService(secret_key="test-super-secret-signing-key-32-bytes")
    replay_claims = worker_c_auth.verify_oauth_state_token(
        state,
        expected_user_id="u-1",
        expected_broker_name="zerodha",
    )
    with pytest.raises(OAuthStateRejectedError, match="replay rejected"):
        _step(
            repo.consume_oauth_state_record(
                nonce_hash=hash_oauth_token(replay_claims.nonce),
                broker_name="zerodha",
                expected_user_id="u-1",
                client_binding_hash=binding_hash,
            )
        )


def test_zerodha_authorization_url_format_and_missing_credentials():
    unconfigured = ZerodhaBrokerAdapter()
    with pytest.raises(BrokerConfigurationError):
        unconfigured.build_authorization_url("state-123")

    adapter = ZerodhaBrokerAdapter(api_key="kite_pub_key", api_secret="kite_sec_key")
    url = adapter.build_authorization_url("signed_state_token")
    assert url.startswith("https://kite.zerodha.com/connect/login?")
    assert "v=3" in url
    assert "api_key=kite_pub_key" in url
    assert "redirect_params=state%3Dsigned_state_token" in url
    assert "kite_sec_key" not in url
