"""Cryptographic multi-user authentication, PBKDF2 password hashing, and OAuth CSRF state."""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from typing import Any

from app.auth.models import AuthenticatedUser, OAuthStateClaims
from app.config import settings


def hash_oauth_token(raw_value: str) -> str:
    """Compute SHA-256 hex digest for persisting OAuth nonces or browser binding cookies without raw storage."""
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def generate_oauth_binding_secret() -> str:
    """Generate a high-entropy cryptographic random secret for HttpOnly OAuth flow binding cookies."""
    return secrets.token_urlsafe(32)


class AuthError(Exception):
    """Raised when authentication credentials, bearer JWTs, or CSRF state are invalid."""


class AuthConfigurationError(Exception):
    """Raised when required authentication signing secrets are not configured."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


class AuthService:
    """Signs/verifies HS256 bearer access tokens, PBKDF2 passwords, and OAuth CSRF state."""

    PBKDF2_ITERATIONS = 210_000

    def __init__(
        self,
        secret_key: str,
        *,
        token_ttl_seconds: int | None = None,
        oauth_state_ttl_seconds: int | None = None,
    ) -> None:
        cleaned = secret_key.strip()
        if len(cleaned) < 16:
            raise AuthConfigurationError(
                "AUTH_SECRET_KEY must be at least 16 characters long."
            )
        self._secret_bytes = cleaned.encode("utf-8")
        self.token_ttl_seconds = (
            token_ttl_seconds if token_ttl_seconds is not None else settings.auth_token_ttl_seconds
        )
        self.oauth_state_ttl_seconds = (
            oauth_state_ttl_seconds
            if oauth_state_ttl_seconds is not None
            else settings.broker_oauth_state_ttl_seconds
        )

    @classmethod
    def from_settings(cls) -> "AuthService":
        """Build AuthService strictly from environment settings (no insecure defaults)."""
        if settings.auth_secret_key is None or not settings.auth_secret_key.get_secret_value().strip():
            raise AuthConfigurationError("AUTH_SECRET_KEY is not configured in the environment.")
        return cls(settings.auth_secret_key.get_secret_value())

    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hash a plaintext password using PBKDF2-HMAC-SHA256 and a random 16-byte salt."""
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            cls.PBKDF2_ITERATIONS,
        )
        return f"pbkdf2_sha256${cls.PBKDF2_ITERATIONS}${_b64url_encode(salt)}${_b64url_encode(derived)}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify plaintext password against stored PBKDF2-HMAC-SHA256 hash in constant time."""
        try:
            algorithm, iter_str, salt_b64, digest_b64 = stored_hash.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            iterations = int(iter_str)
            if iterations < 10_000:
                return False
            salt = _b64url_decode(salt_b64)
            expected = _b64url_decode(digest_b64)
        except Exception:
            return False

        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)

    def _sign_hs256(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        sig = hmac.new(self._secret_bytes, signing_input, hashlib.sha256).digest()
        return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"

    def _verify_hs256(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or token.count(".") != 2:
            raise AuthError("Malformed authentication token.")
        header_b64, payload_b64, sig_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(self._secret_bytes, signing_input, hashlib.sha256).digest()
        try:
            provided_sig = _b64url_decode(sig_b64)
            header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
            payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise AuthError("Corrupted authentication token payload.") from exc

        if not isinstance(header, dict) or header.get("alg") != "HS256":
            raise AuthError("Unsupported token signature algorithm.")
        if not hmac.compare_digest(provided_sig, expected_sig):
            raise AuthError("Invalid token signature.")
        if not isinstance(payload, dict):
            raise AuthError("Invalid token claims payload.")

        now_ts = int(datetime.now(timezone.utc).timestamp())
        exp = payload.get("exp")
        if isinstance(exp, bool) or not isinstance(exp, int):
            raise AuthError("Authentication token expiration claim is invalid.")
        if now_ts >= exp:
            raise AuthError("Authentication token has expired.")

        if "iat" in payload:
            iat = payload.get("iat")
            if isinstance(iat, bool) or not isinstance(iat, int):
                raise AuthError("Authentication token issued-at claim is invalid.")

        if "nbf" in payload:
            nbf = payload.get("nbf")
            if isinstance(nbf, bool) or not isinstance(nbf, int):
                raise AuthError("Authentication token not-before claim is invalid.")
            if now_ts < nbf:
                raise AuthError("Authentication token is not yet valid.")
        return payload

    def issue_access_token(
        self,
        *,
        user_id: str,
        email: str,
        ttl_seconds: int | None = None,
    ) -> tuple[str, datetime]:
        """Issue an HS256-signed AlphaPilot bearer token for the authenticated user."""
        ttl = ttl_seconds if ttl_seconds is not None else self.token_ttl_seconds
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl)
        payload = {
            "sub": user_id,
            "email": email,
            "token_use": "access",
            "jti": secrets.token_urlsafe(12),
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        return self._sign_hs256(payload), expires_at

    def verify_access_token(self, token: str) -> AuthenticatedUser:
        """Validate bearer JWT signature/expiry and return parsed identity (active status checked from DB)."""
        payload = self._verify_hs256(token)
        if payload.get("token_use") != "access":
            raise AuthError("Token is not an AlphaPilot access token.")
        raw_sub = payload.get("sub")
        raw_email = payload.get("email")
        if not isinstance(raw_sub, str) or not isinstance(raw_email, str):
            raise AuthError("Access token is missing required user subject claims.")
        user_id = raw_sub.strip()
        email = raw_email.strip()
        if not user_id or not email:
            raise AuthError("Access token is missing required user subject claims.")
        # Deliberately never trust `is_active` from JWT claims; database lookup in `get_current_user` is authoritative.
        return AuthenticatedUser(user_id=user_id, email=email, is_active=False)

    def create_oauth_state_with_claims(
        self,
        *,
        user_id: str,
        broker_name: str,
        ttl_seconds: int | None = None,
    ) -> tuple[str, OAuthStateClaims, datetime]:
        """Create signed CSRF state token and return `(token, claims, expires_at)` for persistent storage."""
        ttl = ttl_seconds if ttl_seconds is not None else self.oauth_state_ttl_seconds
        now = datetime.now(timezone.utc)
        issued_ts = int(now.timestamp())
        expires_ts = int((now + timedelta(seconds=ttl)).timestamp())
        expires_at = datetime.fromtimestamp(expires_ts, tz=timezone.utc)
        nonce = secrets.token_urlsafe(16)
        clean_broker = broker_name.strip().lower()
        payload = {
            "sub": user_id,
            "broker": clean_broker,
            "nonce": nonce,
            "token_use": "broker_oauth_state",
            "iat": issued_ts,
            "exp": expires_ts,
        }
        claims = OAuthStateClaims(
            user_id=user_id,
            broker_name=clean_broker,
            nonce=nonce,
            issued_at=datetime.fromtimestamp(issued_ts, tz=timezone.utc),
            expires_at=expires_at,
        )
        return self._sign_hs256(payload), claims, expires_at

    def create_oauth_state(
        self,
        *,
        user_id: str,
        broker_name: str,
        ttl_seconds: int | None = None,
    ) -> tuple[str, datetime]:
        """Create a signed CSRF state token bound to user_id, broker_name, and single-use nonce."""
        token, _, expires_at = self.create_oauth_state_with_claims(
            user_id=user_id,
            broker_name=broker_name,
            ttl_seconds=ttl_seconds,
        )
        return token, expires_at

    def verify_oauth_state_token(
        self,
        state_token: str,
        *,
        expected_broker_name: str,
        expected_user_id: str | None = None,
    ) -> OAuthStateClaims:
        """Statelessly verify signed CSRF state claims; single-use consumption is enforced in persistent DB storage."""
        payload = self._verify_hs256(state_token)
        if payload.get("token_use") != "broker_oauth_state":
            raise AuthError("Invalid OAuth state token purpose.")

        raw_sub = payload.get("sub")
        raw_broker = payload.get("broker")
        raw_nonce = payload.get("nonce")
        iat = payload.get("iat")
        exp = payload.get("exp")

        if (
            not isinstance(raw_sub, str)
            or not isinstance(raw_broker, str)
            or not isinstance(raw_nonce, str)
            or isinstance(iat, bool)
            or not isinstance(iat, int)
            or isinstance(exp, bool)
            or not isinstance(exp, int)
        ):
            raise AuthError("OAuth state token is missing required claims.")

        user_id = raw_sub.strip()
        broker_name = raw_broker.strip().lower()
        nonce = raw_nonce.strip()
        if not user_id or not broker_name or not nonce:
            raise AuthError("OAuth state token is missing required claims.")

        if expected_user_id is not None and not hmac.compare_digest(user_id, expected_user_id.strip()):
            raise AuthError("OAuth state user binding mismatch.")
        if not hmac.compare_digest(broker_name, expected_broker_name.strip().lower()):
            raise AuthError("OAuth state broker binding mismatch.")

        return OAuthStateClaims(
            user_id=user_id,
            broker_name=broker_name,
            nonce=nonce,
            issued_at=datetime.fromtimestamp(iat, tz=timezone.utc),
            expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
        )

