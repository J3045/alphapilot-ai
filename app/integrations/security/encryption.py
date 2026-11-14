"""Authenticated AES-256-GCM token encryption at rest with key rotation support."""

import base64
import json
import os
import re
from typing import Mapping

from app.config import settings
from app.integrations.brokers.exceptions import BrokerConfigurationError, BrokerError

_STRICT_URLSAFE_B64_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
_STRICT_STD_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _decode_strict_base64(encoded: str, *, allow_standard: bool = True) -> bytes:
    """Decode base64 strictly rejecting whitespace, non-alphabet bytes, or invalid padding."""
    if not isinstance(encoded, str) or not encoded or len(encoded) % 4 != 0:
        raise ValueError("Invalid base64 length or padding.")
    if _STRICT_URLSAFE_B64_RE.match(encoded):
        normalized = encoded.replace("-", "+").replace("_", "/")
        return base64.b64decode(normalized.encode("ascii"), validate=True)
    if allow_standard and _STRICT_STD_B64_RE.match(encoded):
        return base64.b64decode(encoded.encode("ascii"), validate=True)
    raise ValueError("Invalid base64 characters.")


class TokenEncryptionError(BrokerError):
    """Raised when encrypting, decrypting, or rotating broker access tokens fails."""


def _load_aesgcm_class():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BrokerConfigurationError(
            "The 'cryptography' package is required for AES-256-GCM broker token encryption. "
            "Install dependencies via 'pip install -r requirements.txt'."
        ) from exc
    return AESGCM


def parse_encryption_key_ring(raw_value: str) -> dict[str, bytes]:
    """Parse versioned 256-bit encryption keys from JSON or comma-separated `kid:b64_key` format."""
    cleaned = raw_value.strip()
    if not cleaned:
        raise BrokerConfigurationError("BROKER_TOKEN_ENCRYPTION_KEYS must not be empty.")

    raw_mapping: Mapping[str, str]
    if cleaned.startswith("{"):
        try:
            loaded = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise BrokerConfigurationError("BROKER_TOKEN_ENCRYPTION_KEYS contains invalid JSON.") from exc
        if not isinstance(loaded, dict):
            raise BrokerConfigurationError("BROKER_TOKEN_ENCRYPTION_KEYS JSON must be an object.")
        raw_mapping = {str(k).strip(): str(v).strip() for k, v in loaded.items()}
    else:
        parsed: dict[str, str] = {}
        for entry in cleaned.split(","):
            item = entry.strip()
            if not item:
                continue
            if ":" not in item:
                raise BrokerConfigurationError(
                    "Each entry in BROKER_TOKEN_ENCRYPTION_KEYS must use '<key_id>:<base64_key>' format."
                )
            kid, b64_key = item.split(":", 1)
            kid = kid.strip()
            b64_key = b64_key.strip()
            if not kid or not b64_key:
                raise BrokerConfigurationError(
                    "Empty key ID or key material in BROKER_TOKEN_ENCRYPTION_KEYS."
                )
            parsed[kid] = b64_key
        raw_mapping = parsed

    if not raw_mapping:
        raise BrokerConfigurationError("No encryption keys found in BROKER_TOKEN_ENCRYPTION_KEYS.")

    decoded_keys: dict[str, bytes] = {}
    for kid, b64_key in raw_mapping.items():
        if not kid or ":" in kid or len(kid) > 64:
            raise BrokerConfigurationError(f"Invalid encryption key identifier '{kid}'.")
        try:
            key_bytes = _decode_strict_base64(b64_key, allow_standard=True)
        except Exception as exc:
            raise BrokerConfigurationError(
                f"Encryption key '{kid}' must be valid base64-encoded 32-byte key material."
            ) from exc
        if len(key_bytes) != 32:
            raise BrokerConfigurationError(
                f"Encryption key '{kid}' must decode to exactly 32 bytes (256 bits) for AES-256-GCM."
            )
        decoded_keys[kid] = key_bytes

    return decoded_keys


class TokenEncryptionService:
    """Encrypts and decrypts brokerage tokens using AES-256-GCM with AAD ownership binding."""

    FORMAT_PREFIX = "enc:v1"

    def __init__(self, keys: Mapping[str, bytes], active_key_id: str | None = None) -> None:
        if not keys:
            raise BrokerConfigurationError("At least one 256-bit encryption key is required.")
        self._keys: dict[str, bytes] = {}
        for kid, key_bytes in keys.items():
            if not kid or not kid.strip() or ":" in kid or len(kid) > 64:
                raise BrokerConfigurationError(f"Invalid encryption key identifier '{kid}'.")
            if len(key_bytes) != 32:
                raise BrokerConfigurationError(
                    f"Encryption key '{kid}' must be exactly 32 bytes for AES-256-GCM."
                )
            self._keys[kid] = bytes(key_bytes)

        if active_key_id is not None:
            if active_key_id not in self._keys:
                raise BrokerConfigurationError(
                    f"Configured BROKER_ACTIVE_ENCRYPTION_KEY_ID '{active_key_id}' does not exist in key ring."
                )
            self._active_key_id = active_key_id
        elif len(self._keys) == 1:
            self._active_key_id = next(iter(self._keys))
        else:
            raise BrokerConfigurationError(
                "BROKER_ACTIVE_ENCRYPTION_KEY_ID is required when multiple encryption keys are configured."
            )

    @classmethod
    def from_settings(cls) -> "TokenEncryptionService":
        """Construct encryption service from environment-backed Settings (no hardcoded fallbacks)."""
        if settings.broker_token_encryption_keys is None:
            raise BrokerConfigurationError(
                "BROKER_TOKEN_ENCRYPTION_KEYS is not configured in the environment."
            )
        keys = parse_encryption_key_ring(settings.broker_token_encryption_keys.get_secret_value())
        return cls(keys=keys, active_key_id=settings.broker_active_encryption_key_id)

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    @staticmethod
    def _build_aad(user_id: str, broker_name: str) -> bytes:
        clean_user = user_id.strip()
        clean_broker = broker_name.strip().lower()
        if not clean_user or not clean_broker:
            raise TokenEncryptionError("Both user_id and broker_name are required for AAD binding.")
        return f"user:{clean_user}|broker:{clean_broker}".encode("utf-8")

    def encrypt(
        self,
        plaintext_token: str,
        *,
        user_id: str,
        broker_name: str,
    ) -> tuple[str, str]:
        """Encrypt a broker token with AES-256-GCM and return `(ciphertext_envelope, key_id)`."""
        if not isinstance(plaintext_token, str) or not plaintext_token.strip():
            raise TokenEncryptionError("Cannot encrypt an empty broker token.")

        AESGCM = _load_aesgcm_class()
        key_bytes = self._keys[self._active_key_id]
        nonce = os.urandom(12)
        aad = self._build_aad(user_id=user_id, broker_name=broker_name)
        aesgcm = AESGCM(key_bytes)
        ciphertext = aesgcm.encrypt(nonce, plaintext_token.encode("utf-8"), aad)

        nonce_b64 = base64.urlsafe_b64encode(nonce).decode("ascii")
        ct_b64 = base64.urlsafe_b64encode(ciphertext).decode("ascii")
        envelope = f"{self.FORMAT_PREFIX}:{self._active_key_id}:{nonce_b64}:{ct_b64}"
        return envelope, self._active_key_id

    def decrypt(
        self,
        encrypted_envelope: str,
        *,
        user_id: str,
        broker_name: str,
    ) -> str:
        """Decrypt an encrypted token envelope after verifying AAD and GCM authentication tag."""
        if not isinstance(encrypted_envelope, str) or not encrypted_envelope.strip():
            raise TokenEncryptionError("Encrypted broker credential payload is missing.")

        parts = encrypted_envelope.split(":")
        if len(parts) != 5 or f"{parts[0]}:{parts[1]}" != self.FORMAT_PREFIX:
            raise TokenEncryptionError("Encrypted broker credential envelope format is invalid.")

        _, _, key_id, nonce_b64, ct_b64 = parts
        key_bytes = self._keys.get(key_id)
        if key_bytes is None:
            raise TokenEncryptionError(f"Decryption key '{key_id}' is not present in key ring.")

        try:
            nonce = _decode_strict_base64(nonce_b64, allow_standard=False)
            ciphertext = _decode_strict_base64(ct_b64, allow_standard=False)
        except Exception as exc:
            raise TokenEncryptionError("Encrypted broker credential encoding is corrupted.") from exc

        if len(nonce) != 12:
            raise TokenEncryptionError("Encrypted broker credential nonce length is invalid.")
        if len(ciphertext) < 16:
            raise TokenEncryptionError("Encrypted broker credential ciphertext length is invalid.")

        AESGCM = _load_aesgcm_class()
        aad = self._build_aad(user_id=user_id, broker_name=broker_name)
        try:
            plaintext_bytes = AESGCM(key_bytes).decrypt(nonce, ciphertext, aad)
            return plaintext_bytes.decode("utf-8")
        except Exception as exc:
            raise TokenEncryptionError(
                "Broker credential decryption or authentication tag verification failed."
            ) from exc

    def extract_key_id(self, encrypted_envelope: str) -> str:
        """Return the key ID embedded in an encrypted credential envelope."""
        parts = encrypted_envelope.split(":")
        if len(parts) != 5 or f"{parts[0]}:{parts[1]}" != self.FORMAT_PREFIX:
            raise TokenEncryptionError("Encrypted broker credential envelope format is invalid.")
        return parts[2]

    def needs_rotation(self, encrypted_envelope: str) -> bool:
        """Return True if the credential was encrypted with a retired key ID."""
        return self.extract_key_id(encrypted_envelope) != self._active_key_id

    def rotate(
        self,
        encrypted_envelope: str,
        *,
        user_id: str,
        broker_name: str,
    ) -> tuple[str, str]:
        """Decrypt with the prior key and re-encrypt using the active key ID."""
        plaintext = self.decrypt(
            encrypted_envelope,
            user_id=user_id,
            broker_name=broker_name,
        )
        return self.encrypt(plaintext, user_id=user_id, broker_name=broker_name)

