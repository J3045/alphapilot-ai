import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

_STANDARD_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

_SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api_?key|secret|access_token|refresh_token|session_token|auth_token|bearer_token|request_token|authorization_code|checksum|encryption_key|encrypted_access_token|jwt|password|authorization|credential|database_url|db_url|dsn|connection_string|oauth_state|state_token|client_binding|set_cookie|cookie|alphapilot_oauth_bind|broker_account_id)",
    re.IGNORECASE,
)
_API_KEY_REGEX = re.compile(r"sk-[a-zA-Z0-9_\-]{15,}")
_BEARER_TOKEN_REGEX = re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE)
_KITE_TOKEN_REGEX = re.compile(r"(Authorization\s*:\s*token\s+|token\s+)[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+", re.IGNORECASE)
_QUERY_SECRET_REGEX = re.compile(
    r"(?<![a-zA-Z0-9_])((?:request_token|access_token|api_secret|checksum|oauth_state|state|alphapilot_oauth_bind|token)=)([^&\s'\";]+)",
    re.IGNORECASE,
)
_COOKIE_HEADER_REGEX = re.compile(r"((?:Set-)?Cookie\s*:\s*)([^\r\n]+)", re.IGNORECASE)
_DB_URL_PASSWORD_REGEX = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://[^:/\s@]+:)([^@\s]+)(@)"
)
_DSN_PASSWORD_REGEX = re.compile(r"(password\s*=\s*)([^\s'\"&;]+)", re.IGNORECASE)


def redact_sensitive_text(text: str) -> str:
    """Scrub sensitive API keys, tokens, OAuth state/cookies, database credentials, and authorization headers from text."""
    if not isinstance(text, str):
        return text
    cleaned = _API_KEY_REGEX.sub("sk-[REDACTED]", text)
    cleaned = _BEARER_TOKEN_REGEX.sub("Bearer [REDACTED]", cleaned)
    cleaned = _KITE_TOKEN_REGEX.sub(r"\1[REDACTED]", cleaned)
    cleaned = _QUERY_SECRET_REGEX.sub(r"\1[REDACTED]", cleaned)
    cleaned = _COOKIE_HEADER_REGEX.sub(r"\1[REDACTED]", cleaned)
    cleaned = _DB_URL_PASSWORD_REGEX.sub(r"\1[REDACTED]\3", cleaned)
    cleaned = _DSN_PASSWORD_REGEX.sub(r"\1[REDACTED]", cleaned)
    return cleaned


def redact_sensitive_payload(val: Any) -> Any:
    """Recursively redact sensitive keys and values from logging payloads."""
    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if _SENSITIVE_KEY_PATTERNS.search(str(k)):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = redact_sensitive_payload(v)
        return sanitized
    elif isinstance(val, (list, tuple)):
        return [redact_sensitive_payload(item) for item in val]
    elif isinstance(val, str):
        return redact_sensitive_text(val)
    return val


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter with automated secret and API key redaction."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_text(record.getMessage()),
        }

        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                if _SENSITIVE_KEY_PATTERNS.search(key):
                    payload[key] = "[REDACTED]"
                else:
                    try:
                        clean_val = redact_sensitive_payload(val)
                        json.dumps(clean_val)
                        payload[key] = clean_val
                    except (TypeError, OverflowError):
                        payload[key] = redact_sensitive_text(str(val))

        if record.exc_info:
            raw_exc = self.formatException(record.exc_info)
            payload["exception"] = redact_sensitive_text(raw_exc)

        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
