"""Opt-in safety guardrails and isolated database URL resolution for live PostgreSQL tests."""

import os
from urllib.parse import urlparse

import pytest

from app.config import settings
from app.db.session import normalize_async_database_url


def _normalized_db_identity(raw_url: str) -> tuple[str | None, int | None, str]:
    parsed = urlparse(normalize_async_database_url(raw_url.strip()))
    host = (parsed.hostname or "").strip().lower() or None
    port = parsed.port
    db_name = parsed.path.lstrip("/").strip().lower()
    return host, port, db_name


def require_isolated_pg_test_url(*, allow_legacy_pgvector_flag: bool = False) -> str:
    """Require explicit opt-in AND a dedicated test database distinct from working DATABASE_URL.

    Skips when opt-in env vars or ALPHAPILOT_TEST_DATABASE_URL are not provided.
    Refuses loudly (raises RuntimeError) if ALPHAPILOT_TEST_DATABASE_URL targets the
    working 'alphapilot' database or matches configured DATABASE_URL.
    """
    enabled = os.environ.get("ALPHAPILOT_RUN_PG_TESTS") == "1"
    if not enabled and allow_legacy_pgvector_flag:
        enabled = os.environ.get("ALPHAPILOT_RUN_PGVECTOR_TESTS") == "1"

    if not enabled:
        pytest.skip(
            "Live PostgreSQL write integration tests disabled by default "
            "(set ALPHAPILOT_RUN_PG_TESTS=1 and ALPHAPILOT_TEST_DATABASE_URL to enable)."
        )

    raw_test_url = (os.environ.get("ALPHAPILOT_TEST_DATABASE_URL") or "").strip()
    if not raw_test_url:
        pytest.skip(
            "ALPHAPILOT_TEST_DATABASE_URL must be explicitly set to a dedicated test database "
            "(tests never write to working DATABASE_URL)."
        )

    test_host, test_port, test_db_name = _normalized_db_identity(raw_test_url)
    if not test_db_name or test_db_name == "alphapilot":
        raise RuntimeError(
            "Safety refusal: ALPHAPILOT_TEST_DATABASE_URL cannot target the working 'alphapilot' "
            "database; configure a dedicated test database such as 'alphapilot_test'."
        )

    working_url = (
        (os.environ.get("DATABASE_URL") or "").strip()
        or (settings.database_url.get_secret_value().strip() if settings.database_url else "")
    )
    if working_url:
        work_host, work_port, work_db_name = _normalized_db_identity(working_url)
        if (test_host, test_port, test_db_name) == (work_host, work_port, work_db_name):
            raise RuntimeError(
                "Safety refusal: ALPHAPILOT_TEST_DATABASE_URL matches working DATABASE_URL; "
                "refusing to run integration writes against working database."
            )

    for pkg in ("sqlalchemy", "asyncpg", "pgvector", "alembic"):
        try:
            __import__(pkg)
        except ImportError:
            pytest.skip(f"Required database package '{pkg}' is not installed.")

    return normalize_async_database_url(raw_test_url)

