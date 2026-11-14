import pytest

from app.config import settings
from app.db.session import reset_default_database_manager
from app.services.rag_service import reset_default_rag_service
from app.services.resilience import reset_default_llm_limiter


@pytest.fixture(autouse=True)
def isolate_non_integration_backend(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep non-integration unit tests independent of developer shell/env DB backend overrides."""
    reset_default_rag_service()
    reset_default_llm_limiter()
    reset_default_database_manager()

    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr(settings, "rag_storage_backend", "in_memory")

    yield

    reset_default_rag_service()
    reset_default_llm_limiter()
    reset_default_database_manager()

