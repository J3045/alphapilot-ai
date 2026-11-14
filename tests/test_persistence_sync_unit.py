import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, normalize_async_database_url
from app.logging_config import JsonFormatter, redact_sensitive_payload, redact_sensitive_text
from app.models.document import TextChunk
from app.services.rag_service import RagService
from app.services.resilience import sanitize_provider_error
from app.services.retrieval_service import (
    EmbeddingDimensionMismatchError,
    FakeEmbeddingClient,
    InMemoryVectorIndex,
    RetrievalService,
    VectorStoreConnectionError,
    VectorStoreError,
    validate_chunk_vector_batch,
    validate_embedding_vector,
)


def _sample_chunk(chunk_id: str = "DOC-1_chunk_0", index: int = 0) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        document_title="Sample Policy",
        chunk_index=index,
        chunk_text="Maximum single position concentration limit is 10 percent.",
        metadata={"category": "risk_policy"},
    )


class TestDatabaseConfigurationValidation:
    @staticmethod
    def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
        for field_name in Settings.model_fields:
            monkeypatch.delenv(field_name.upper(), raising=False)
            monkeypatch.delenv(field_name, raising=False)

    def test_defaults_keep_in_memory_backend_for_offline_and_unit_tests(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._clear_settings_env(monkeypatch)
        cfg = Settings(_env_file=None)
        assert cfg.rag_storage_backend == "in_memory"
        assert cfg.database_url is None
        assert cfg.embedding_dimension == 1536
        assert cfg.db_pool_size == 5
        assert cfg.db_max_overflow == 10
        assert cfg.db_pool_timeout_seconds == 10.0

    def test_env_vars_configure_postgres_backend_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._clear_settings_env(monkeypatch)
        monkeypatch.setenv("RAG_STORAGE_BACKEND", "postgres")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql://postgres:DevSecret99@localhost:5432/alphapilot",
        )
        cfg = Settings(_env_file=None)
        assert cfg.rag_storage_backend == "postgres"
        assert cfg.database_url is not None
        assert (
            cfg.get_async_database_url()
            == "postgresql+asyncpg://postgres:DevSecret99@localhost:5432/alphapilot"
        )
        assert "DevSecret99" not in (cfg.get_redacted_database_url() or "")

    def test_postgres_backend_requires_database_url(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._clear_settings_env(monkeypatch)
        with pytest.raises(ValueError, match="database_url is required"):
            Settings(_env_file=None, rag_storage_backend="postgres", database_url=None)

        with pytest.raises(ValueError, match="database_url is required"):
            Settings(_env_file=None, rag_storage_backend="postgres", database_url="   ")

    def test_rejects_non_postgres_database_url(self, monkeypatch: pytest.MonkeyPatch):
        self._clear_settings_env(monkeypatch)
        with pytest.raises(ValueError, match="PostgreSQL connection URL"):
            Settings(
                _env_file=None,
                rag_storage_backend="postgres",
                database_url="sqlite+aiosqlite:///./local.db",
            )

    def test_normalize_async_database_url_converts_sync_schemes_cleanly(self):
        assert (
            normalize_async_database_url("postgres://u:p@localhost:5432/alphapilot")
            == "postgresql+asyncpg://u:p@localhost:5432/alphapilot"
        )
        assert (
            normalize_async_database_url("postgresql://u:p@localhost:5432/alphapilot")
            == "postgresql+asyncpg://u:p@localhost:5432/alphapilot"
        )
        assert (
            normalize_async_database_url("postgresql+asyncpg://u:p@localhost:5432/alphapilot")
            == "postgresql+asyncpg://u:p@localhost:5432/alphapilot"
        )

    def test_rejects_mismatched_known_embedding_model_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._clear_settings_env(monkeypatch)
        with pytest.raises(ValueError, match="does not match configured openai_embedding_model"):
            Settings(
                _env_file=None,
                openai_embedding_model="text-embedding-3-small",
                embedding_dimension=64,
            )

    def test_redacted_database_url_masks_password(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        self._clear_settings_env(monkeypatch)
        cfg = Settings(
            _env_file=None,
            rag_storage_backend="postgres",
            database_url="postgresql+asyncpg://pguser:UltraSecretPass123@db.internal:5432/alphapilot",
        )
        masked = cfg.get_redacted_database_url()
        assert masked is not None
        assert "UltraSecretPass123" not in masked
        assert "pguser:[REDACTED]@db.internal:5432/alphapilot" in masked


class TestCredentialRedactionAndErrorSanitization:
    def test_redact_sensitive_text_masks_postgres_urls_and_dsn_passwords(self):
        raw_url_error = (
            "asyncpg failed connecting to "
            "postgresql+asyncpg://postgres:SuperSecretDBPass!@localhost:5432/alphapilot"
        )
        cleaned = redact_sensitive_text(raw_url_error)
        assert "SuperSecretDBPass!" not in cleaned
        assert "postgresql+asyncpg://postgres:[REDACTED]@localhost:5432/alphapilot" in cleaned

        raw_dsn = "host=localhost port=5432 user=postgres password=AnotherSecretValue dbname=alphapilot"
        cleaned_dsn = redact_sensitive_text(raw_dsn)
        assert "AnotherSecretValue" not in cleaned_dsn
        assert "password=[REDACTED]" in cleaned_dsn

    def test_redact_sensitive_payload_and_json_formatter_hide_db_url_keys(self):
        payload = {
            "database_url": "postgresql+asyncpg://postgres:TopSecret@localhost:5432/alphapilot",
            "dsn": "postgres://u:pw@localhost/db",
            "nested": {"connection_string": "postgresql://u:pw@host/db"},
        }
        redacted = redact_sensitive_payload(payload)
        assert redacted["database_url"] == "[REDACTED]"
        assert redacted["dsn"] == "[REDACTED]"
        assert redacted["nested"]["connection_string"] == "[REDACTED]"

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="DB error for postgresql+asyncpg://postgres:LeakCheck99@db:5432/app",
            args=(),
            exc_info=None,
        )
        formatted = json.loads(formatter.format(record))
        assert "LeakCheck99" not in formatted["message"]
        assert "[REDACTED]" in formatted["message"]

    def test_sanitize_provider_error_never_leaks_connection_credentials(self):
        conn_exc = VectorStoreConnectionError(
            "Cannot reach postgresql+asyncpg://postgres:Secret999@dbhost:5432/alphapilot"
        )
        msg = sanitize_provider_error(conn_exc)
        assert "Secret999" not in msg
        assert "dbhost" not in msg
        assert "connection" in msg.lower()

        dim_exc = EmbeddingDimensionMismatchError("Expected 1536, got 64")
        dim_msg = sanitize_provider_error(dim_exc)
        assert "dimension" in dim_msg.lower()

        generic_store_exc = VectorStoreError("raw SQL error near table secret_hint")
        generic_msg = sanitize_provider_error(generic_store_exc)
        assert "secret_hint" not in generic_msg
        assert "Vector storage" in generic_msg


class TestVectorValidationAndFailClosedBackend:
    def test_validate_embedding_vector_enforces_dimension_and_finite_floats(self):
        assert validate_embedding_vector([1, 2.5, -0.5], expected_dimension=3) == [1.0, 2.5, -0.5]

        with pytest.raises(EmbeddingDimensionMismatchError, match="expected 4, got 2"):
            validate_embedding_vector([0.1, 0.2], expected_dimension=4)

        with pytest.raises(VectorStoreError, match="must be finite"):
            validate_embedding_vector([0.1, float("nan"), 0.3], expected_dimension=3)

        with pytest.raises(VectorStoreError, match="not a numeric value"):
            validate_embedding_vector([True, 0.2, 0.3], expected_dimension=3)

    def test_validate_chunk_vector_batch_enforces_alignment_and_uniform_dimension(self):
        c1 = _sample_chunk("DOC-1_chunk_0", 0)
        c2 = _sample_chunk("DOC-1_chunk_1", 1)

        with pytest.raises(ValueError, match="Count mismatch"):
            validate_chunk_vector_batch([c1, c2], [[0.1, 0.2]])

        with pytest.raises(EmbeddingDimensionMismatchError, match="expected 3, got 2"):
            validate_chunk_vector_batch([c1, c2], [[0.1, 0.2, 0.3], [0.4, 0.5]])

        dim, cleaned = validate_chunk_vector_batch([c1], [[1, 0, 0]], expected_dimension=3)
        assert dim == 3
        assert cleaned == [[1.0, 0.0, 0.0]]

    def test_rag_service_rejects_invalid_backend_and_sync_calls_on_postgres(self):
        with pytest.raises(ValueError, match="Unsupported RAG storage_backend"):
            RagService(storage_backend="redis_vectors")

        fake_store = MagicMock()
        fake_store.embedding_dimension = 1536
        fake_store.embedding_model = "text-embedding-3-small"

        # Sync RagService call with postgres backend must fail cleanly without falling back to memory
        rag = RagService(
            llm_client=MagicMock(),
            embedding_client=FakeEmbeddingClient(dimension=1536),
            storage_backend="postgres",
            vector_store=fake_store,
        )
        result = rag.answer_question("What is the cash reserve minimum?")
        assert result.status == "failed"
        assert "Vector storage" in (result.error or "")

    def test_rag_service_fails_fast_on_embedding_client_dimension_mismatch(self):
        fake_store = MagicMock()
        fake_store.embedding_dimension = 1536
        fake_store.embedding_model = "text-embedding-3-small"

        rag = RagService(
            llm_client=MagicMock(),
            embedding_client=FakeEmbeddingClient(dimension=64),
            storage_backend="postgres",
            vector_store=fake_store,
        )
        with pytest.raises(EmbeddingDimensionMismatchError, match="dimension \\(64\\) does not match"):
            rag._ensure_postgres_retrieval_service()

    def test_sync_retrieval_service_rejects_sync_calls_when_async_store_is_configured(self):
        fake_store = MagicMock()
        fake_store.embedding_dimension = 64
        fake_store.embedding_model = "fake-64"

        retrieval = RetrievalService(
            embedding_client=FakeEmbeddingClient(dimension=64),
            vector_store=fake_store,
        )
        with pytest.raises(VectorStoreError, match="use async"):
            retrieval.index_chunks([_sample_chunk()])
        with pytest.raises(VectorStoreError, match="use async"):
            retrieval.retrieve("concentration limit")

    def test_existing_in_memory_index_remains_fully_functional(self):
        index = InMemoryVectorIndex()
        retrieval = RetrievalService(
            embedding_client=FakeEmbeddingClient(dimension=64),
            vector_index=index,
        )
        chunk = _sample_chunk()
        assert retrieval.index_chunks([chunk]) == 1
        results = retrieval.retrieve("single position concentration limit", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id
        assert results[0].metadata == {"category": "risk_policy"}

