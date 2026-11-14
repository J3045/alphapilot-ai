from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def normalize_async_database_url(raw_url: str) -> str:
    """Validate and normalize a PostgreSQL connection URL to use SQLAlchemy's asyncpg driver."""
    url = raw_url.strip()
    if not url:
        raise ValueError("database_url must not be empty.")
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgresql+asyncpg://"):
        return url
    raise ValueError(
        "database_url must be a PostgreSQL connection URL "
        "(starting with postgresql+asyncpg://, postgresql://, or postgres://)."
    )


class Settings(BaseSettings):
    app_name: str = "AlphaPilot AI"
    environment: str = "development"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = 30.0
    max_agent_iterations: int = 2

    # Security & Resource Boundary Limits
    max_query_length: int = Field(
        default=2000,
        ge=1,
        le=50000,
        description="Maximum allowed character length for user questions/queries to prevent DoS/overflow.",
    )
    max_retrieval_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum allowed top_k chunks for retrieval to bound context and compute.",
    )
    min_relevance_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity relevance score threshold for RAG evidence.",
    )
    max_holdings_count: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of holdings permitted in a single portfolio payload.",
    )
    max_portfolio_id_length: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Maximum allowed length for a portfolio ID identifier string.",
    )
    max_symbol_length: int = Field(
        default=32,
        ge=1,
        le=64,
        description="Maximum allowed length for a ticker symbol (supports long Indian ETF/SGB symbols up to 32 chars).",
    )

    # Resilience & Retry Configuration
    # Semantics: max_retries is the maximum number of retry attempts following an initial failure.
    # Total potential attempts = 1 + max_retries (e.g. max_retries=2 means at most 3 total attempts).
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts after initial failure. Total attempts = 1 + max_retries.",
    )
    retry_initial_backoff_seconds: float = Field(
        default=0.5,
        ge=0.01,
        le=60.0,
        description="Initial backoff delay in seconds for exponential retry backoff.",
    )
    retry_max_backoff_seconds: float = Field(
        default=2.0,
        ge=0.01,
        le=300.0,
        description="Maximum ceiling for exponential retry backoff delay.",
    )

    # Workflow Execution Budget
    # Overall execution deadline in seconds for an end-to-end logical workflow (RAG or portfolio agent).
    workflow_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Overall execution budget in seconds for a complete workflow run.",
    )

    # Bounded Concurrency & Backpressure Configuration
    # Controls in-flight async LLM calls per worker process to prevent unbounded queueing.
    max_concurrent_llm_requests: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Maximum number of simultaneous in-flight LLM provider calls permitted per worker process.",
    )
    llm_queue_timeout_seconds: float = Field(
        default=5.0,
        ge=0.01,
        le=300.0,
        description="Maximum wait time in seconds to acquire an LLM concurrency slot before rejecting with backpressure.",
    )
    rag_prewarm_on_startup: bool = Field(
        default=False,
        description="Whether FastAPI lifespan prewarms the process-local RAG vector index during worker startup.",
    )

    # Persistent Vector Storage (PostgreSQL + pgvector)
    rag_storage_backend: Literal["in_memory", "postgres"] = Field(
        default="in_memory",
        description="RAG vector storage backend: 'in_memory' for offline dev/tests or 'postgres' for PostgreSQL + pgvector.",
    )
    database_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL async connection URL (e.g. postgresql+asyncpg://user:pass@host:5432/alphapilot).",
    )
    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        le=8192,
        description="Embedding vector dimension for the configured embedding model (1536 for text-embedding-3-small).",
    )
    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Base SQLAlchemy async connection pool size per worker process.",
    )
    db_max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum overflow connections permitted beyond db_pool_size.",
    )
    db_pool_timeout_seconds: float = Field(
        default=10.0,
        ge=0.5,
        le=120.0,
        description="Maximum wait time in seconds to acquire a database connection from the pool.",
    )
    db_echo: bool = Field(
        default=False,
        description="Whether SQLAlchemy logs SQL statements (keep false outside local debugging).",
    )

    # Multi-User Auth & Broker Token Encryption (Days 17-20)
    auth_secret_key: SecretStr | None = Field(
        default=None,
        description="HMAC secret key for signing AlphaPilot user bearer tokens and OAuth CSRF state.",
    )
    auth_token_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Lifetime in seconds for AlphaPilot bearer access tokens.",
    )
    broker_token_encryption_keys: SecretStr | None = Field(
        default=None,
        description="Versioned AES-256-GCM key ring as JSON {kid: b64_key} or comma-separated kid:b64_key entries.",
    )
    broker_active_encryption_key_id: str | None = Field(
        default=None,
        description="Active encryption key ID used to encrypt new or rotated broker access tokens.",
    )
    broker_oauth_state_ttl_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Expiry window in seconds for signed broker OAuth CSRF state tokens.",
    )

    # Zerodha Kite Connect v3 Integration
    zerodha_api_key: SecretStr | None = Field(
        default=None,
        description="Zerodha Kite Connect API key (never hardcoded or logged).",
    )
    zerodha_api_secret: SecretStr | None = Field(
        default=None,
        description="Zerodha Kite Connect API secret used only for SHA-256 session checksum exchange.",
    )
    zerodha_redirect_uri: str | None = Field(
        default=None,
        description="Optional configured callback redirect URI for Zerodha Kite Connect verification.",
    )
    zerodha_post_callback_redirect_uri: str | None = Field(
        default=None,
        description=(
            "Optional frontend destination URI for HTTP 302 redirect after a successful "
            "Zerodha OAuth browser callback; when unset, callback returns BrokerConnectionSummary JSON."
        ),
    )
    zerodha_api_base_url: str = Field(
        default="https://api.kite.trade",
        description="Base URL for Zerodha Kite Connect v3 REST API.",
    )
    zerodha_login_base_url: str = Field(
        default="https://kite.zerodha.com/connect/login",
        description="Base URL for Zerodha Kite Connect v3 user login consent endpoint.",
    )
    zerodha_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="HTTP request timeout in seconds for Zerodha Kite Connect API calls.",
    )
    zerodha_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum bounded retries for transient Zerodha 5xx/429/network failures (never retries auth failures).",
    )
    zerodha_workflow_deadline_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Overall per-refresh workflow deadline budget across Zerodha API calls and bounded retries.",
    )
    oauth_cookie_secure: bool = Field(
        default=False,
        description="Whether to mark the HttpOnly OAuth flow binding cookie as Secure (enable in HTTPS deployments).",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.retry_max_backoff_seconds < self.retry_initial_backoff_seconds:
            raise ValueError(
                f"retry_max_backoff_seconds ({self.retry_max_backoff_seconds}) must be greater than or equal to "
                f"retry_initial_backoff_seconds ({self.retry_initial_backoff_seconds})."
            )

        expected_dim = KNOWN_EMBEDDING_DIMENSIONS.get(self.openai_embedding_model)
        if expected_dim is not None and self.embedding_dimension != expected_dim:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) does not match configured "
                f"openai_embedding_model '{self.openai_embedding_model}' (expected {expected_dim})."
            )

        if self.rag_storage_backend == "postgres":
            if self.database_url is None or not self.database_url.get_secret_value().strip():
                raise ValueError(
                    "database_url is required when rag_storage_backend='postgres'."
                )
            normalize_async_database_url(self.database_url.get_secret_value())

        return self

    def get_async_database_url(self) -> str:
        """Return normalized asyncpg PostgreSQL connection string for SQLAlchemy."""
        if self.database_url is None:
            raise ValueError("database_url is not configured.")
        return normalize_async_database_url(self.database_url.get_secret_value())

    def get_redacted_database_url(self) -> str | None:
        """Return database URL with embedded password masked for safe diagnostics."""
        if self.database_url is None:
            return None
        from app.logging_config import redact_sensitive_text

        return redact_sensitive_text(self.database_url.get_secret_value())

    # Pricing registry configuration: rates in USD per 1,000,000 tokens.
    # Note: Provider rates are subject to change. Estimated costs are approximations
    # and do not reflect actual invoices or billing guarantees.
    model_pricing_input_per_million: dict[str, float] = {
        "gpt-4o-mini": 0.15,
        "gpt-4o": 2.50,
        "text-embedding-3-small": 0.02,
    }
    model_pricing_output_per_million: dict[str, float] = {
        "gpt-4o-mini": 0.60,
        "gpt-4o": 10.00,
        "text-embedding-3-small": 0.0,
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
