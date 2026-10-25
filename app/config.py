from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AlphaPilot AI"
    environment: str = "development"
    log_level: str = "INFO"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = 30.0
    max_agent_iterations: int = 2

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

    @model_validator(mode="after")
    def validate_backoff_bounds(self) -> "Settings":
        if self.retry_max_backoff_seconds < self.retry_initial_backoff_seconds:
            raise ValueError(
                f"retry_max_backoff_seconds ({self.retry_max_backoff_seconds}) must be greater than or equal to "
                f"retry_initial_backoff_seconds ({self.retry_initial_backoff_seconds})."
            )
        return self

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
