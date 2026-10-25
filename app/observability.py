# pyrefly: ignore[missing-import]
"""LLM Observability and Cost Tracking for AlphaPilot AI.

Tracks:
- Operation name
- Model name
- Monotonic operation duration in milliseconds
- Provider-reported token usage (without inventing missing counts)
- Estimated API cost via configurable pricing registry
- Standardized outcomes: success, failure, timeout, insufficient_evidence
- Privacy-safe error categorization without sensitive payloads
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from app.config import settings

logger = logging.getLogger("app.observability")

OutcomeType = Literal["success", "failure", "timeout", "insufficient_evidence"]


# ---------------------------------------------------------------------------
# Token Usage & Pricing Models
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Provider-reported token usage counts.

    Design rationale:
    - If a token count is unavailable or non-numeric, it remains None (represented as null in JSON).
    - AlphaPilot AI never synthesizes or guesses missing counts.
    - If provider reports an explicit total_tokens that differs from input + output
      (e.g., due to reasoning tokens or internal provider accounting), the provider's
      reported total is preserved as-is.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelPricing:
    """Pricing rates in USD per 1,000,000 tokens for a specific model."""

    input_price_per_million: float
    output_price_per_million: float


class PricingRegistry:
    """Configurable registry mapping model identifiers to pricing rates.

    DISCLAIMER:
    Model provider pricing is subject to periodic changes. Estimated costs produced by this
    registry are deterministic approximations based on configured rates and reported token usage;
    they do not constitute actual provider invoices, guarantees, or billing statements.
    """

    def __init__(self, rates: dict[str, ModelPricing] | None = None) -> None:
        self._rates: dict[str, ModelPricing] = {}
        if rates:
            for model_name, pricing in rates.items():
                self.register(model_name, pricing)

    def register(self, model: str, pricing: ModelPricing) -> None:
        """Register or override pricing rates for a model identifier."""
        if not model or not model.strip():
            raise ValueError("Model name must be a non-empty string.")
        self._rates[model.strip().lower()] = pricing

    def get(self, model: str | None) -> ModelPricing | None:
        """Retrieve configured pricing for a model if explicitly configured."""
        if not model or not isinstance(model, str):
            return None
        return self._rates.get(model.strip().lower())

    def calculate_cost(
        self,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """Calculate estimated cost in USD.

        Returns None if:
        - The model is not explicitly configured in the registry.
        - Either input_tokens or output_tokens is unavailable (None).

        Formula:
        (input_tokens / 1,000,000) * input_price_per_million +
        (output_tokens / 1,000,000) * output_price_per_million
        """
        if model is None or input_tokens is None or output_tokens is None:
            return None

        pricing = self.get(model)
        if pricing is None:
            return None

        # Validate non-negative numbers
        if input_tokens < 0 or output_tokens < 0:
            return None

        input_cost = (input_tokens / 1_000_000.0) * pricing.input_price_per_million
        output_cost = (output_tokens / 1_000_000.0) * pricing.output_price_per_million
        return round(input_cost + output_cost, 8)


def create_default_pricing_registry() -> PricingRegistry:
    """Build pricing registry populated from application settings."""
    registry = PricingRegistry()
    input_rates = getattr(settings, "model_pricing_input_per_million", {})
    output_rates = getattr(settings, "model_pricing_output_per_million", {})

    all_models = set(input_rates.keys()) | set(output_rates.keys())
    for model in all_models:
        in_rate = input_rates.get(model, 0.0)
        out_rate = output_rates.get(model, 0.0)
        registry.register(model, ModelPricing(in_rate, out_rate))

    return registry


# Global shared pricing registry
_DEFAULT_PRICING_REGISTRY: PricingRegistry = create_default_pricing_registry()


def get_default_pricing_registry() -> PricingRegistry:
    return _DEFAULT_PRICING_REGISTRY


# ---------------------------------------------------------------------------
# Telemetry Event Model
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """Structured event payload for completed operations."""

    operation: str
    model: str | None
    duration_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    outcome: OutcomeType = "success"
    error_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-serializable dictionary with explicit nulls for missing values."""
        return {
            "operation": self.operation,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "estimated_cost": self.estimated_cost,
            "outcome": self.outcome,
            "error_category": self.error_category,
        }


# ---------------------------------------------------------------------------
# In-Memory Event Sink (for testing and verification)
# ---------------------------------------------------------------------------

_RECORDED_EVENTS: list[TelemetryEvent] = []


def get_recorded_events() -> list[TelemetryEvent]:
    """Retrieve in-memory copy of recorded telemetry events."""
    return list(_RECORDED_EVENTS)


def clear_recorded_events() -> None:
    """Clear recorded telemetry events."""
    _RECORDED_EVENTS.clear()


def emit_telemetry_event(event: TelemetryEvent) -> None:
    """Record event in memory and emit a structured JSON log entry."""
    _RECORDED_EVENTS.append(event)
    payload = event.to_dict()
    logger.info("llm_telemetry_event", extra={"telemetry": payload})


# ---------------------------------------------------------------------------
# Metadata Extraction & Categorization Helpers
# ---------------------------------------------------------------------------

def _parse_int_safely(val: Any) -> int | None:
    """Convert value to int if non-boolean integer, else None."""
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return None


def extract_token_usage(response: Any) -> TokenUsage:
    """Extract token usage counts from provider response metadata.

    Supports:
    - LangChain AIMessage.usage_metadata ({input_tokens, output_tokens, total_tokens})
    - LangChain AIMessage.response_metadata (token_usage / usage dicts)
    - OpenAI ChatCompletion response objects or dictionaries (usage dict/object)

    Never estimates missing counts; unavailable counts remain None.
    Preserves provider's reported total_tokens without overriding it.
    """
    if response is None:
        return TokenUsage()

    in_tok: int | None = None
    out_tok: int | None = None
    tot_tok: int | None = None

    # 1. LangChain usage_metadata (standard in langchain_core >= 0.2)
    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        in_tok = _parse_int_safely(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens"))
        out_tok = _parse_int_safely(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens"))
        tot_tok = _parse_int_safely(usage_meta.get("total_tokens"))
        if in_tok is not None or out_tok is not None or tot_tok is not None:
            return TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)

    # 2. LangChain response_metadata
    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        tu = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(tu, dict):
            in_tok = _parse_int_safely(tu.get("prompt_tokens") or tu.get("input_tokens"))
            out_tok = _parse_int_safely(tu.get("completion_tokens") or tu.get("output_tokens"))
            tot_tok = _parse_int_safely(tu.get("total_tokens"))
            if in_tok is not None or out_tok is not None or tot_tok is not None:
                return TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)

    # 3. Direct usage attribute (e.g., OpenAI API ChatCompletion object)
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        if isinstance(usage_obj, dict):
            in_tok = _parse_int_safely(usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens"))
            out_tok = _parse_int_safely(usage_obj.get("completion_tokens") or usage_obj.get("output_tokens"))
            tot_tok = _parse_int_safely(usage_obj.get("total_tokens"))
        else:
            in_tok = _parse_int_safely(getattr(usage_obj, "prompt_tokens", getattr(usage_obj, "input_tokens", None)))
            out_tok = _parse_int_safely(getattr(usage_obj, "completion_tokens", getattr(usage_obj, "output_tokens", None)))
            tot_tok = _parse_int_safely(getattr(usage_obj, "total_tokens", None))

        if in_tok is not None or out_tok is not None or tot_tok is not None:
            return TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)

    # 4. Dictionary response structure
    if isinstance(response, dict):
        tu = response.get("usage_metadata") or response.get("token_usage") or response.get("usage")
        if isinstance(tu, dict):
            in_tok = _parse_int_safely(tu.get("prompt_tokens") or tu.get("input_tokens"))
            out_tok = _parse_int_safely(tu.get("completion_tokens") or tu.get("output_tokens"))
            tot_tok = _parse_int_safely(tu.get("total_tokens"))
            return TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=tot_tok)

    return TokenUsage(input_tokens=None, output_tokens=None, total_tokens=None)


def categorize_error(exc: Exception | None) -> str | None:
    """Categorize an exception safely without exposing sensitive data, keys, or payloads."""
    if exc is None:
        return None

    if isinstance(exc, TimeoutError):
        return "timeout"

    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    if "timeout" in exc_type or "timed out" in exc_msg or "timeout" in exc_msg or "deadline" in exc_type or "deadline" in exc_msg:
        return "timeout"
    if "auth" in exc_type or "auth" in exc_msg or "unauthorized" in exc_msg or "api_key" in exc_msg or "401" in exc_msg:
        return "authentication"
    if "ratelimit" in exc_type or "rate_limit" in exc_msg or "429" in exc_msg:
        return "rate_limit"
    if "connection" in exc_type or "connect" in exc_msg or "dns" in exc_msg or "unreachable" in exc_msg:
        return "connection"
    if "context_length" in exc_type or "maximum context" in exc_msg or "context window" in exc_msg:
        return "context_length_exceeded"
    if "invalid_request" in exc_type or "bad_request" in exc_msg or "400" in exc_msg:
        return "bad_request"
    return "provider_error"


def resolve_model_name(
    client: Any = None,
    response: Any = None,
    fallback: str | None = None,
) -> str | None:
    """Resolve model name safely from response metadata or client object without MagicMock attribute traps."""
    # 1. From response metadata if present
    if response is not None:
        resp_meta = getattr(response, "response_metadata", None)
        if isinstance(resp_meta, dict):
            m = resp_meta.get("model_name") or resp_meta.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()

        resp_model = getattr(response, "model", None)
        if isinstance(resp_model, str) and resp_model.strip():
            return resp_model.strip()

    # 2. From client object
    if client is not None:
        for attr in ("model_name", "model"):
            val = getattr(client, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        bound = getattr(client, "bound", None)
        if bound is not None:
            for attr in ("model_name", "model"):
                val = getattr(bound, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # 3. Fallback
    if fallback is not None and isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return getattr(settings, "openai_model", None)


# ---------------------------------------------------------------------------
# Operation Tracker Context Manager
# ---------------------------------------------------------------------------

class TelemetryTracker:
    """Context manager for measuring monotonic duration and emitting exactly one final event."""

    def __init__(
        self,
        operation: str,
        model: str | None = None,
        pricing_registry: PricingRegistry | None = None,
    ) -> None:
        self.operation = operation
        self.model = model
        self.pricing_registry = pricing_registry or get_default_pricing_registry()
        self.start_time: float = 0.0
        self.event: TelemetryEvent | None = None
        self._completed = False

    def __enter__(self) -> "TelemetryTracker":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._completed and exc_val is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000.0
            error_cat = categorize_error(exc_val)
            outcome: OutcomeType = "timeout" if error_cat == "timeout" else "failure"

            self.event = TelemetryEvent(
                operation=self.operation,
                model=self.model,
                duration_ms=round(duration_ms, 2),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                estimated_cost=None,
                outcome=outcome,
                error_category=error_cat,
            )
            emit_telemetry_event(self.event)
            self._completed = True
        return False

    def record_success(
        self,
        response: Any,
        outcome: OutcomeType = "success",
        model: str | None = None,
    ) -> TelemetryEvent:
        """Record a successful operation outcome and emit a telemetry event."""
        if self._completed:
            return self.event  # type: ignore[return-value]

        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        resolved_model = resolve_model_name(
            client=None,
            response=response,
            fallback=model or self.model,
        )

        usage = extract_token_usage(response)
        cost = self.pricing_registry.calculate_cost(
            model=resolved_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        self.event = TelemetryEvent(
            operation=self.operation,
            model=resolved_model,
            duration_ms=round(duration_ms, 2),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=cost,
            outcome=outcome,
            error_category=None,
        )
        emit_telemetry_event(self.event)
        self._completed = True
        return self.event


def track_llm_operation(
    operation: str,
    model: str | None = None,
    pricing_registry: PricingRegistry | None = None,
) -> TelemetryTracker:
    """Create a TelemetryTracker for an LLM operation."""
    return TelemetryTracker(
        operation=operation,
        model=model,
        pricing_registry=pricing_registry,
    )


def record_insufficient_evidence_without_llm(
    operation: str = "rag_answer",
    duration_ms: float = 0.0,
) -> TelemetryEvent:
    """Record an insufficient-evidence outcome when no LLM invocation occurred.

    Explicitly keeps model=None and token counts=None to ensure no false LLM calls are reported.
    """
    event = TelemetryEvent(
        operation=operation,
        model=None,
        duration_ms=round(duration_ms, 2),
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        estimated_cost=None,
        outcome="insufficient_evidence",
        error_category=None,
    )
    emit_telemetry_event(event)
    return event
