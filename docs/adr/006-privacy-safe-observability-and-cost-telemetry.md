# ADR 006: Privacy-Safe Observability and Cost Telemetry

## Status
Accepted

## Context
Operating LLM applications in production requires granular visibility into token consumption, operation latency, failure rates, and estimated cost. However, logging entire prompt bodies, user portfolio holdings, or provider credentials introduces severe security and compliance liabilities (PII leakage, SEC compliance violations, secret exposure in centralized log collectors).

## Decision
We implemented a dedicated telemetry and logging layer (`app.observability` and `app.logging_config`):
1. **Telemetry Event Tracking (`track_llm_operation`)**: Emits concise structured events containing operation name, model name, duration, provider-reported token usage (`input_tokens`, `output_tokens`), and deterministic cost estimates. Prompts and responses are strictly omitted from telemetry records.
2. **Missing Token Handling**: Missing or unparseable token counts remain `null`; counts are never synthesized or guessed.
3. **Structured JSON Logging with Secret Redaction (`JsonFormatter`)**: Automatically strips and masks API keys matching `sk-*`, bearer tokens, and fields named `api_key`, `secret`, `password`, `authorization`, or `credential` across all log streams.

## Benefits
- **Cost Visibility**: Exact approximate cost per operation is logged in real-time for billing governance.
- **Data Privacy**: Client portfolio data, financial holdings, and proprietary research questions are not stored in log aggregators (e.g., Datadog, CloudWatch).
- **Zero Accidental Secret Leaks**: Provider authentication keys are redacted even if an unhandled library exception includes the key in its error message.

## Trade-offs
- Debugging prompt regressions requires reproducing the issue with controlled local tests, as production prompts are not preserved in log files.

## Alternatives Considered
- *Full Request/Response Payload Logging*: Standard in basic tutorials, but completely unacceptable for enterprise financial applications.
- *OpenLLMetry / Langfuse*: Powerful but requires external database/service infrastructure.

## Conditions for Revisit
If compliance or debugging requires capturing full prompt traces, implement end-to-end encrypted trace storage with role-based access control and short data retention policies.
