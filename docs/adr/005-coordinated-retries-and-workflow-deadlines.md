# ADR 005: Coordinated Retry Policies and Workflow Deadlines

## Status
Accepted

## Context
LLM providers intermittently experience transient network errors, rate limits (HTTP 429), and server blips (HTTP 500/503). Retrying failed requests is necessary for reliability. However, naive retries with fixed delays can amplify outages (thundering herd problem) and cause compounding delays that freeze client requests.

## Decision
We implement two separate but tightly coordinated resilience controls in `app.services.resilience`:
1. **Exponential Backoff with Bounded Jitter**: Retries only transient exceptions (`is_retryable_error`), doubling delay per attempt with randomized jitter to prevent synchronized retries. Terminal errors (e.g., HTTP 401 auth failure, validation error) fail immediately without retrying.
2. **Monotonic Workflow Deadline (`WorkflowDeadline`)**: Tracks the overall execution budget of the logical request using monotonic time (`time.perf_counter`). Regardless of retry attempts or agent loop iterations, once the overall deadline expires, all further provider calls and tool executions are aborted.

## Benefits
- **No Infinite Retry Storms**: Retries are capped both by count (`max_retries`) and by the overarching deadline.
- **Fail-Fast Terminal Errors**: Bad inputs or expired credentials do not waste retry attempts.
- **Clock Drift Immunity**: Monotonic clocks guarantee that NTP time synchronizations do not prematurely trigger or stall deadlines.

## Trade-offs
- Setting timeouts too conservatively can cancel legitimate slow responses during high network traffic.

## Alternatives Considered
- *Third-party retry library (Tenacity)*: Tenacity handles retries well but does not natively propagate cross-node workflow deadlines across LangGraph state transitions.

## Conditions for Revisit
Introduce distributed request tracing (e.g., OpenTelemetry span deadlines) if services are split into independent microservices over gRPC or HTTP.
