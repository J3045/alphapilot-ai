# ADR 009 — Broker Integration, Credential Security, and Immutable Portfolio Snapshot Architecture

- **Status:** Accepted
- **Date:** 2026-11-12
- **Related Milestones:** Days 16–21 (Broker Integration & Portfolio Ingestion)

## Context

AlphaPilot AI previously analyzed portfolios passed inline via request payloads (`POST /portfolio/assess` and `POST /agent/run`). To support connecting live brokerage accounts—starting with **Zerodha Kite Connect v3**—and feeding real portfolio state into deterministic risk checks and LangGraph workflows, AlphaPilot needs:

1. A broker-agnostic adapter contract capable of onboarding additional brokers (such as Upstox and Angel One) without rewriting persistence or risk logic.
2. Strict read-only scope (no trading, order placement, or money movement).
3. Secure credential lifecycle management using authenticated encryption at rest.
4. Manual-only refresh semantics (no polling daemons, cron syncs, or background queues).
5. Immutable historical portfolio snapshots stored in the existing PostgreSQL + pgvector database.
6. Genuine multi-user ownership isolation across FastAPI endpoints and downstream workflows.

## Decisions

### 1. Read-Only Async Broker Abstraction & Registry
- Define `BrokerAdapter` (`app/integrations/brokers/base.py`) exposing only read and session lifecycle methods: `build_authorization_url`, `exchange_request_token`, `revoke_session`, `get_profile`, `get_holdings`, `get_positions`, and `get_balance`.
- No order placement, modification, or execution methods exist in the interface.
- Resolve broker implementations via `BrokerRegistry` so future brokers (Upstox OAuth2 refresh tokens, Angel One SmartAPI) plug into the same ingestion and snapshot pipelines.

### 2. Zerodha Kite Connect v3 Official Protocol & Daily Token Lifecycle
- Login redirects users to `https://kite.zerodha.com/connect/login?v=3&api_key=...&redirect_params=...` where `redirect_params` wraps a short-lived HMAC-SHA256 signed CSRF `state` bound to the authenticated AlphaPilot `user_id` and single-use nonce.
- Callback exchanges the one-time `request_token` via `POST https://api.kite.trade/session/token` using `checksum = SHA-256(api_key + request_token + api_secret)` and `X-Kite-Version: 3`.
- Because Kite Connect access tokens invalidate daily (~07:30–07:35 AM IST) and Kite Connect provides **no refresh token**, AlphaPilot explicitly models `expired` connections (`BrokerTokenExpiredError` on HTTP 403 `TokenException`) and instructs users to reauthorize via `/brokers/zerodha/connect` rather than fabricating synthetic refresh flows.

### 3. Authenticated Credential Encryption & Key Rotation
- Encrypt broker access tokens at rest in `broker_connections.encrypted_access_token` using **AES-256-GCM** (`TokenEncryptionService`) with a random 96-bit nonce per encryption, authenticated associated data (`user_id:broker_name`) binding ciphertexts to their owner/broker, and an explicit key ID (`kid`) supporting non-disruptive key rotation via `BROKER_TOKEN_ENCRYPTION_KEYS`.
- Never log plaintext tokens, encrypted payloads, `request_token`, `checksum`, or `api_secret`.
- Revocation executes provider `DELETE /session/token` best-effort and irreversibly clears encrypted credential material locally (`status='revoked'`).

### 4. Separation of Purchase Cost vs. Market Value & Honest Limitations
- Normalized holdings record `average_purchase_price` separately from current market `last_price` and annotate `price_source` (`broker_ltp` vs `unavailable`).
- If any holding lacks a valid positive `last_price` or exceeds strict core `Portfolio`/`Holding` domain boundaries, AlphaPilot marks `valuation_available=False` and reports explicit limitations instead of silently falling back to `average_purchase_price` or dropping holdings (which would distort concentration percentages).
- Currency (`INR` for Zerodha equities) and missing sector metadata are surfaced explicitly in snapshot metadata and downstream LangGraph context.

### 5. Immutable Snapshots in Existing PostgreSQL Infrastructure & Manual Refresh Only
- Reuse AlphaPilot's existing `DatabaseManager` and Alembic migration lineage (`alembic/versions/0002_add_broker_and_portfolio_snapshots.py`) rather than introducing a second database engine or external queue (Redis/Celery/Kafka).
- Model `users`, `broker_connections`, `user_portfolios`, `portfolio_snapshots` (with `Numeric(18,4)`/`Numeric(18,6)` financial precision and `(portfolio_id, idempotency_key)` uniqueness), and `portfolio_holdings`.
- Refreshes run strictly on explicit `POST /portfolios/{portfolio_id}/refresh` calls, write atomically inside a single database transaction, and never promote `failed` or `partial` refreshes to `"completed"` status.
- Risk intelligence queries select only the latest `"completed"` snapshot belonging to the authenticated user and pass sanitized `Portfolio` + provenance into `assess_portfolio_risk` and `run_llm_portfolio_workflow_async` without placing tokens into `LLMPortfolioState`.

## Consequences

- **Positive:** Strict separation between broker wire quirks, encryption, canonical financial normalization, SQLAlchemy repositories, and risk/LLM reasoning.
- **Positive:** Future brokers reuse `user_portfolios`, `portfolio_snapshots`, `portfolio_holdings`, ownership checks, and intelligence routes without schema changes.
- **Trade-off:** Live brokerage market prices are static at snapshot `fetched_at` until the user manually triggers a new refresh, and sector concentration still requires external instrument metadata when the broker does not provide sector classification.

