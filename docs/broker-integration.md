# AlphaPilot AI — Broker Integration, Credential Security & Portfolio Ingestion Guide (Days 16–21)

## 1. Overview & Multi-Broker Architecture

AlphaPilot AI integrates external brokerage accounts through a strict, read-only layered pipeline designed to onboard additional Indian brokers (such as **Upstox** or **Angel One SmartAPI**) without changing persistence, ownership, or risk intelligence code:

1. **Adapter Layer (`app/integrations/brokers/`):**
   - [`base.py`](../app/integrations/brokers/base.py): `BrokerAdapter` abstract interface and `BrokerRegistry`.
   - [`zerodha.py`](../app/integrations/brokers/zerodha.py): Official **Zerodha Kite Connect v3** REST client over `httpx.AsyncClient`.
   - [`exceptions.py`](../app/integrations/brokers/exceptions.py): Sanitized broker error hierarchy (`BrokerTokenExpiredError`, `BrokerRateLimitError`, `BrokerPermissionError`, `BrokerUnavailableError`, `BrokerResponseValidationError`).
2. **Security & Multi-User Layer (`app/integrations/security/` & `app/auth/`):**
   - [`encryption.py`](../app/integrations/security/encryption.py): AES-256-GCM authenticated encryption at rest with versioned key IDs, AAD (`user_id` + `broker_name`), and non-disruptive key rotation.
   - [`service.py`](../app/auth/service.py) & [`dependencies.py`](../app/auth/dependencies.py): PBKDF2-HMAC-SHA256 password hashing, HS256 bearer authentication (`get_current_user`), and single-use HMAC-signed OAuth CSRF state validation.
3. **Normalization Layer (`app/integrations/normalizers/`):**
   - [`portfolio_normalizer.py`](../app/integrations/normalizers/portfolio_normalizer.py): Converts provider wire payloads into canonical `NormalizedPortfolioResult` and core AlphaPilot `Portfolio`/`Holding` models with explicit INR currency, settled + T+1 quantities, and strict separation of `average_purchase_price` vs `last_price`.
4. **Persistence Layer (`app/db/` & `alembic/`):**
   - Reuses existing [`DatabaseManager`](../app/db/session.py) and PostgreSQL connection pool.
   - [`models.py`](../app/db/models.py) & [`0002_add_broker_and_portfolio_snapshots.py`](../alembic/versions/0002_add_broker_and_portfolio_snapshots.py): Adds `users`, `broker_connections`, `user_portfolios`, `portfolio_snapshots`, and `portfolio_holdings`.
   - [`broker_repository.py`](../app/db/broker_repository.py): Atomic transactional snapshot persistence, `(portfolio_id, idempotency_key)` deduplication, strict `user_id` filtering, and historical snapshot diffing.
5. **Service, API & Intelligence Layers:**
   - [`broker_service.py`](../app/services/broker_service.py), [`portfolio_ingestion_service.py`](../app/services/portfolio_ingestion_service.py)
   - [`auth_routes.py`](../app/api/auth_routes.py), [`broker_routes.py`](../app/api/broker_routes.py), [`portfolio_routes.py`](../app/api/portfolio_routes.py)
   - [`portfolio_graph.py`](../app/agents/portfolio_graph.py): Receives sanitized `snapshot_provenance`, validated `Portfolio`, deterministic risk results, and explicit limitations without any secrets in `LLMPortfolioState`.

---

## 2. Zerodha Kite Connect v3 Protocol & Daily Session Lifecycle

- **Consent Login URL:**
  `https://kite.zerodha.com/connect/login?v=3&api_key=<ZERODHA_API_KEY>&redirect_params=state%3D<signed_csrf_state>`
- **Callback Exchange:**
  `POST https://api.kite.trade/session/token` (`X-Kite-Version: 3`, form-encoded `api_key`, `request_token`, `checksum = SHA-256(api_key + request_token + api_secret)`).
- **Authenticated Read Endpoints:**
  - `GET /user/profile`
  - `GET /portfolio/holdings` (`quantity` + `t1_quantity`, `average_price`, `last_price`, `isin`, `instrument_token`)
  - `GET /portfolio/positions` (`net` consolidated book)
  - `GET /user/margins` (`equity.available.live_balance` with fallback to `equity.available.cash`)
  - Header: `Authorization: token {api_key}:{access_token}` and `X-Kite-Version: 3`
- **Session Revocation:**
  `DELETE https://api.kite.trade/session/token?api_key=...&access_token=...` followed by wiping local encrypted ciphertext (`status='revoked'`).
- **Daily Expiry Policy (No Synthetic Refresh Token):**
  Zerodha invalidates Kite Connect access tokens daily around `07:30–07:35 IST` and does **not** provide OAuth refresh tokens. When a token reaches `token_expires_at` or Zerodha returns HTTP `403` `TokenException`, AlphaPilot marks the connection `"expired"` and returns HTTP `401` (`reauth_required`).

---

## 3. Zerodha Developer Setup Checklist

1. Create a Kite Connect app in the Zerodha Developer Console and obtain `api_key` and `api_secret`.
2. Configure the app's **Redirect URL** to your deployed callback endpoint (e.g. `https://your-domain/brokers/zerodha/callback` or local `http://localhost:8000/brokers/zerodha/callback`).
3. Generate cryptographic keys locally (never commit secrets):
   ```powershell
   # Generate AUTH_SECRET_KEY (32+ bytes)
   .venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"

   # Generate base64 32-byte AES-256-GCM key for BROKER_TOKEN_ENCRYPTION_KEYS
   .venv\Scripts\python.exe -c "import os, base64; print('v1:' + base64.urlsafe_b64encode(os.urandom(32)).decode())"
   ```
4. Apply `.env` variables and run PostgreSQL migrations (`alembic upgrade head`).

---

## 4. Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Yes (for API persistence) | `None` | PostgreSQL async connection URL (`postgresql+asyncpg://...`). |
| `AUTH_SECRET_KEY` | Yes (for user auth) | `None` | HMAC-SHA256 signing secret for AlphaPilot JWTs & CSRF OAuth state (>=16 chars). |
| `AUTH_TOKEN_TTL_SECONDS` | No | `3600` | Bearer token expiration window (`60`–`86400` seconds). |
| `BROKER_TOKEN_ENCRYPTION_KEYS` | Yes (for token encryption) | `None` | JSON `{kid: b64_key}` or CSV `v1:<b64>,v0:<b64>` 32-byte AES-256 keys. |
| `BROKER_ACTIVE_ENCRYPTION_KEY_ID` | Conditional | `None` | Required when multiple keys are present for key rotation. |
| `BROKER_OAUTH_STATE_TTL_SECONDS` | No | `600` | Lifetime of CSRF `state` token during Zerodha redirect (`60`–`3600` s). |
| `ZERODHA_API_KEY` | Yes (for Zerodha) | `None` | Kite Connect public API key (`SecretStr`). |
| `ZERODHA_API_SECRET` | Yes (for Zerodha) | `None` | Kite Connect API secret (`SecretStr`), used strictly for `SHA-256` checksums. |
| `ZERODHA_REDIRECT_URI` | Optional | `None` | Optional documentation/reference redirect URI. |
| `ZERODHA_API_BASE_URL` | No | `https://api.kite.trade` | Zerodha REST API base URL. |
| `ZERODHA_LOGIN_BASE_URL` | No | `https://kite.zerodha.com/connect/login` | Zerodha consent login URL. |
| `ZERODHA_TIMEOUT_SECONDS` | No | `10.0` | HTTP request timeout per Zerodha call. |
| `ZERODHA_MAX_RETRIES` | No | `2` | Bounded retries for transient `5xx`/network errors (never retries `400`/`401`/`403`). |

---

## 5. FastAPI Endpoints Reference

### Authentication
- `POST /auth/register` — Register account (`email`, `password`), returns bearer token (`201`).
- `POST /auth/token` — Authenticate credentials, returns bearer token (`200` / `401`).

### Broker Connections (Requires `Authorization: Bearer <token>`)
- `POST /brokers/zerodha/connect` — Returns Zerodha Kite login URL with CSRF-bound `redirect_params`.
- `GET /brokers/zerodha/callback?request_token=...&state=...&status=success` — Validates state matches authenticated user + single-use nonce, exchanges `request_token`, encrypts `access_token`, stores active connection (`BrokerConnectionSummary`).
- `GET /brokers/connections` — Lists user-owned connections without encrypted blobs or tokens.
- `DELETE /brokers/connections/{connection_id}` — Calls Kite `DELETE /session/token` best-effort and wipes encrypted token locally (`status='revoked'`).

### Portfolios & Immutable Snapshots (Requires `Authorization: Bearer <token>`)
- `POST /portfolios` — Create logical portfolio (`name`, `broker_name`, `broker_connection_id`, `currency`).
- `GET /portfolios` — List user's portfolios.
- `POST /portfolios/{portfolio_id}/refresh` — **Manual refresh only**: fetches profile, holdings, positions, margins; normalizes; stores immutable snapshot + holdings atomically (`idempotency_key` deduplicates retries). Failed/partial refreshes never replace `"completed"` snapshots.
- `GET /portfolios/{portfolio_id}` — Returns portfolio + latest `"completed"` snapshot.
- `GET /portfolios/{portfolio_id}/snapshots` — Reverse-chronological snapshot history.
- `GET /portfolios/{portfolio_id}/snapshots/{snapshot_id}` — Specific snapshot belonging to authenticated owner.
- `GET /portfolios/{portfolio_id}/risk` — Runs deterministic `assess_portfolio_risk` over the latest completed snapshot and returns provenance + limitations.
- `POST /portfolios/{portfolio_id}/analyze` — Feeds validated snapshot + risk assessment + provenance + limitations into `run_llm_portfolio_workflow_async`.

### Backward Compatibility
Existing endpoints `GET /health`, `POST /portfolio/assess`, `POST /agent/run`, and `POST /rag/ask` remain unchanged.

---

## 6. Database Schema & Financial Precision Summary

Migrated in [`alembic/versions/0002_add_broker_and_portfolio_snapshots.py`](../alembic/versions/0002_add_broker_and_portfolio_snapshots.py):
- `users`: `user_id` PK, unique `email`, `hashed_password`, `is_active`, UTC timestamps.
- `broker_connections`: `connection_id` PK, `user_id` FK (`CASCADE`), unique `(user_id, broker_name)`, nullable `encrypted_access_token` (nullified upon revocation), `encryption_key_id`, `token_expires_at`, `status`.
- `user_portfolios`: `portfolio_id` PK, `user_id` FK (`CASCADE`), `broker_connection_id` FK (`SET NULL`), unique `(user_id, name)`.
- `portfolio_snapshots`: `snapshot_id` PK, `portfolio_id`/`user_id` FKs (`CASCADE`), unique `(portfolio_id, idempotency_key)`, `cash_balance` `NUMERIC(18,4)`, `valuation_available`, composite `(portfolio_id, status, fetched_at)` index.
- `portfolio_holdings`: `holding_id` PK, `snapshot_id` FK (`CASCADE`), unique `(snapshot_id, symbol, exchange, instrument_identifier)`, `quantity` `NUMERIC(18,6)`, `average_purchase_price` `NUMERIC(18,4)`, `last_price` `NUMERIC(18,4)` nullable.

---

## 7. Known Portfolio Data Limitations

1. **Manual Point-in-Time Freshness:** Portfolio balances and `last_price` values reflect `snapshot.fetched_at` until the user explicitly requests `/portfolios/{portfolio_id}/refresh`.
2. **Purchase Cost vs. Market Price:** When a holding lacks a valid positive `last_price` (e.g. `0` or `null`), `average_purchase_price` is stored separately and **never** silently treated as market value; deterministic concentration risk is skipped with an explicit explanation instead of distorting portfolio weights.
3. **Missing Sector Metadata:** Zerodha holdings responses do not classify equity sector/industry tags; sector exposure breakdown requires an external instrument classification registry and is explicitly flagged in snapshot limitations and LangGraph prompts.
4. **Restricted Execution Environments:** AES-256-GCM runtime encryption requires the `cryptography` package (`pip install -r requirements.txt`), and live PostgreSQL integration tests require an accessible PostgreSQL instance with `DATABASE_URL` configured.

