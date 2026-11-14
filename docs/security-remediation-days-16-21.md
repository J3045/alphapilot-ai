# AlphaPilot AI — Days 16–21 Security & Architecture Remediation Notes

## 1. JWT Authentication & User Account Lifecycle

- [`verify_access_token`](../app/auth/service.py) rejects non-integer and boolean `exp`, `iat`, and `nbf` claims (`isinstance(val, bool)` is explicitly rejected before `int` checks), validates HS256 signature and `token_use == "access"`, and deliberately returns `AuthenticatedUser(..., is_active=False)`.
- [`get_current_user`](../app/auth/dependencies.py) resolves the `BrokerPersistenceRepository` and loads the authoritative `UserEntity` by `user_id` on every request:
  - Deleted / missing user -> `401 Unauthorized`.
  - Deactivated user (`is_active=False`) -> `403 Forbidden`.
  - JWT claims alone never establish active account status.

---

## 2. Browser-Compatible Persistent OAuth Flow & Replay Prevention

Process-local `_consumed_state_nonces` has been removed from [`AuthService`](../app/auth/service.py). OAuth state nonces are stored as SHA-256 digests in the persistent `broker_oauth_states` table with a browser flow-binding hash.

```text
User Browser / SPA             AlphaPilot Backend (/brokers/zerodha/*)               PostgreSQL                Zerodha Kite Login
      |                                         |                                         |                            |
      |-- POST /brokers/zerodha/connect ------->|                                         |                            |
      |   (Authorization: Bearer <jwt>)         |-- verify active user + issue random     |                            |
      |                                         |   binding secret & signed state JWT     |                            |
      |                                         |-- INSERT broker_oauth_states ---------->|                            |
      |                                         |   (nonce_hash, client_binding_hash,     |                            |
      |                                         |    user_id, expires_at, consumed=NULL)  |                            |
      |<-- 200 OK {authorization_url} ----------|                                         |                            |
      |    Set-Cookie: alphapilot_oauth_bind=.. |                                         |                            |
      |    (HttpOnly; SameSite=Lax; Path=/brokers/zerodha/callback)                       |                            |
      |                                         |                                         |                            |
      |-- User visits authorization_url ------------------------------------------------------------------------------>|
      |<-- 302 Redirect /brokers/zerodha/callback?request_token=...&state=...&status=success --------------------------|
      |                                         |                                         |                            |
      |-- GET /brokers/zerodha/callback ------->|                                         |                            |
      |   Cookie: alphapilot_oauth_bind=...     |-- Verify state JWT + SHA256 hashes      |                            |
      |   (Optional Bearer fallback supported)  |-- SELECT ... FOR UPDATE + check owner   |                            |
      |                                         |   active + match binding + set          |                            |
      |                                         |   consumed_at = NOW() ----------------->|                            |
      |                                         |-- POST /session/token (checksum) ----------------------------------->|
      |                                         |-- AES-256-GCM encrypt token under       |                            |
      |                                         |   resolved user_id + broker_name ------>|                            |
      |<-- 200 BrokerConnectionSummary JSON ----|                                         |                            |
      |    (or 302 -> ZERODHA_POST_CALLBACK_REDIRECT_URI if configured)                   |                            |
      |    Delete-Cookie: alphapilot_oauth_bind |                                         |                            |
```

Security invariants:
- `GET /brokers/zerodha/callback` works natively on browser redirects (where `Authorization` headers are absent) using the `HttpOnly`, `SameSite=Lax` cookie `alphapilot_oauth_bind`, while still permitting explicit API `Authorization: Bearer` headers (and failing closed with `401`/`403` if an invalid Bearer token is sent).
- User identity on callback is derived exclusively from the verified, single-use database state record bound to the caller's cookie or Bearer proof — never from query parameters.
- Cross-worker / multi-instance replay attempts hit the row lock (`SELECT ... FOR UPDATE`) and fail with `401 Unauthorized` once `consumed_at` is set.
- `build_authorization_url` rejects mismatched `redirect_uri` overrides against configured `ZERODHA_REDIRECT_URI`.

---

## 3. AES-256-GCM Key Ring Validation & Automatic Re-Encryption on Read

- [`TokenEncryptionService`](../app/integrations/security/encryption.py) rejects key IDs containing `":"`, blank IDs, or IDs longer than 64 characters both during key ring parsing and direct construction.
- Base64 decoding rejects whitespace, non-alphabet characters, invalid padding, non-12-byte nonces, and `< 16`-byte AES-GCM ciphertexts.
- During [`PortfolioIngestionService.refresh_portfolio`](../app/services/portfolio_ingestion_service.py), if a stored token decrypts cleanly under a retired `kid` (`needs_rotation(...)`), it is immediately re-encrypted under the active key ID and updated via compare-and-set (`rotate_broker_connection_token` checks `encrypted_access_token == expected_encrypted_token` and `status == 'active'`) so concurrent rotations or revocations cannot overwrite newer credentials.

---

## 4. Record-Level Normalization, Extended Ticker Validation & ISIN Concentration Aggregation

- [`ZerodhaBrokerAdapter`](../app/integrations/brokers/zerodha.py) exposes typed `get_holdings_batch()` / `get_positions_batch()` returning `BrokerHoldingsBatch` / `BrokerPositionsBatch` so one malformed holding row no longer aborts healthy rows before `partial_policy` evaluates (`'mark_partial'` preserves valid rows and marks `status='partial'`; `'fail'` rejects the snapshot).
- [`Holding`](../app/models/portfolio.py) supports ticker symbols up to 32 characters (`SETTINGS.max_symbol_length` default `32`) including `&` (`M&M`, `M&MFIN`, `J&KBANK`), `-`, `.`, `_` (`MCDOWELL-N`, `NIFTY100BEES`, `SGBDEC2512-GB`) while rejecting spaces, control chars, XSS (`<SCRIPT>`), and SQL syntax (`DROP TABLE`).
- [`NormalizedPortfolioResult.to_domain_portfolio`](../app/integrations/brokers/models.py) and [`assess_portfolio_risk`](../app/services/risk_engine.py) aggregate holdings across exchanges when they share a verified canonical 12-character ISIN (`instrument_id="isin:<ISIN>"`) using quantity-weighted market prices. If multiple rows share a symbol across exchanges without verifiable matching ISINs, rows are **not** merged (`unverified:<EXCHANGE>:<ID>`) and an explicit limitation warning is emitted.
- Before persisting a manual refresh, [`PortfolioIngestionService.refresh_portfolio`](../app/services/portfolio_ingestion_service.py) verifies `profile.broker_account_id == conn.broker_account_id` (when known) and fails cleanly if mismatched.

---

## 5. Open Positions Persistence & Analytics Boundary

- Open/intraday positions from `GET /portfolio/positions` (`net` book, supporting signed positive/negative quantities) are persisted in the dedicated `portfolio_positions` table and returned in [`PortfolioSnapshotResponse.positions`](../app/api/schemas.py).
- Open/intraday/derivative positions are **excluded** from long-only equity demat concentration risk percentages calculated by `assess_portfolio_risk()`. When positions are present, an explicit limitation note (`POSITIONS_EXCLUSION_LIMITATION`) is attached to `/risk` and `/analyze` responses.

---

## 6. Snapshot Idempotency & Resilience

- `portfolio_snapshots` replaces the unconditional `(portfolio_id, idempotency_key)` constraint with the partial unique index `uq_portfolio_snapshots_portfolio_idempotency_active` (`WHERE idempotency_key IS NOT NULL AND status <> 'failed'`).
- Failed refresh audit records store caller keys inside `metadata["requested_idempotency_key"]` with `idempotency_key = NULL`, preventing transient errors from poisoning retries with the same `Idempotency-Key`.
- [`PortfolioIngestionService.refresh_portfolio`](../app/services/portfolio_ingestion_service.py) checks active idempotency records before hitting Zerodha endpoints, coordinates per-portfolio concurrency, and [`PostgresBrokerPersistenceRepository.create_portfolio_snapshot`](../app/db/broker_repository.py) locks the parent `UserPortfolioRecord` row (`FOR UPDATE`).
- [`ZerodhaBrokerAdapter._request`](../app/integrations/brokers/zerodha.py) delegates transient `429`/timeout/`5xx` retries to [`execute_with_retry_async`](../app/services/resilience.py) with `WorkflowDeadline`, exponential backoff + jitter, bounded `Retry-After` handling, and zero retries on `401`/`403`/`400`/validation failures.

---

## 7. Financial Data Inventory Sent to External LLMs & Log Redaction

When `POST /portfolios/{portfolio_id}/analyze` invokes [`run_llm_portfolio_workflow_async`](../app/agents/portfolio_graph.py), **only** the following fields are included in prompt messages sent to the configured LLM provider:
1. User natural-language `query`.
2. Allowlisted `snapshot_provenance` (`portfolio_id`, `portfolio_name`, `broker_name`, `snapshot_id`, `fetched_at`, `currency`, `valuation_available`, `holdings_count`, `snapshot_available`). **Removed**: `broker_account_id` is excluded from `/risk`, `/analyze` provenance, and structured logs.
3. Normalized equity `Portfolio` JSON (`portfolio_id`, `cash`, and equity `holdings`: `symbol`, `quantity`, current `price`, optional `instrument_id`).
4. Deterministic `risk_assessment` summary (`total_value`, `cash_pct`, per-holding `weight_pct`, `passed`, `violations`).
5. Non-secret `limitations` strings.

**Never sent to the LLM:** brokerage API keys/secrets, OAuth state tokens, `alphapilot_oauth_bind` cookies, decrypted or encrypted broker access tokens, raw `broker_account_id`, user email/password hashes, cost-basis (`average_purchase_price`), raw broker P&L, or open/intraday positions. [`logging_config.py`](../app/logging_config.py) additionally redacts `oauth_state`, `Cookie:`/`Set-Cookie:` headers, `alphapilot_oauth_bind`, and `broker_account_id` keys/values.

---

## 8. Alembic Migration `0003` Review Plan & Isolated PostgreSQL Testing

### Reviewed Migration Plan (`0003_sec_oauth_pos_idemp`)
Revision [`0003_security_oauth_positions_and_idempotency.py`](../alembic/versions/0003_security_oauth_positions_and_idempotency.py) extends `0002_add_broker_portfolio_snapshots`:
1. Creates `broker_oauth_states` table + indexes (`nonce_hash` unique, `(user_id, broker_name, expires_at)`, `(expires_at, consumed_at)`).
2. Creates `portfolio_positions` table + unique constraint `(snapshot_id, symbol, exchange, instrument_identifier, product)` and symbol index.
3. Non-destructively backfills any legacy `status='failed'` rows in `portfolio_snapshots` by moving `idempotency_key` into `metadata['requested_idempotency_key']` and nulling `idempotency_key`.
4. Drops `uq_portfolio_snapshots_portfolio_idempotency` and creates partial unique index `uq_portfolio_snapshots_portfolio_idempotency_active` (`WHERE idempotency_key IS NOT NULL AND status <> 'failed'`).

*(Not applied automatically to the working `alphapilot` database; awaiting explicit user review and execution outside the sandbox.)*

### Isolated PostgreSQL Test Execution
Live PostgreSQL integration tests require **both**:
- `ALPHAPILOT_RUN_PG_TESTS=1`
- `ALPHAPILOT_TEST_DATABASE_URL` pointing to a **dedicated** test database distinct from working `DATABASE_URL` (`alphapilot`), e.g.:
  ```powershell
  $env:ALPHAPILOT_RUN_PG_TESTS="1"
  $env:ALPHAPILOT_TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/alphapilot_test"
  .venv\Scripts\python.exe -m pytest -v tests/test_pgvector_integration.py tests/test_portfolio_snapshots_db.py
  ```
Tests only delete records created with their own test-generated IDs/prefixes and never run table-wide `TRUNCATE` or unfiltered `DELETE`.

