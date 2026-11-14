"""Application service orchestrating manual broker portfolio refreshes, snapshots, and intelligence context."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import logging
import threading
from typing import Any, Literal
import weakref

from app.config import settings
from app.db.broker_repository import (
    BrokerConnectionEntity,
    BrokerPersistenceRepository,
    EntityNotFoundError,
    PortfolioSnapshotEntity,
    SnapshotComparisonResult,
    UserPortfolioEntity,
    compare_portfolio_snapshots,
)
from app.integrations.brokers.base import BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConnectionRevokedError,
    BrokerError,
    BrokerTokenExpiredError,
)
from app.integrations.brokers.models import (
    BrokerPositionItem,
    NormalizedHolding,
    NormalizedPortfolioResult,
    PriceSource,
)
from app.integrations.normalizers.portfolio_normalizer import ZerodhaPortfolioNormalizer
from app.integrations.security.encryption import TokenEncryptionError, TokenEncryptionService
from app.models.portfolio import Portfolio
from app.services.resilience import (
    WorkflowDeadline,
    sanitize_provider_error,
    workflow_deadline_scope,
)
from app.services.risk_engine import assess_portfolio_risk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioIntelligenceContext:
    """Sanitized portfolio snapshot data prepared for risk analysis and LangGraph reasoning.

    Explicitly contains zero credentials, access tokens, or encrypted secrets.
    """

    portfolio: UserPortfolioEntity
    snapshot: PortfolioSnapshotEntity | None
    domain_portfolio: Portfolio | None
    risk_assessment: dict[str, Any] | None
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


def snapshot_entity_to_normalized(snapshot: PortfolioSnapshotEntity) -> NormalizedPortfolioResult:
    """Reconstruct a NormalizedPortfolioResult from a persisted snapshot entity."""
    holdings: list[NormalizedHolding] = []
    for h in snapshot.holdings:
        meta = dict(h.metadata or {})
        raw_source = str(meta.get("price_source") or ("broker_ltp" if h.last_price and h.last_price > 0 else "unavailable"))
        price_source: PriceSource = "broker_ltp" if raw_source == "broker_ltp" else "unavailable"
        item_valued = bool(
            meta.get("valuation_available", h.last_price is not None and h.last_price > Decimal("0"))
        )
        settled_qty = Decimal(str(meta.get("settled_quantity", h.quantity)))
        t1_qty = Decimal(str(meta.get("t1_quantity", "0")))
        holdings.append(
            NormalizedHolding(
                symbol=h.symbol,
                exchange=h.exchange,
                instrument_identifier=h.instrument_identifier or None,
                isin=h.isin,
                quantity=h.quantity,
                settled_quantity=settled_qty,
                t1_quantity=t1_qty,
                average_purchase_price=h.average_purchase_price,
                last_price=h.last_price,
                currency=h.currency,
                price_source=price_source,
                valuation_available=item_valued,
                metadata=meta,
            )
        )

    meta_root = dict(snapshot.metadata or {})
    limitations = [str(x) for x in meta_root.get("limitations", [])]
    malformed = [dict(x) for x in meta_root.get("malformed_records", []) if isinstance(x, dict)]

    positions = [
        BrokerPositionItem(
            symbol=p.symbol,
            exchange=p.exchange,
            instrument_identifier=p.instrument_identifier or None,
            product=p.product or None,
            quantity=p.quantity,
            average_price=p.average_price,
            last_price=p.last_price,
            pnl=p.pnl,
            unrealised=p.unrealised,
            realised=p.realised,
            currency=p.currency,
        )
        for p in (snapshot.positions or [])
    ]

    return NormalizedPortfolioResult(
        broker_name=snapshot.broker_name,  # type: ignore[arg-type]
        broker_account_id=snapshot.broker_account_id or "",
        currency=snapshot.currency,
        cash_balance=snapshot.cash_balance,
        holdings=holdings,
        positions=positions,
        status=snapshot.status,
        valuation_available=snapshot.valuation_available,
        malformed_records=malformed,
        limitations=limitations,
        fetched_at=snapshot.fetched_at,
        profile=None,
    )


class PortfolioIngestionService:
    """Manual-only portfolio refresh, immutable snapshot retrieval, and intelligence preparation."""

    def __init__(
        self,
        *,
        repository: BrokerPersistenceRepository,
        registry: BrokerRegistry,
        encryption_service: TokenEncryptionService | None = None,
        normalizer: ZerodhaPortfolioNormalizer | None = None,
    ) -> None:
        self._repo = repository
        self._registry = registry
        self._encryption = encryption_service
        self._normalizer = normalizer or ZerodhaPortfolioNormalizer()
        self._refresh_lock_guard = threading.Lock()
        self._refresh_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

    def _get_portfolio_lock(self, lock_key: str) -> asyncio.Lock | None:
        """Resolve per-event-loop coordination lock for concurrent manual refreshes."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        with self._refresh_lock_guard:
            loop_locks = self._refresh_locks.get(loop)
            if loop_locks is None:
                loop_locks = {}
                self._refresh_locks[loop] = loop_locks
            lock = loop_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                loop_locks[lock_key] = lock
            return lock

    def _get_encryption(self) -> TokenEncryptionService:
        if self._encryption is None:
            self._encryption = TokenEncryptionService.from_settings()
        return self._encryption

    async def create_portfolio(
        self,
        *,
        user_id: str,
        name: str,
        broker_name: str = "zerodha",
        broker_connection_id: str | None = None,
        currency: str = "INR",
    ) -> UserPortfolioEntity:
        """Create a user-owned logical portfolio bound optionally to a broker connection."""
        conn_id = broker_connection_id
        if conn_id is not None:
            conn = await self._repo.get_broker_connection(conn_id, user_id=user_id)
            if conn is None:
                raise EntityNotFoundError("Broker connection not found for authenticated user.")
            if conn.broker_name != broker_name.strip().lower():
                raise ValueError("Broker connection provider does not match requested broker_name.")

        return await self._repo.create_user_portfolio(
            user_id=user_id,
            name=name,
            broker_name=broker_name,
            broker_connection_id=conn_id,
            currency=currency,
        )

    async def list_portfolios(self, *, user_id: str) -> list[UserPortfolioEntity]:
        return await self._repo.list_user_portfolios(user_id=user_id)

    async def _resolve_active_connection(
        self,
        *,
        user_id: str,
        portfolio: UserPortfolioEntity,
    ) -> BrokerConnectionEntity:
        conn: BrokerConnectionEntity | None = None
        if portfolio.broker_connection_id:
            conn = await self._repo.get_broker_connection(
                portfolio.broker_connection_id,
                user_id=user_id,
            )
        if conn is None:
            conn = await self._repo.get_user_broker_connection(
                user_id=user_id,
                broker_name=portfolio.broker_name,
            )
        if conn is None:
            raise BrokerAuthenticationError(
                f"No active '{portfolio.broker_name}' broker connection exists for this account.",
                broker_name=portfolio.broker_name,
            )
        if conn.status == "revoked":
            raise BrokerConnectionRevokedError(
                "Broker connection was revoked. Reconnect your brokerage account before refreshing.",
                broker_name=conn.broker_name,
            )
        if conn.status == "expired":
            raise BrokerTokenExpiredError(
                "Broker session has expired; reauthorization is required.",
                broker_name=conn.broker_name,
            )
        now = datetime.now(timezone.utc)
        if conn.token_expires_at is not None and conn.token_expires_at <= now:
            await self._repo.mark_broker_connection_expired(conn.connection_id, user_id=user_id)
            raise BrokerTokenExpiredError(
                "Broker daily session token has expired; reauthorization is required.",
                broker_name=conn.broker_name,
            )
        if not conn.encrypted_access_token or conn.status != "active":
            raise BrokerAuthenticationError(
                "Broker connection is missing valid credentials and must be reauthorized.",
                broker_name=conn.broker_name,
            )
        return conn

    async def refresh_portfolio(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        idempotency_key: str | None = None,
        partial_policy: Literal["fail", "mark_partial"] = "mark_partial",
    ) -> PortfolioSnapshotEntity:
        """Execute an explicit manual portfolio refresh and persist an immutable snapshot."""
        portfolio = await self._repo.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")

        clean_idem = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None

        async def _run_refresh() -> PortfolioSnapshotEntity:
            if clean_idem:
                existing = await self._repo.get_snapshot_by_idempotency_key(
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    idempotency_key=clean_idem,
                )
                if existing is not None:
                    return existing

            conn = await self._resolve_active_connection(user_id=user_id, portfolio=portfolio)
            adapter = self._registry.get(portfolio.broker_name)
            encryption = self._get_encryption()

            try:
                assert conn.encrypted_access_token is not None
                access_token = encryption.decrypt(
                    conn.encrypted_access_token,
                    user_id=user_id,
                    broker_name=conn.broker_name,
                )
                if callable(getattr(encryption, "needs_rotation", None)) and encryption.needs_rotation(
                    conn.encrypted_access_token
                ):
                    rotated_blob, rotated_kid = encryption.encrypt(
                        access_token,
                        user_id=user_id,
                        broker_name=conn.broker_name,
                    )
                    await self._repo.rotate_broker_connection_token(
                        conn.connection_id,
                        user_id=user_id,
                        expected_encrypted_token=conn.encrypted_access_token,
                        new_encrypted_token=rotated_blob,
                        new_encryption_key_id=rotated_kid,
                    )
            except TokenEncryptionError as exc:
                await self._repo.record_failed_refresh_snapshot(
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    broker_name=portfolio.broker_name,
                    broker_account_id=conn.broker_account_id,
                    broker_connection_id=conn.connection_id,
                    error_detail="Stored broker credential could not be decrypted.",
                    idempotency_key=clean_idem,
                )
                raise BrokerAuthenticationError(
                    "Stored broker credential could not be verified; please reconnect your broker.",
                    broker_name=conn.broker_name,
                ) from exc

            deadline = WorkflowDeadline(timeout_seconds=settings.zerodha_workflow_deadline_seconds)
            try:
                with workflow_deadline_scope(deadline):
                    profile = await adapter.get_profile(access_token)
                    if (
                        conn.broker_account_id
                        and profile.broker_account_id != conn.broker_account_id
                    ):
                        raise BrokerAuthenticationError(
                            "Authenticated broker profile account does not match connected brokerage account.",
                            broker_name=conn.broker_name,
                        )
                    holdings_batch = await adapter.get_holdings_batch(access_token, deadline=deadline)
                    positions_batch = await adapter.get_positions_batch(access_token, deadline=deadline)
                    balance = await adapter.get_balance(access_token)
                    normalized = self._normalizer.normalize(
                        profile=profile,
                        raw_holdings=holdings_batch.items,
                        balance=balance,
                        raw_positions=positions_batch.items,
                        pre_malformed_holdings=holdings_batch.malformed_records,
                        pre_malformed_positions=positions_batch.malformed_records,
                        partial_policy=partial_policy,
                    )
            except BrokerTokenExpiredError as exc:
                await self._repo.mark_broker_connection_expired(conn.connection_id, user_id=user_id)
                await self._repo.record_failed_refresh_snapshot(
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    broker_name=portfolio.broker_name,
                    broker_account_id=conn.broker_account_id,
                    broker_connection_id=conn.connection_id,
                    error_detail=sanitize_provider_error(exc),
                    idempotency_key=clean_idem,
                )
                raise
            except BrokerError as exc:
                await self._repo.record_failed_refresh_snapshot(
                    user_id=user_id,
                    portfolio_id=portfolio_id,
                    broker_name=portfolio.broker_name,
                    broker_account_id=conn.broker_account_id,
                    broker_connection_id=conn.connection_id,
                    error_detail=sanitize_provider_error(exc),
                    idempotency_key=clean_idem,
                )
                raise

            snapshot = await self._repo.create_portfolio_snapshot(
                user_id=user_id,
                portfolio_id=portfolio_id,
                normalized=normalized,
                broker_connection_id=conn.connection_id,
                idempotency_key=clean_idem,
            )
            logger.info(
                "portfolio_manual_refresh_recorded",
                extra={
                    "user_id": user_id,
                    "portfolio_id": portfolio_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "status": snapshot.status,
                    "holdings_count": snapshot.holdings_count,
                    "positions_count": len(snapshot.positions),
                    "valuation_available": snapshot.valuation_available,
                },
            )
            return snapshot

        lock = self._get_portfolio_lock(f"{user_id}:{portfolio_id}")
        if lock is None:
            return await _run_refresh()
        async with lock:
            return await _run_refresh()

    async def get_latest_portfolio_with_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
    ) -> tuple[UserPortfolioEntity, PortfolioSnapshotEntity | None]:
        portfolio = await self._repo.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")
        latest = await self._repo.get_latest_completed_snapshot(portfolio_id, user_id=user_id)
        return portfolio, latest

    async def list_snapshots(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        limit: int = 50,
    ) -> list[PortfolioSnapshotEntity]:
        portfolio = await self._repo.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")
        return await self._repo.list_portfolio_snapshots(portfolio_id, user_id=user_id, limit=limit)

    async def get_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        snapshot_id: str,
    ) -> PortfolioSnapshotEntity:
        portfolio = await self._repo.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")
        snap = await self._repo.get_snapshot_by_id(portfolio_id, snapshot_id, user_id=user_id)
        if snap is None:
            raise EntityNotFoundError("Snapshot not found for requested portfolio.")
        return snap

    async def compare_snapshots(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        older_snapshot_id: str,
        newer_snapshot_id: str,
    ) -> SnapshotComparisonResult:
        older = await self.get_snapshot(
            user_id=user_id,
            portfolio_id=portfolio_id,
            snapshot_id=older_snapshot_id,
        )
        newer = await self.get_snapshot(
            user_id=user_id,
            portfolio_id=portfolio_id,
            snapshot_id=newer_snapshot_id,
        )
        return compare_portfolio_snapshots(older, newer)

    async def build_intelligence_context(
        self,
        *,
        user_id: str,
        portfolio_id: str,
    ) -> PortfolioIntelligenceContext:
        """Load latest successful snapshot, evaluate risk safely, and attach sanitized provenance."""
        portfolio, latest = await self.get_latest_portfolio_with_snapshot(
            user_id=user_id,
            portfolio_id=portfolio_id,
        )
        if latest is None:
            return PortfolioIntelligenceContext(
                portfolio=portfolio,
                snapshot=None,
                domain_portfolio=None,
                risk_assessment=None,
                provenance={
                    "portfolio_id": portfolio.portfolio_id,
                    "portfolio_name": portfolio.name,
                    "broker_name": portfolio.broker_name,
                    "currency": portfolio.currency,
                    "snapshot_available": False,
                },
                limitations=[
                    "No completed portfolio snapshot exists yet. Trigger a manual refresh first."
                ],
            )

        normalized = snapshot_entity_to_normalized(latest)
        domain_portfolio, reasons = normalized.to_domain_portfolio(portfolio.portfolio_id)
        risk_result: dict[str, Any] | None = None

        if domain_portfolio is not None:
            if domain_portfolio.total_value > 0:
                risk_result = assess_portfolio_risk(domain_portfolio)
            else:
                reasons.append(
                    "Total portfolio market value is 0; concentration percentages cannot be computed "
                    "on an unfunded empty account."
                )

        provenance = {
            "portfolio_id": portfolio.portfolio_id,
            "portfolio_name": portfolio.name,
            "broker_name": latest.broker_name,
            "snapshot_id": latest.snapshot_id,
            "fetched_at": latest.fetched_at.isoformat(),
            "currency": latest.currency,
            "valuation_available": latest.valuation_available,
            "holdings_count": latest.holdings_count,
            "snapshot_available": True,
        }
        return PortfolioIntelligenceContext(
            portfolio=portfolio,
            snapshot=latest,
            domain_portfolio=domain_portfolio,
            risk_assessment=risk_result,
            provenance=provenance,
            limitations=reasons,
        )

