"""PostgreSQL SQLAlchemy 2.0 async repository (and Protocol) for users, broker connections, and immutable snapshots."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hmac
from typing import Any, Protocol
import uuid

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.db.models import (
    BrokerConnectionRecord,
    BrokerOAuthStateRecord,
    PortfolioHoldingRecord,
    PortfolioPositionRecord,
    PortfolioSnapshotRecord,
    UserPortfolioRecord,
    UserRecord,
)
from app.db.session import DatabaseManager
from app.integrations.brokers.models import (
    BrokerConnectionStatus,
    NormalizedHolding,
    NormalizedPortfolioResult,
    SnapshotStatus,
)


class EntityNotFoundError(Exception):
    """Raised when a requested record does not exist or is not owned by the authenticated user."""


class DuplicateEntityError(Exception):
    """Raised when inserting a record violates a natural uniqueness constraint (e.g. duplicate email)."""


class OAuthStateRejectedError(Exception):
    """Raised when persistent OAuth state verification fails (missing, expired, replayed, or unbound)."""


@dataclass(frozen=True)
class UserEntity:
    user_id: str
    email: str
    hashed_password: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BrokerConnectionEntity:
    connection_id: str
    user_id: str
    broker_name: str
    broker_account_id: str | None
    encrypted_access_token: str | None
    encryption_key_id: str | None
    token_expires_at: datetime | None
    status: BrokerConnectionStatus
    last_authenticated_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class UserPortfolioEntity:
    portfolio_id: str
    user_id: str
    broker_connection_id: str | None
    name: str
    broker_name: str
    currency: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PortfolioHoldingEntity:
    holding_id: str
    snapshot_id: str
    symbol: str
    exchange: str
    instrument_identifier: str
    isin: str | None
    quantity: Decimal
    average_purchase_price: Decimal
    last_price: Decimal | None
    currency: str
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class BrokerOAuthStateEntity:
    state_id: str
    nonce_hash: str
    user_id: str
    broker_name: str
    client_binding_hash: str
    expires_at: datetime
    consumed_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class PortfolioPositionEntity:
    position_id: str
    snapshot_id: str
    symbol: str
    exchange: str
    instrument_identifier: str
    product: str
    quantity: Decimal
    average_price: Decimal
    last_price: Decimal | None
    pnl: Decimal | None
    unrealised: Decimal | None
    realised: Decimal | None
    currency: str
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class PortfolioSnapshotEntity:
    snapshot_id: str
    portfolio_id: str
    user_id: str
    broker_connection_id: str | None
    broker_name: str
    broker_account_id: str | None
    fetched_at: datetime
    status: SnapshotStatus
    idempotency_key: str | None
    cash_balance: Decimal | None
    currency: str
    valuation_available: bool
    holdings_count: int
    metadata: dict[str, Any]
    error_detail: str | None
    created_at: datetime
    holdings: list[PortfolioHoldingEntity] = field(default_factory=list)
    positions: list[PortfolioPositionEntity] = field(default_factory=list)


@dataclass(frozen=True)
class HoldingDelta:
    symbol: str
    exchange: str
    previous_quantity: Decimal | None
    current_quantity: Decimal | None
    quantity_delta: Decimal
    previous_last_price: Decimal | None
    current_last_price: Decimal | None


@dataclass(frozen=True)
class SnapshotComparisonResult:
    older_snapshot_id: str
    newer_snapshot_id: str
    older_fetched_at: datetime
    newer_fetched_at: datetime
    cash_delta: Decimal | None
    added_symbols: list[str]
    removed_symbols: list[str]
    holding_changes: list[HoldingDelta]


def compare_portfolio_snapshots(
    older: PortfolioSnapshotEntity,
    newer: PortfolioSnapshotEntity,
) -> SnapshotComparisonResult:
    """Compute a clean deterministic diff between two historical snapshots of a portfolio."""
    if older.portfolio_id != newer.portfolio_id or older.user_id != newer.user_id:
        raise ValueError("Cannot compare snapshots across different portfolios or users.")

    older_map = {
        (h.symbol, h.exchange, h.instrument_identifier): h
        for h in older.holdings
    }
    newer_map = {
        (h.symbol, h.exchange, h.instrument_identifier): h
        for h in newer.holdings
    }

    added_symbols = sorted(
        {key[0] for key in newer_map if key not in older_map}
    )
    removed_symbols = sorted(
        {key[0] for key in older_map if key not in newer_map}
    )

    holding_changes: list[HoldingDelta] = []
    for key in sorted(set(older_map.keys()) | set(newer_map.keys())):
        prev = older_map.get(key)
        curr = newer_map.get(key)
        prev_qty = prev.quantity if prev else Decimal("0")
        curr_qty = curr.quantity if curr else Decimal("0")
        prev_price = prev.last_price if prev else None
        curr_price = curr.last_price if curr else None
        if prev_qty != curr_qty or prev_price != curr_price or prev is None or curr is None:
            holding_changes.append(
                HoldingDelta(
                    symbol=key[0],
                    exchange=key[1],
                    previous_quantity=prev.quantity if prev else None,
                    current_quantity=curr.quantity if curr else None,
                    quantity_delta=curr_qty - prev_qty,
                    previous_last_price=prev_price,
                    current_last_price=curr_price,
                )
            )

    cash_delta: Decimal | None = None
    if older.cash_balance is not None and newer.cash_balance is not None:
        cash_delta = newer.cash_balance - older.cash_balance

    return SnapshotComparisonResult(
        older_snapshot_id=older.snapshot_id,
        newer_snapshot_id=newer.snapshot_id,
        older_fetched_at=older.fetched_at,
        newer_fetched_at=newer.fetched_at,
        cash_delta=cash_delta,
        added_symbols=added_symbols,
        removed_symbols=removed_symbols,
        holding_changes=holding_changes,
    )


class BrokerPersistenceRepository(Protocol):
    """Async repository interface shared between Postgres and deterministic test stores."""

    async def create_user(self, *, email: str, hashed_password: str, user_id: str | None = None) -> UserEntity: ...

    async def get_user_by_email(self, email: str) -> UserEntity | None: ...

    async def get_user_by_id(self, user_id: str) -> UserEntity | None: ...

    async def upsert_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
        broker_account_id: str | None,
        encrypted_access_token: str,
        encryption_key_id: str,
        token_expires_at: datetime | None,
        status: BrokerConnectionStatus,
        last_authenticated_at: datetime,
    ) -> BrokerConnectionEntity: ...

    async def get_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None: ...

    async def get_user_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
    ) -> BrokerConnectionEntity | None: ...

    async def list_broker_connections(self, *, user_id: str) -> list[BrokerConnectionEntity]: ...

    async def mark_broker_connection_expired(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None: ...

    async def revoke_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None: ...

    async def set_user_active(
        self,
        user_id: str,
        *,
        is_active: bool,
    ) -> UserEntity | None: ...

    async def delete_user(self, user_id: str) -> bool: ...

    async def create_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        user_id: str,
        broker_name: str,
        client_binding_hash: str,
        expires_at: datetime,
    ) -> BrokerOAuthStateEntity: ...

    async def consume_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        broker_name: str,
        expected_user_id: str | None = None,
        client_binding_hash: str | None = None,
        now: datetime | None = None,
    ) -> BrokerOAuthStateEntity: ...

    async def rotate_broker_connection_token(
        self,
        connection_id: str,
        *,
        user_id: str,
        expected_encrypted_token: str,
        new_encrypted_token: str,
        new_encryption_key_id: str,
    ) -> BrokerConnectionEntity | None: ...

    async def create_user_portfolio(
        self,
        *,
        user_id: str,
        name: str,
        broker_name: str,
        broker_connection_id: str | None = None,
        currency: str = "INR",
        portfolio_id: str | None = None,
    ) -> UserPortfolioEntity: ...

    async def get_user_portfolio(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> UserPortfolioEntity | None: ...

    async def list_user_portfolios(self, *, user_id: str) -> list[UserPortfolioEntity]: ...

    async def create_portfolio_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        normalized: NormalizedPortfolioResult,
        broker_connection_id: str | None = None,
        idempotency_key: str | None = None,
        error_detail: str | None = None,
    ) -> PortfolioSnapshotEntity: ...

    async def record_failed_refresh_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        broker_name: str,
        broker_account_id: str | None = None,
        broker_connection_id: str | None = None,
        error_detail: str,
        idempotency_key: str | None = None,
        fetched_at: datetime | None = None,
    ) -> PortfolioSnapshotEntity: ...

    async def get_snapshot_by_idempotency_key(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        idempotency_key: str,
    ) -> PortfolioSnapshotEntity | None: ...

    async def get_latest_completed_snapshot(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None: ...

    async def list_portfolio_snapshots(
        self,
        portfolio_id: str,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[PortfolioSnapshotEntity]: ...

    async def get_snapshot_by_id(
        self,
        portfolio_id: str,
        snapshot_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None: ...


def _to_user_entity(rec: UserRecord) -> UserEntity:
    return UserEntity(
        user_id=rec.user_id,
        email=rec.email,
        hashed_password=rec.hashed_password,
        is_active=rec.is_active,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


def _to_connection_entity(rec: BrokerConnectionRecord) -> BrokerConnectionEntity:
    return BrokerConnectionEntity(
        connection_id=rec.connection_id,
        user_id=rec.user_id,
        broker_name=rec.broker_name,
        broker_account_id=rec.broker_account_id,
        encrypted_access_token=rec.encrypted_access_token,
        encryption_key_id=rec.encryption_key_id,
        token_expires_at=rec.token_expires_at,
        status=rec.status,  # type: ignore[arg-type]
        last_authenticated_at=rec.last_authenticated_at,
        revoked_at=rec.revoked_at,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


def _to_portfolio_entity(rec: UserPortfolioRecord) -> UserPortfolioEntity:
    return UserPortfolioEntity(
        portfolio_id=rec.portfolio_id,
        user_id=rec.user_id,
        broker_connection_id=rec.broker_connection_id,
        name=rec.name,
        broker_name=rec.broker_name,
        currency=rec.currency,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


def _to_holding_entity(rec: PortfolioHoldingRecord) -> PortfolioHoldingEntity:
    return PortfolioHoldingEntity(
        holding_id=rec.holding_id,
        snapshot_id=rec.snapshot_id,
        symbol=rec.symbol,
        exchange=rec.exchange,
        instrument_identifier=rec.instrument_identifier,
        isin=rec.isin,
        quantity=Decimal(rec.quantity),
        average_purchase_price=Decimal(rec.average_purchase_price),
        last_price=Decimal(rec.last_price) if rec.last_price is not None else None,
        currency=rec.currency,
        metadata=dict(rec.holding_metadata or {}),
        created_at=rec.created_at,
    )


def _to_oauth_state_entity(rec: BrokerOAuthStateRecord) -> BrokerOAuthStateEntity:
    return BrokerOAuthStateEntity(
        state_id=rec.state_id,
        nonce_hash=rec.nonce_hash,
        user_id=rec.user_id,
        broker_name=rec.broker_name,
        client_binding_hash=rec.client_binding_hash,
        expires_at=rec.expires_at,
        consumed_at=rec.consumed_at,
        created_at=rec.created_at,
    )


def _to_position_entity(rec: PortfolioPositionRecord) -> PortfolioPositionEntity:
    return PortfolioPositionEntity(
        position_id=rec.position_id,
        snapshot_id=rec.snapshot_id,
        symbol=rec.symbol,
        exchange=rec.exchange,
        instrument_identifier=rec.instrument_identifier,
        product=rec.product,
        quantity=Decimal(rec.quantity),
        average_price=Decimal(rec.average_price),
        last_price=Decimal(rec.last_price) if rec.last_price is not None else None,
        pnl=Decimal(rec.pnl) if rec.pnl is not None else None,
        unrealised=Decimal(rec.unrealised) if rec.unrealised is not None else None,
        realised=Decimal(rec.realised) if rec.realised is not None else None,
        currency=rec.currency,
        metadata=dict(rec.position_metadata or {}),
        created_at=rec.created_at,
    )


def _to_snapshot_entity(rec: PortfolioSnapshotRecord) -> PortfolioSnapshotEntity:
    return PortfolioSnapshotEntity(
        snapshot_id=rec.snapshot_id,
        portfolio_id=rec.portfolio_id,
        user_id=rec.user_id,
        broker_connection_id=rec.broker_connection_id,
        broker_name=rec.broker_name,
        broker_account_id=rec.broker_account_id,
        fetched_at=rec.fetched_at,
        status=rec.status,  # type: ignore[arg-type]
        idempotency_key=rec.idempotency_key,
        cash_balance=Decimal(rec.cash_balance) if rec.cash_balance is not None else None,
        currency=rec.currency,
        valuation_available=rec.valuation_available,
        holdings_count=rec.holdings_count,
        metadata=dict(rec.snapshot_metadata or {}),
        error_detail=rec.error_detail,
        created_at=rec.created_at,
        holdings=[_to_holding_entity(h) for h in (rec.holdings or [])],
        positions=[_to_position_entity(p) for p in (rec.positions or [])],
    )


class PostgresBrokerPersistenceRepository:
    """SQLAlchemy 2.0 async persistence repository reusing AlphaPilot's DatabaseManager."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    async def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        user_id: str | None = None,
    ) -> UserEntity:
        clean_email = email.strip().lower()
        new_id = user_id or f"usr_{uuid.uuid4().hex[:24]}"
        async with self._db.session() as session:
            try:
                async with session.begin():
                    rec = UserRecord(
                        user_id=new_id,
                        email=clean_email,
                        hashed_password=hashed_password,
                        is_active=True,
                    )
                    session.add(rec)
                    await session.flush()
                    await session.refresh(rec)
                    return _to_user_entity(rec)
            except IntegrityError as exc:
                raise DuplicateEntityError("An account with that email already exists.") from exc

    async def get_user_by_email(self, email: str) -> UserEntity | None:
        clean_email = email.strip().lower()
        async with self._db.session() as session:
            result = await session.execute(select(UserRecord).where(UserRecord.email == clean_email))
            rec = result.scalar_one_or_none()
            return _to_user_entity(rec) if rec is not None else None

    async def get_user_by_id(self, user_id: str) -> UserEntity | None:
        async with self._db.session() as session:
            result = await session.execute(select(UserRecord).where(UserRecord.user_id == user_id))
            rec = result.scalar_one_or_none()
            return _to_user_entity(rec) if rec is not None else None

    async def upsert_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
        broker_account_id: str | None,
        encrypted_access_token: str,
        encryption_key_id: str,
        token_expires_at: datetime | None,
        status: BrokerConnectionStatus,
        last_authenticated_at: datetime,
    ) -> BrokerConnectionEntity:
        clean_broker = broker_name.strip().lower()
        async with self._db.session() as session:
            async with session.begin():
                result = await session.execute(
                    select(BrokerConnectionRecord).where(
                        BrokerConnectionRecord.user_id == user_id,
                        BrokerConnectionRecord.broker_name == clean_broker,
                    )
                )
                rec = result.scalar_one_or_none()
                if rec is None:
                    rec = BrokerConnectionRecord(
                        connection_id=f"conn_{uuid.uuid4().hex[:24]}",
                        user_id=user_id,
                        broker_name=clean_broker,
                        broker_account_id=broker_account_id,
                        encrypted_access_token=encrypted_access_token,
                        encryption_key_id=encryption_key_id,
                        token_expires_at=token_expires_at,
                        status=status,
                        last_authenticated_at=last_authenticated_at,
                        revoked_at=None,
                    )
                    session.add(rec)
                else:
                    rec.broker_account_id = broker_account_id
                    rec.encrypted_access_token = encrypted_access_token
                    rec.encryption_key_id = encryption_key_id
                    rec.token_expires_at = token_expires_at
                    rec.status = status
                    rec.last_authenticated_at = last_authenticated_at
                    rec.revoked_at = None
                await session.flush()
                await session.refresh(rec)
                return _to_connection_entity(rec)

    async def get_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(BrokerConnectionRecord).where(
                    BrokerConnectionRecord.connection_id == connection_id,
                    BrokerConnectionRecord.user_id == user_id,
                )
            )
            rec = result.scalar_one_or_none()
            return _to_connection_entity(rec) if rec is not None else None

    async def get_user_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
    ) -> BrokerConnectionEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(BrokerConnectionRecord).where(
                    BrokerConnectionRecord.user_id == user_id,
                    BrokerConnectionRecord.broker_name == broker_name.strip().lower(),
                )
            )
            rec = result.scalar_one_or_none()
            return _to_connection_entity(rec) if rec is not None else None

    async def list_broker_connections(self, *, user_id: str) -> list[BrokerConnectionEntity]:
        async with self._db.session() as session:
            result = await session.execute(
                select(BrokerConnectionRecord)
                .where(BrokerConnectionRecord.user_id == user_id)
                .order_by(BrokerConnectionRecord.created_at.asc())
            )
            return [_to_connection_entity(r) for r in result.scalars().all()]

    async def mark_broker_connection_expired(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        async with self._db.session() as session:
            async with session.begin():
                result = await session.execute(
                    select(BrokerConnectionRecord).where(
                        BrokerConnectionRecord.connection_id == connection_id,
                        BrokerConnectionRecord.user_id == user_id,
                    )
                )
                rec = result.scalar_one_or_none()
                if rec is None:
                    return None
                if rec.status != "revoked":
                    rec.status = "expired"
                await session.flush()
                await session.refresh(rec)
                return _to_connection_entity(rec)

    async def revoke_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        now = datetime.now(timezone.utc)
        async with self._db.session() as session:
            async with session.begin():
                result = await session.execute(
                    select(BrokerConnectionRecord).where(
                        BrokerConnectionRecord.connection_id == connection_id,
                        BrokerConnectionRecord.user_id == user_id,
                    )
                )
                rec = result.scalar_one_or_none()
                if rec is None:
                    return None
                rec.status = "revoked"
                rec.encrypted_access_token = None
                rec.encryption_key_id = None
                rec.token_expires_at = None
                rec.revoked_at = now
                await session.flush()
                await session.refresh(rec)
                return _to_connection_entity(rec)

    async def set_user_active(
        self,
        user_id: str,
        *,
        is_active: bool,
    ) -> UserEntity | None:
        async with self._db.session() as session:
            async with session.begin():
                result = await session.execute(
                    select(UserRecord).where(UserRecord.user_id == user_id)
                )
                rec = result.scalar_one_or_none()
                if rec is None:
                    return None
                rec.is_active = bool(is_active)
                await session.flush()
                await session.refresh(rec)
                return _to_user_entity(rec)

    async def delete_user(self, user_id: str) -> bool:
        async with self._db.session() as session:
            async with session.begin():
                result = await session.execute(
                    select(UserRecord).where(UserRecord.user_id == user_id)
                )
                rec = result.scalar_one_or_none()
                if rec is None:
                    return False
                await session.delete(rec)
                await session.flush()
                return True

    async def create_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        user_id: str,
        broker_name: str,
        client_binding_hash: str,
        expires_at: datetime,
    ) -> BrokerOAuthStateEntity:
        clean_broker = broker_name.strip().lower()
        async with self._db.session() as session:
            async with session.begin():
                owner = (
                    await session.execute(
                        select(UserRecord).where(UserRecord.user_id == user_id)
                    )
                ).scalar_one_or_none()
                if owner is None or not owner.is_active:
                    raise OAuthStateRejectedError("Cannot issue OAuth state for missing or inactive user.")

                rec = BrokerOAuthStateRecord(
                    state_id=f"ost_{uuid.uuid4().hex[:24]}",
                    nonce_hash=nonce_hash,
                    user_id=user_id,
                    broker_name=clean_broker,
                    client_binding_hash=client_binding_hash,
                    expires_at=expires_at,
                    consumed_at=None,
                )
                session.add(rec)
                await session.flush()
                await session.refresh(rec)
                return _to_oauth_state_entity(rec)

    async def consume_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        broker_name: str,
        expected_user_id: str | None = None,
        client_binding_hash: str | None = None,
        now: datetime | None = None,
    ) -> BrokerOAuthStateEntity:
        if expected_user_id is None and client_binding_hash is None:
            raise OAuthStateRejectedError("OAuth callback lacks caller identity or browser binding proof.")

        current_ts = now or datetime.now(timezone.utc)
        clean_broker = broker_name.strip().lower()
        async with self._db.session() as session:
            async with session.begin():
                rec = (
                    await session.execute(
                        select(BrokerOAuthStateRecord)
                        .where(BrokerOAuthStateRecord.nonce_hash == nonce_hash)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if rec is None:
                    raise OAuthStateRejectedError("OAuth callback state is unknown or invalid.")
                if rec.broker_name != clean_broker:
                    raise OAuthStateRejectedError("OAuth state broker binding mismatch.")
                if expected_user_id is not None and not hmac.compare_digest(
                    rec.user_id, expected_user_id.strip()
                ):
                    raise OAuthStateRejectedError("OAuth state user binding mismatch.")
                if client_binding_hash is not None and not hmac.compare_digest(
                    rec.client_binding_hash, client_binding_hash
                ):
                    raise OAuthStateRejectedError("OAuth state browser binding mismatch.")

                owner = (
                    await session.execute(
                        select(UserRecord).where(UserRecord.user_id == rec.user_id)
                    )
                ).scalar_one_or_none()
                if owner is None or not owner.is_active:
                    raise OAuthStateRejectedError("OAuth state owner account is inactive or deleted.")

                if rec.consumed_at is not None:
                    raise OAuthStateRejectedError(
                        "OAuth callback state nonce has already been consumed (replay rejected)."
                    )

                exp = rec.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp <= current_ts:
                    raise OAuthStateRejectedError("OAuth callback state has expired.")

                rec.consumed_at = current_ts
                await session.flush()
                await session.refresh(rec)
                return _to_oauth_state_entity(rec)

    async def rotate_broker_connection_token(
        self,
        connection_id: str,
        *,
        user_id: str,
        expected_encrypted_token: str,
        new_encrypted_token: str,
        new_encryption_key_id: str,
    ) -> BrokerConnectionEntity | None:
        async with self._db.session() as session:
            async with session.begin():
                rec = (
                    await session.execute(
                        select(BrokerConnectionRecord)
                        .where(
                            BrokerConnectionRecord.connection_id == connection_id,
                            BrokerConnectionRecord.user_id == user_id,
                            BrokerConnectionRecord.status == "active",
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if rec is None or rec.encrypted_access_token != expected_encrypted_token:
                    return None
                rec.encrypted_access_token = new_encrypted_token
                rec.encryption_key_id = new_encryption_key_id
                await session.flush()
                await session.refresh(rec)
                return _to_connection_entity(rec)

    async def create_user_portfolio(
        self,
        *,
        user_id: str,
        name: str,
        broker_name: str,
        broker_connection_id: str | None = None,
        currency: str = "INR",
        portfolio_id: str | None = None,
    ) -> UserPortfolioEntity:
        pid = portfolio_id or f"pf_{uuid.uuid4().hex[:24]}"
        async with self._db.session() as session:
            try:
                async with session.begin():
                    rec = UserPortfolioRecord(
                        portfolio_id=pid,
                        user_id=user_id,
                        broker_connection_id=broker_connection_id,
                        name=name.strip(),
                        broker_name=broker_name.strip().lower(),
                        currency=currency.strip().upper(),
                    )
                    session.add(rec)
                    await session.flush()
                    await session.refresh(rec)
                    return _to_portfolio_entity(rec)
            except IntegrityError as exc:
                raise DuplicateEntityError(
                    "A portfolio with that name already exists for this user."
                ) from exc

    async def get_user_portfolio(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> UserPortfolioEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(UserPortfolioRecord).where(
                    UserPortfolioRecord.portfolio_id == portfolio_id,
                    UserPortfolioRecord.user_id == user_id,
                )
            )
            rec = result.scalar_one_or_none()
            return _to_portfolio_entity(rec) if rec is not None else None

    async def list_user_portfolios(self, *, user_id: str) -> list[UserPortfolioEntity]:
        async with self._db.session() as session:
            result = await session.execute(
                select(UserPortfolioRecord)
                .where(UserPortfolioRecord.user_id == user_id)
                .order_by(UserPortfolioRecord.created_at.asc())
            )
            return [_to_portfolio_entity(r) for r in result.scalars().all()]

    async def _find_existing_by_idempotency(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        idempotency_key: str,
    ) -> PortfolioSnapshotEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(PortfolioSnapshotRecord)
                .options(
                    selectinload(PortfolioSnapshotRecord.holdings),
                    selectinload(PortfolioSnapshotRecord.positions),
                )
                .where(
                    PortfolioSnapshotRecord.portfolio_id == portfolio_id,
                    PortfolioSnapshotRecord.user_id == user_id,
                    PortfolioSnapshotRecord.idempotency_key == idempotency_key,
                    PortfolioSnapshotRecord.status != "failed",
                )
            )
            rec = result.scalar_one_or_none()
            return _to_snapshot_entity(rec) if rec is not None else None

    async def get_snapshot_by_idempotency_key(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        idempotency_key: str,
    ) -> PortfolioSnapshotEntity | None:
        clean_idem = idempotency_key.strip()
        if not clean_idem:
            return None
        return await self._find_existing_by_idempotency(
            user_id=user_id,
            portfolio_id=portfolio_id,
            idempotency_key=clean_idem,
        )

    async def create_portfolio_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        normalized: NormalizedPortfolioResult,
        broker_connection_id: str | None = None,
        idempotency_key: str | None = None,
        error_detail: str | None = None,
    ) -> PortfolioSnapshotEntity:
        clean_idem = idempotency_key.strip() if idempotency_key else None
        if clean_idem:
            existing = await self._find_existing_by_idempotency(
                user_id=user_id,
                portfolio_id=portfolio_id,
                idempotency_key=clean_idem,
            )
            if existing is not None:
                return existing

        snap_id = f"snap_{uuid.uuid4().hex[:24]}"
        metadata_payload: dict[str, Any] = {
            "limitations": list(normalized.limitations),
            "malformed_records": list(normalized.malformed_records),
            "positions_count": len(normalized.positions),
            "positions_persisted": True,
        }

        async with self._db.session() as session:
            try:
                async with session.begin():
                    portfolio_row = (
                        await session.execute(
                            select(UserPortfolioRecord)
                            .where(
                                UserPortfolioRecord.portfolio_id == portfolio_id,
                                UserPortfolioRecord.user_id == user_id,
                            )
                            .with_for_update()
                        )
                    ).scalar_one_or_none()
                    if portfolio_row is None:
                        raise EntityNotFoundError("Portfolio not found for authenticated user.")

                    if clean_idem:
                        locked_existing = (
                            await session.execute(
                                select(PortfolioSnapshotRecord)
                                .options(
                                    selectinload(PortfolioSnapshotRecord.holdings),
                                    selectinload(PortfolioSnapshotRecord.positions),
                                )
                                .where(
                                    PortfolioSnapshotRecord.portfolio_id == portfolio_id,
                                    PortfolioSnapshotRecord.user_id == user_id,
                                    PortfolioSnapshotRecord.idempotency_key == clean_idem,
                                    PortfolioSnapshotRecord.status != "failed",
                                )
                            )
                        ).scalar_one_or_none()
                        if locked_existing is not None:
                            return _to_snapshot_entity(locked_existing)

                    snap = PortfolioSnapshotRecord(
                        snapshot_id=snap_id,
                        portfolio_id=portfolio_id,
                        user_id=user_id,
                        broker_connection_id=broker_connection_id,
                        broker_name=normalized.broker_name,
                        broker_account_id=normalized.broker_account_id,
                        fetched_at=normalized.fetched_at,
                        status=normalized.status,
                        idempotency_key=clean_idem,
                        cash_balance=normalized.cash_balance,
                        currency=normalized.currency,
                        valuation_available=normalized.valuation_available,
                        holdings_count=len(normalized.holdings),
                        snapshot_metadata=metadata_payload,
                        error_detail=error_detail,
                    )
                    session.add(snap)

                    for h in normalized.holdings:
                        session.add(
                            PortfolioHoldingRecord(
                                holding_id=f"hold_{uuid.uuid4().hex[:24]}",
                                snapshot_id=snap_id,
                                symbol=h.symbol,
                                exchange=h.exchange,
                                instrument_identifier=h.instrument_identifier or "",
                                isin=h.isin,
                                quantity=h.quantity,
                                average_purchase_price=h.average_purchase_price,
                                last_price=h.last_price,
                                currency=h.currency,
                                holding_metadata={
                                    **dict(h.metadata),
                                    "price_source": h.price_source,
                                    "valuation_available": h.valuation_available,
                                },
                            )
                        )

                    for p in normalized.positions:
                        session.add(
                            PortfolioPositionRecord(
                                position_id=f"pos_{uuid.uuid4().hex[:24]}",
                                snapshot_id=snap_id,
                                symbol=p.symbol,
                                exchange=p.exchange,
                                instrument_identifier=p.instrument_identifier or "",
                                product=p.product or "",
                                quantity=p.quantity,
                                average_price=p.average_price,
                                last_price=p.last_price,
                                pnl=p.pnl,
                                unrealised=p.unrealised,
                                realised=p.realised,
                                currency=p.currency,
                                position_metadata={},
                            )
                        )

                    await session.flush()

                refreshed = await self.get_snapshot_by_id(portfolio_id, snap_id, user_id=user_id)
                assert refreshed is not None
                return refreshed
            except IntegrityError:
                if clean_idem:
                    existing = await self._find_existing_by_idempotency(
                        user_id=user_id,
                        portfolio_id=portfolio_id,
                        idempotency_key=clean_idem,
                    )
                    if existing is not None:
                        return existing
                raise

    async def record_failed_refresh_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        broker_name: str,
        broker_account_id: str | None = None,
        broker_connection_id: str | None = None,
        error_detail: str,
        idempotency_key: str | None = None,
        fetched_at: datetime | None = None,
    ) -> PortfolioSnapshotEntity:
        clean_idem = idempotency_key.strip() if idempotency_key else None
        snap_id = f"snap_{uuid.uuid4().hex[:24]}"
        ts = fetched_at or datetime.now(timezone.utc)
        metadata: dict[str, Any] = {"limitations": [error_detail], "positions_persisted": True}
        if clean_idem:
            metadata["requested_idempotency_key"] = clean_idem

        async with self._db.session() as session:
            async with session.begin():
                portfolio_row = (
                    await session.execute(
                        select(UserPortfolioRecord).where(
                            UserPortfolioRecord.portfolio_id == portfolio_id,
                            UserPortfolioRecord.user_id == user_id,
                        )
                    )
                ).scalar_one_or_none()
                if portfolio_row is None:
                    raise EntityNotFoundError("Portfolio not found for authenticated user.")

                snap = PortfolioSnapshotRecord(
                    snapshot_id=snap_id,
                    portfolio_id=portfolio_id,
                    user_id=user_id,
                    broker_connection_id=broker_connection_id,
                    broker_name=broker_name,
                    broker_account_id=broker_account_id,
                    fetched_at=ts,
                    status="failed",
                    # Failed refreshes must NEVER reserve the indexed idempotency_key so safe retries succeed.
                    idempotency_key=None,
                    cash_balance=None,
                    currency=portfolio_row.currency,
                    valuation_available=False,
                    holdings_count=0,
                    snapshot_metadata=metadata,
                    error_detail=error_detail,
                )
                session.add(snap)
                await session.flush()

        refreshed = await self.get_snapshot_by_id(portfolio_id, snap_id, user_id=user_id)
        assert refreshed is not None
        return refreshed

    async def get_latest_completed_snapshot(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(PortfolioSnapshotRecord)
                .options(
                    selectinload(PortfolioSnapshotRecord.holdings),
                    selectinload(PortfolioSnapshotRecord.positions),
                )
                .where(
                    PortfolioSnapshotRecord.portfolio_id == portfolio_id,
                    PortfolioSnapshotRecord.user_id == user_id,
                    PortfolioSnapshotRecord.status == "completed",
                )
                .order_by(desc(PortfolioSnapshotRecord.fetched_at), desc(PortfolioSnapshotRecord.created_at))
                .limit(1)
            )
            rec = result.scalar_one_or_none()
            return _to_snapshot_entity(rec) if rec is not None else None

    async def list_portfolio_snapshots(
        self,
        portfolio_id: str,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[PortfolioSnapshotEntity]:
        async with self._db.session() as session:
            result = await session.execute(
                select(PortfolioSnapshotRecord)
                .options(
                    selectinload(PortfolioSnapshotRecord.holdings),
                    selectinload(PortfolioSnapshotRecord.positions),
                )
                .where(
                    PortfolioSnapshotRecord.portfolio_id == portfolio_id,
                    PortfolioSnapshotRecord.user_id == user_id,
                )
                .order_by(desc(PortfolioSnapshotRecord.fetched_at), desc(PortfolioSnapshotRecord.created_at))
                .limit(max(1, min(limit, 200)))
            )
            return [_to_snapshot_entity(r) for r in result.scalars().all()]

    async def get_snapshot_by_id(
        self,
        portfolio_id: str,
        snapshot_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(PortfolioSnapshotRecord)
                .options(
                    selectinload(PortfolioSnapshotRecord.holdings),
                    selectinload(PortfolioSnapshotRecord.positions),
                )
                .where(
                    PortfolioSnapshotRecord.portfolio_id == portfolio_id,
                    PortfolioSnapshotRecord.snapshot_id == snapshot_id,
                    PortfolioSnapshotRecord.user_id == user_id,
                )
            )
            rec = result.scalar_one_or_none()
            return _to_snapshot_entity(rec) if rec is not None else None


class InMemoryBrokerPersistenceRepository:
    """In-memory transactional fake of BrokerPersistenceRepository for isolated unit/API tests."""

    def __init__(self) -> None:
        self._users: dict[str, UserEntity] = {}
        self._connections: dict[str, BrokerConnectionEntity] = {}
        self._portfolios: dict[str, UserPortfolioEntity] = {}
        self._snapshots: dict[str, PortfolioSnapshotEntity] = {}
        self._oauth_states: dict[str, BrokerOAuthStateEntity] = {}
        self.fail_on_snapshot_insert: Exception | None = None

    async def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        user_id: str | None = None,
    ) -> UserEntity:
        clean_email = email.strip().lower()
        if any(u.email == clean_email for u in self._users.values()):
            raise DuplicateEntityError("An account with that email already exists.")
        now = datetime.now(timezone.utc)
        uid = user_id or f"usr_{uuid.uuid4().hex[:24]}"
        entity = UserEntity(
            user_id=uid,
            email=clean_email,
            hashed_password=hashed_password,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self._users[uid] = entity
        return entity

    async def get_user_by_email(self, email: str) -> UserEntity | None:
        clean_email = email.strip().lower()
        for user in self._users.values():
            if user.email == clean_email:
                return user
        return None

    async def get_user_by_id(self, user_id: str) -> UserEntity | None:
        return self._users.get(user_id)

    async def upsert_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
        broker_account_id: str | None,
        encrypted_access_token: str,
        encryption_key_id: str,
        token_expires_at: datetime | None,
        status: BrokerConnectionStatus,
        last_authenticated_at: datetime,
    ) -> BrokerConnectionEntity:
        now = datetime.now(timezone.utc)
        clean_broker = broker_name.strip().lower()
        existing = await self.get_user_broker_connection(user_id=user_id, broker_name=clean_broker)
        conn_id = existing.connection_id if existing else f"conn_{uuid.uuid4().hex[:24]}"
        created_at = existing.created_at if existing else now
        entity = BrokerConnectionEntity(
            connection_id=conn_id,
            user_id=user_id,
            broker_name=clean_broker,
            broker_account_id=broker_account_id,
            encrypted_access_token=encrypted_access_token,
            encryption_key_id=encryption_key_id,
            token_expires_at=token_expires_at,
            status=status,
            last_authenticated_at=last_authenticated_at,
            revoked_at=None,
            created_at=created_at,
            updated_at=now,
        )
        self._connections[conn_id] = entity
        return entity

    async def get_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        conn = self._connections.get(connection_id)
        if conn is None or conn.user_id != user_id:
            return None
        return conn

    async def get_user_broker_connection(
        self,
        *,
        user_id: str,
        broker_name: str,
    ) -> BrokerConnectionEntity | None:
        clean_broker = broker_name.strip().lower()
        for conn in self._connections.values():
            if conn.user_id == user_id and conn.broker_name == clean_broker:
                return conn
        return None

    async def list_broker_connections(self, *, user_id: str) -> list[BrokerConnectionEntity]:
        return [c for c in self._connections.values() if c.user_id == user_id]

    async def mark_broker_connection_expired(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        conn = await self.get_broker_connection(connection_id, user_id=user_id)
        if conn is None:
            return None
        if conn.status == "revoked":
            return conn
        updated = BrokerConnectionEntity(
            connection_id=conn.connection_id,
            user_id=conn.user_id,
            broker_name=conn.broker_name,
            broker_account_id=conn.broker_account_id,
            encrypted_access_token=conn.encrypted_access_token,
            encryption_key_id=conn.encryption_key_id,
            token_expires_at=conn.token_expires_at,
            status="expired",
            last_authenticated_at=conn.last_authenticated_at,
            revoked_at=conn.revoked_at,
            created_at=conn.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._connections[conn.connection_id] = updated
        return updated

    async def revoke_broker_connection(
        self,
        connection_id: str,
        *,
        user_id: str,
    ) -> BrokerConnectionEntity | None:
        conn = await self.get_broker_connection(connection_id, user_id=user_id)
        if conn is None:
            return None
        now = datetime.now(timezone.utc)
        updated = BrokerConnectionEntity(
            connection_id=conn.connection_id,
            user_id=conn.user_id,
            broker_name=conn.broker_name,
            broker_account_id=conn.broker_account_id,
            encrypted_access_token=None,
            encryption_key_id=None,
            token_expires_at=None,
            status="revoked",
            last_authenticated_at=conn.last_authenticated_at,
            revoked_at=now,
            created_at=conn.created_at,
            updated_at=now,
        )
        self._connections[conn.connection_id] = updated
        return updated

    async def set_user_active(
        self,
        user_id: str,
        *,
        is_active: bool,
    ) -> UserEntity | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        updated = UserEntity(
            user_id=user.user_id,
            email=user.email,
            hashed_password=user.hashed_password,
            is_active=bool(is_active),
            created_at=user.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._users[user_id] = updated
        return updated

    async def delete_user(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False
        del self._users[user_id]
        for key in [k for k, v in self._oauth_states.items() if v.user_id == user_id]:
            del self._oauth_states[key]
        for key in [k for k, v in self._connections.items() if v.user_id == user_id]:
            del self._connections[key]
        for key in [k for k, v in self._snapshots.items() if v.user_id == user_id]:
            del self._snapshots[key]
        for key in [k for k, v in self._portfolios.items() if v.user_id == user_id]:
            del self._portfolios[key]
        return True

    async def create_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        user_id: str,
        broker_name: str,
        client_binding_hash: str,
        expires_at: datetime,
    ) -> BrokerOAuthStateEntity:
        user = self._users.get(user_id)
        if user is None or not user.is_active:
            raise OAuthStateRejectedError("Cannot issue OAuth state for missing or inactive user.")
        if nonce_hash in self._oauth_states:
            raise DuplicateEntityError("Duplicate OAuth state nonce hash.")
        now = datetime.now(timezone.utc)
        entity = BrokerOAuthStateEntity(
            state_id=f"ost_{uuid.uuid4().hex[:24]}",
            nonce_hash=nonce_hash,
            user_id=user_id,
            broker_name=broker_name.strip().lower(),
            client_binding_hash=client_binding_hash,
            expires_at=expires_at,
            consumed_at=None,
            created_at=now,
        )
        self._oauth_states[nonce_hash] = entity
        return entity

    async def consume_oauth_state_record(
        self,
        *,
        nonce_hash: str,
        broker_name: str,
        expected_user_id: str | None = None,
        client_binding_hash: str | None = None,
        now: datetime | None = None,
    ) -> BrokerOAuthStateEntity:
        if expected_user_id is None and client_binding_hash is None:
            raise OAuthStateRejectedError("OAuth callback lacks caller identity or browser binding proof.")

        rec = self._oauth_states.get(nonce_hash)
        if rec is None:
            raise OAuthStateRejectedError("OAuth callback state is unknown or invalid.")
        if rec.broker_name != broker_name.strip().lower():
            raise OAuthStateRejectedError("OAuth state broker binding mismatch.")
        if expected_user_id is not None and not hmac.compare_digest(
            rec.user_id, expected_user_id.strip()
        ):
            raise OAuthStateRejectedError("OAuth state user binding mismatch.")
        if client_binding_hash is not None and not hmac.compare_digest(
            rec.client_binding_hash, client_binding_hash
        ):
            raise OAuthStateRejectedError("OAuth state browser binding mismatch.")

        owner = self._users.get(rec.user_id)
        if owner is None or not owner.is_active:
            raise OAuthStateRejectedError("OAuth state owner account is inactive or deleted.")

        if rec.consumed_at is not None:
            raise OAuthStateRejectedError(
                "OAuth callback state nonce has already been consumed (replay rejected)."
            )

        current_ts = now or datetime.now(timezone.utc)
        exp = rec.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp <= current_ts:
            raise OAuthStateRejectedError("OAuth callback state has expired.")

        consumed = BrokerOAuthStateEntity(
            state_id=rec.state_id,
            nonce_hash=rec.nonce_hash,
            user_id=rec.user_id,
            broker_name=rec.broker_name,
            client_binding_hash=rec.client_binding_hash,
            expires_at=rec.expires_at,
            consumed_at=current_ts,
            created_at=rec.created_at,
        )
        self._oauth_states[nonce_hash] = consumed
        return consumed

    async def rotate_broker_connection_token(
        self,
        connection_id: str,
        *,
        user_id: str,
        expected_encrypted_token: str,
        new_encrypted_token: str,
        new_encryption_key_id: str,
    ) -> BrokerConnectionEntity | None:
        conn = await self.get_broker_connection(connection_id, user_id=user_id)
        if conn is None or conn.status != "active" or conn.encrypted_access_token != expected_encrypted_token:
            return None
        updated = BrokerConnectionEntity(
            connection_id=conn.connection_id,
            user_id=conn.user_id,
            broker_name=conn.broker_name,
            broker_account_id=conn.broker_account_id,
            encrypted_access_token=new_encrypted_token,
            encryption_key_id=new_encryption_key_id,
            token_expires_at=conn.token_expires_at,
            status=conn.status,
            last_authenticated_at=conn.last_authenticated_at,
            revoked_at=conn.revoked_at,
            created_at=conn.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._connections[conn.connection_id] = updated
        return updated

    async def create_user_portfolio(
        self,
        *,
        user_id: str,
        name: str,
        broker_name: str,
        broker_connection_id: str | None = None,
        currency: str = "INR",
        portfolio_id: str | None = None,
    ) -> UserPortfolioEntity:
        clean_name = name.strip()
        if any(
            p.user_id == user_id and p.name == clean_name
            for p in self._portfolios.values()
        ):
            raise DuplicateEntityError("A portfolio with that name already exists for this user.")
        now = datetime.now(timezone.utc)
        pid = portfolio_id or f"pf_{uuid.uuid4().hex[:24]}"
        entity = UserPortfolioEntity(
            portfolio_id=pid,
            user_id=user_id,
            broker_connection_id=broker_connection_id,
            name=clean_name,
            broker_name=broker_name.strip().lower(),
            currency=currency.strip().upper(),
            created_at=now,
            updated_at=now,
        )
        self._portfolios[pid] = entity
        return entity

    async def get_user_portfolio(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> UserPortfolioEntity | None:
        p = self._portfolios.get(portfolio_id)
        if p is None or p.user_id != user_id:
            return None
        return p

    async def list_user_portfolios(self, *, user_id: str) -> list[UserPortfolioEntity]:
        return [p for p in self._portfolios.values() if p.user_id == user_id]

    async def create_portfolio_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        normalized: NormalizedPortfolioResult,
        broker_connection_id: str | None = None,
        idempotency_key: str | None = None,
        error_detail: str | None = None,
    ) -> PortfolioSnapshotEntity:
        portfolio = await self.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")

        clean_idem = idempotency_key.strip() if idempotency_key else None
        if clean_idem:
            existing = await self.get_snapshot_by_idempotency_key(
                user_id=user_id,
                portfolio_id=portfolio_id,
                idempotency_key=clean_idem,
            )
            if existing is not None:
                return existing

        if self.fail_on_snapshot_insert is not None:
            raise self.fail_on_snapshot_insert

        now = datetime.now(timezone.utc)
        snap_id = f"snap_{uuid.uuid4().hex[:24]}"
        holdings: list[PortfolioHoldingEntity] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for h in normalized.holdings:
            key = (h.symbol, h.exchange, h.instrument_identifier or "")
            if key in seen_keys:
                raise IntegrityError("duplicate holding key", params=None, orig=Exception("duplicate"))
            seen_keys.add(key)
            holdings.append(
                PortfolioHoldingEntity(
                    holding_id=f"hold_{uuid.uuid4().hex[:24]}",
                    snapshot_id=snap_id,
                    symbol=h.symbol,
                    exchange=h.exchange,
                    instrument_identifier=h.instrument_identifier or "",
                    isin=h.isin,
                    quantity=h.quantity,
                    average_purchase_price=h.average_purchase_price,
                    last_price=h.last_price,
                    currency=h.currency,
                    metadata={
                        **dict(h.metadata),
                        "price_source": h.price_source,
                        "valuation_available": h.valuation_available,
                    },
                    created_at=now,
                )
            )

        positions: list[PortfolioPositionEntity] = []
        seen_pos_keys: set[tuple[str, str, str, str]] = set()
        for p in normalized.positions:
            pos_key = (p.symbol, p.exchange, p.instrument_identifier or "", p.product or "")
            if pos_key in seen_pos_keys:
                raise IntegrityError("duplicate position key", params=None, orig=Exception("duplicate"))
            seen_pos_keys.add(pos_key)
            positions.append(
                PortfolioPositionEntity(
                    position_id=f"pos_{uuid.uuid4().hex[:24]}",
                    snapshot_id=snap_id,
                    symbol=p.symbol,
                    exchange=p.exchange,
                    instrument_identifier=p.instrument_identifier or "",
                    product=p.product or "",
                    quantity=p.quantity,
                    average_price=p.average_price,
                    last_price=p.last_price,
                    pnl=p.pnl,
                    unrealised=p.unrealised,
                    realised=p.realised,
                    currency=p.currency,
                    metadata={},
                    created_at=now,
                )
            )

        snap = PortfolioSnapshotEntity(
            snapshot_id=snap_id,
            portfolio_id=portfolio_id,
            user_id=user_id,
            broker_connection_id=broker_connection_id,
            broker_name=normalized.broker_name,
            broker_account_id=normalized.broker_account_id,
            fetched_at=normalized.fetched_at,
            status=normalized.status,
            idempotency_key=clean_idem,
            cash_balance=normalized.cash_balance,
            currency=normalized.currency,
            valuation_available=normalized.valuation_available,
            holdings_count=len(holdings),
            metadata={
                "limitations": list(normalized.limitations),
                "malformed_records": list(normalized.malformed_records),
                "positions_count": len(positions),
                "positions_persisted": True,
            },
            error_detail=error_detail,
            created_at=now,
            holdings=holdings,
            positions=positions,
        )
        self._snapshots[snap_id] = snap
        return snap

    async def record_failed_refresh_snapshot(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        broker_name: str,
        broker_account_id: str | None = None,
        broker_connection_id: str | None = None,
        error_detail: str,
        idempotency_key: str | None = None,
        fetched_at: datetime | None = None,
    ) -> PortfolioSnapshotEntity:
        portfolio = await self.get_user_portfolio(portfolio_id, user_id=user_id)
        if portfolio is None:
            raise EntityNotFoundError("Portfolio not found for authenticated user.")
        clean_idem = idempotency_key.strip() if idempotency_key else None
        now = datetime.now(timezone.utc)
        snap_id = f"snap_{uuid.uuid4().hex[:24]}"
        metadata: dict[str, Any] = {"limitations": [error_detail], "positions_persisted": True}
        if clean_idem:
            metadata["requested_idempotency_key"] = clean_idem
        snap = PortfolioSnapshotEntity(
            snapshot_id=snap_id,
            portfolio_id=portfolio_id,
            user_id=user_id,
            broker_connection_id=broker_connection_id,
            broker_name=broker_name,
            broker_account_id=broker_account_id,
            fetched_at=fetched_at or now,
            status="failed",
            idempotency_key=None,
            cash_balance=None,
            currency=portfolio.currency,
            valuation_available=False,
            holdings_count=0,
            metadata=metadata,
            error_detail=error_detail,
            created_at=now,
            holdings=[],
            positions=[],
        )
        self._snapshots[snap_id] = snap
        return snap

    async def get_snapshot_by_idempotency_key(
        self,
        *,
        user_id: str,
        portfolio_id: str,
        idempotency_key: str,
    ) -> PortfolioSnapshotEntity | None:
        clean_idem = idempotency_key.strip()
        if not clean_idem:
            return None
        for snap in self._snapshots.values():
            if (
                snap.portfolio_id == portfolio_id
                and snap.user_id == user_id
                and snap.status != "failed"
                and snap.idempotency_key == clean_idem
            ):
                return snap
        return None

    async def get_latest_completed_snapshot(
        self,
        portfolio_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None:
        matches = [
            s
            for s in self._snapshots.values()
            if s.portfolio_id == portfolio_id
            and s.user_id == user_id
            and s.status == "completed"
        ]
        if not matches:
            return None
        matches.sort(key=lambda s: (s.fetched_at, s.created_at), reverse=True)
        return matches[0]

    async def list_portfolio_snapshots(
        self,
        portfolio_id: str,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[PortfolioSnapshotEntity]:
        matches = [
            s
            for s in self._snapshots.values()
            if s.portfolio_id == portfolio_id and s.user_id == user_id
        ]
        matches.sort(key=lambda s: (s.fetched_at, s.created_at), reverse=True)
        return matches[: max(1, min(limit, 200))]

    async def get_snapshot_by_id(
        self,
        portfolio_id: str,
        snapshot_id: str,
        *,
        user_id: str,
    ) -> PortfolioSnapshotEntity | None:
        snap = self._snapshots.get(snapshot_id)
        if snap is None or snap.portfolio_id != portfolio_id or snap.user_id != user_id:
            return None
        return snap

