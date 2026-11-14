"""Database schema, Alembic migration, repository integrity, and PostgreSQL tests (Day 19)."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
from pathlib import Path
from typing import Any

from alembic import command as alembic_command
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import Numeric, UniqueConstraint, delete, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.config import settings
from app.db.broker_repository import (
    InMemoryBrokerPersistenceRepository,
    OAuthStateRejectedError,
    PostgresBrokerPersistenceRepository,
    compare_portfolio_snapshots,
)
from app.db.models import (
    Base,
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
    BrokerPositionItem,
    NormalizedHolding,
    NormalizedPortfolioResult,
)
from tests.pg_test_utils import require_isolated_pg_test_url


def _step_no_io(coro: Any) -> Any:
    """Drive purely in-memory repository coroutines cleanly without creating OS loops."""
    try:
        while True:
            yielded = coro.send(None)
            if yielded is not None:
                raise RuntimeError("Unexpected async wait primitive in synchronous harness.")
    except StopIteration as stop:
        return stop.value


def test_orm_schema_tables_precision_constraints_and_ddl_compile():
    table_names = set(Base.metadata.tables.keys())
    expected_tables = {
        "research_documents",
        "document_chunks",
        "users",
        "broker_connections",
        "broker_oauth_states",
        "user_portfolios",
        "portfolio_snapshots",
        "portfolio_holdings",
        "portfolio_positions",
    }
    assert expected_tables.issubset(table_names)

    # Financial Numeric precision check (never Float)
    snap_cash_col = PortfolioSnapshotRecord.__table__.c.cash_balance
    assert isinstance(snap_cash_col.type, Numeric)
    assert snap_cash_col.type.precision == 18
    assert snap_cash_col.type.scale == 4

    hold_qty_col = PortfolioHoldingRecord.__table__.c.quantity
    assert isinstance(hold_qty_col.type, Numeric)
    assert hold_qty_col.type.precision == 18
    assert hold_qty_col.type.scale == 6

    pos_qty_col = PortfolioPositionRecord.__table__.c.quantity
    assert isinstance(pos_qty_col.type, Numeric)
    assert pos_qty_col.type.precision == 18
    assert pos_qty_col.type.scale == 6

    hold_avg_col = PortfolioHoldingRecord.__table__.c.average_purchase_price
    assert isinstance(hold_avg_col.type, Numeric)
    assert hold_avg_col.type.precision == 18
    assert hold_avg_col.type.scale == 4

    # Foreign key CASCADE & uniqueness constraints
    conn_uqs = {
        c.name for c in BrokerConnectionRecord.__table__.constraints if isinstance(c, UniqueConstraint)
    }
    assert "uq_broker_connections_user_broker" in conn_uqs

    oauth_uqs = {
        c.name for c in BrokerOAuthStateRecord.__table__.constraints if isinstance(c, UniqueConstraint)
    }
    assert "uq_broker_oauth_states_nonce_hash" in oauth_uqs

    snap_indexes = {idx.name: idx for idx in PortfolioSnapshotRecord.__table__.indexes}
    assert "uq_portfolio_snapshots_portfolio_idempotency_active" in snap_indexes
    partial_idx = snap_indexes["uq_portfolio_snapshots_portfolio_idempotency_active"]
    assert partial_idx.unique is True

    hold_uqs = {
        c.name for c in PortfolioHoldingRecord.__table__.constraints if isinstance(c, UniqueConstraint)
    }
    assert "uq_portfolio_holdings_snapshot_instrument" in hold_uqs

    pos_uqs = {
        c.name for c in PortfolioPositionRecord.__table__.constraints if isinstance(c, UniqueConstraint)
    }
    assert "uq_portfolio_positions_snapshot_instrument_product" in pos_uqs

    # Verify clean PostgreSQL DDL compilation for all tables & indexes
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        ddl_sql = str(CreateTable(table).compile(dialect=dialect))
        assert f"CREATE TABLE {table.name}" in ddl_sql
        for idx in table.indexes:
            idx_sql = str(CreateIndex(idx).compile(dialect=dialect))
            assert "CREATE INDEX" in idx_sql or "CREATE UNIQUE INDEX" in idx_sql
            if idx.name == "uq_portfolio_snapshots_portfolio_idempotency_active":
                assert "WHERE" in idx_sql
                assert "status <> 'failed'" in idx_sql


def test_alembic_revisions_chain_cleanly_to_0003():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = Config(str(repo_root / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    assert head == "0003_sec_oauth_pos_idemp"

    revs = list(script.walk_revisions())
    assert len(revs) >= 3
    assert revs[0].revision == "0003_sec_oauth_pos_idemp"
    assert revs[0].down_revision == "0002_add_broker_portfolio_snapshots"
    assert revs[1].revision == "0002_add_broker_portfolio_snapshots"
    assert revs[1].down_revision == "0001_create_pgvector_rag"


def test_repository_atomic_snapshots_idempotency_isolation_and_diffs():
    repo = InMemoryBrokerPersistenceRepository()

    alice = _step_no_io(repo.create_user(email="alice@example.com", hashed_password="h1"))
    bob = _step_no_io(repo.create_user(email="bob@example.com", hashed_password="h2"))

    pf_alice = _step_no_io(
        repo.create_user_portfolio(
            user_id=alice.user_id,
            name="Zerodha Core",
            broker_name="zerodha",
        )
    )

    t1 = datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc)
    t2 = t1 + timedelta(hours=2)
    t3 = t2 + timedelta(hours=1)

    norm_1 = NormalizedPortfolioResult(
        broker_name="zerodha",
        broker_account_id="AB1111",
        currency="INR",
        cash_balance=Decimal("10000.5000"),
        holdings=[
            NormalizedHolding(
                symbol="INFY",
                exchange="NSE",
                instrument_identifier="408065",
                isin="INE009A01021",
                quantity=Decimal("10.000000"),
                settled_quantity=Decimal("10"),
                t1_quantity=Decimal("0"),
                average_purchase_price=Decimal("1500.0000"),
                last_price=Decimal("1800.0000"),
                currency="INR",
                price_source="broker_ltp",
                valuation_available=True,
            )
        ],
        status="completed",
        valuation_available=True,
        fetched_at=t1,
    )

    # Idempotency check: same key returns exact original snapshot without duplicate insertion
    snap1 = _step_no_io(
        repo.create_portfolio_snapshot(
            user_id=alice.user_id,
            portfolio_id=pf_alice.portfolio_id,
            normalized=norm_1,
            idempotency_key="refresh-001",
        )
    )
    snap1_dup = _step_no_io(
        repo.create_portfolio_snapshot(
            user_id=alice.user_id,
            portfolio_id=pf_alice.portfolio_id,
            normalized=norm_1,
            idempotency_key="refresh-001",
        )
    )
    assert snap1_dup.snapshot_id == snap1.snapshot_id

    # Cross-user ownership denial on read & snapshot insert
    assert _step_no_io(repo.get_user_portfolio(pf_alice.portfolio_id, user_id=bob.user_id)) is None
    assert _step_no_io(repo.get_snapshot_by_id(pf_alice.portfolio_id, snap1.snapshot_id, user_id=bob.user_id)) is None

    # Failed refresh NEVER supersedes completed snapshot as latest completed
    failed_snap = _step_no_io(
        repo.record_failed_refresh_snapshot(
            user_id=alice.user_id,
            portfolio_id=pf_alice.portfolio_id,
            broker_name="zerodha",
            error_detail="Upstream rate limit",
            fetched_at=t2,
        )
    )
    assert failed_snap.status == "failed"
    latest = _step_no_io(
        repo.get_latest_completed_snapshot(pf_alice.portfolio_id, user_id=alice.user_id)
    )
    assert latest is not None and latest.snapshot_id == snap1.snapshot_id

    # Second completed refresh creates new immutable snapshot and supports historical comparison
    norm_2 = NormalizedPortfolioResult(
        broker_name="zerodha",
        broker_account_id="AB1111",
        currency="INR",
        cash_balance=Decimal("8200.5000"),
        holdings=[
            NormalizedHolding(
                symbol="INFY",
                exchange="NSE",
                instrument_identifier="408065",
                isin="INE009A01021",
                quantity=Decimal("11.000000"),
                settled_quantity=Decimal("10"),
                t1_quantity=Decimal("1"),
                average_purchase_price=Decimal("1520.0000"),
                last_price=Decimal("1850.0000"),
                currency="INR",
                price_source="broker_ltp",
                valuation_available=True,
            ),
            NormalizedHolding(
                symbol="TCS",
                exchange="NSE",
                instrument_identifier="2953217",
                quantity=Decimal("2.000000"),
                settled_quantity=Decimal("2"),
                t1_quantity=Decimal("0"),
                average_purchase_price=Decimal("4000.0000"),
                last_price=Decimal("4100.0000"),
                currency="INR",
                price_source="broker_ltp",
                valuation_available=True,
            ),
        ],
        status="completed",
        valuation_available=True,
        fetched_at=t3,
    )
    snap2 = _step_no_io(
        repo.create_portfolio_snapshot(
            user_id=alice.user_id,
            portfolio_id=pf_alice.portfolio_id,
            normalized=norm_2,
            idempotency_key="refresh-002",
        )
    )

    latest_after = _step_no_io(
        repo.get_latest_completed_snapshot(pf_alice.portfolio_id, user_id=alice.user_id)
    )
    assert latest_after is not None and latest_after.snapshot_id == snap2.snapshot_id

    history = _step_no_io(
        repo.list_portfolio_snapshots(pf_alice.portfolio_id, user_id=alice.user_id)
    )
    assert [s.snapshot_id for s in history] == [snap2.snapshot_id, failed_snap.snapshot_id, snap1.snapshot_id]

    diff = compare_portfolio_snapshots(snap1, snap2)
    assert diff.cash_delta == Decimal("-1800.0000")
    assert diff.added_symbols == ["TCS"]
    assert diff.removed_symbols == []
    infy_change = next(c for c in diff.holding_changes if c.symbol == "INFY")
    assert infy_change.quantity_delta == Decimal("1.000000")


@pytest.mark.integration
def test_postgres_snapshot_persistence_and_rollback_integration():
    """Isolated live PostgreSQL verification for snapshots, positions, failed retry, rollback, and concurrency."""
    test_db_url = require_isolated_pg_test_url()
    repo_root = Path(__file__).resolve().parent.parent
    alembic_cfg = Config(str(repo_root / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
    alembic_command.upgrade(alembic_cfg, "head")

    async def _run() -> None:
        db = DatabaseManager(database_url=test_db_url)
        created_user_id: str | None = None
        repo = PostgresBrokerPersistenceRepository(db)
        try:
            await db.verify_connection_and_pgvector()
            email = f"pgtest_{os.getpid()}_{datetime.now(timezone.utc).timestamp()}@example.com"
            user = await repo.create_user(email=email, hashed_password="pbkdf2_sha256$210000$s$d")
            created_user_id = user.user_id
            portfolio = await repo.create_user_portfolio(
                user_id=user.user_id,
                name="PG Snapshot Portfolio",
                broker_name="zerodha",
            )

            # 1. Failed refresh must NOT consume or poison idempotency_key
            failed = await repo.record_failed_refresh_snapshot(
                user_id=user.user_id,
                portfolio_id=portfolio.portfolio_id,
                broker_name="zerodha",
                error_detail="Simulated transient failure",
                idempotency_key="pg-idem-retryable",
            )
            assert failed.status == "failed"
            assert failed.idempotency_key is None
            assert failed.metadata.get("requested_idempotency_key") == "pg-idem-retryable"

            # 2. Retrying the exact same idempotency_key succeeds and persists holdings + positions atomically
            norm = NormalizedPortfolioResult(
                broker_name="zerodha",
                broker_account_id="PG001",
                currency="INR",
                cash_balance=Decimal("5000.0000"),
                holdings=[
                    NormalizedHolding(
                        symbol="INFY",
                        exchange="NSE",
                        instrument_identifier="408065",
                        quantity=Decimal("4"),
                        settled_quantity=Decimal("4"),
                        t1_quantity=Decimal("0"),
                        average_purchase_price=Decimal("1500"),
                        last_price=Decimal("1800"),
                        currency="INR",
                        price_source="broker_ltp",
                        valuation_available=True,
                    )
                ],
                positions=[
                    BrokerPositionItem(
                        symbol="TCS",
                        exchange="NFO",
                        instrument_identifier="999111",
                        product="MIS",
                        quantity=Decimal("-2"),
                        average_price=Decimal("4100"),
                        last_price=Decimal("4050"),
                    )
                ],
                status="completed",
                valuation_available=True,
                fetched_at=datetime.now(timezone.utc),
            )
            snap_results = await asyncio.gather(
                repo.create_portfolio_snapshot(
                    user_id=user.user_id,
                    portfolio_id=portfolio.portfolio_id,
                    normalized=norm,
                    idempotency_key="pg-idem-retryable",
                ),
                repo.create_portfolio_snapshot(
                    user_id=user.user_id,
                    portfolio_id=portfolio.portfolio_id,
                    normalized=norm,
                    idempotency_key="pg-idem-retryable",
                ),
            )
            assert snap_results[0].snapshot_id == snap_results[1].snapshot_id
            loaded = await repo.get_latest_completed_snapshot(portfolio.portfolio_id, user_id=user.user_id)
            assert loaded is not None
            assert loaded.snapshot_id == snap_results[0].snapshot_id
            assert loaded.holdings[0].quantity == Decimal("4.000000")
            assert len(loaded.positions) == 1
            assert loaded.positions[0].quantity == Decimal("-2.000000")

            # 3. Rollback atomicity: duplicate holding inside same snapshot rolls back entire snapshot header
            dup_item = norm.holdings[0]
            broken_norm = NormalizedPortfolioResult(
                broker_name="zerodha",
                broker_account_id="PG001",
                currency="INR",
                cash_balance=Decimal("1000"),
                holdings=[dup_item, dup_item],
                positions=[],
                status="completed",
                valuation_available=True,
                fetched_at=datetime.now(timezone.utc),
            )
            before_count = len(await repo.list_portfolio_snapshots(portfolio.portfolio_id, user_id=user.user_id))
            with pytest.raises(Exception):
                await repo.create_portfolio_snapshot(
                    user_id=user.user_id,
                    portfolio_id=portfolio.portfolio_id,
                    normalized=broken_norm,
                )
            after_count = len(await repo.list_portfolio_snapshots(portfolio.portfolio_id, user_id=user.user_id))
            assert after_count == before_count

            # 4. Concurrent OAuth state consumption allows exactly one winner
            await repo.create_oauth_state_record(
                nonce_hash="a" * 64,
                user_id=user.user_id,
                broker_name="zerodha",
                client_binding_hash="b" * 64,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            oauth_outcomes = await asyncio.gather(
                repo.consume_oauth_state_record(
                    nonce_hash="a" * 64,
                    broker_name="zerodha",
                    client_binding_hash="b" * 64,
                ),
                repo.consume_oauth_state_record(
                    nonce_hash="a" * 64,
                    broker_name="zerodha",
                    client_binding_hash="b" * 64,
                ),
                return_exceptions=True,
            )
            successes = [r for r in oauth_outcomes if not isinstance(r, BaseException)]
            failures = [r for r in oauth_outcomes if isinstance(r, OAuthStateRejectedError)]
            assert len(successes) == 1
            assert len(failures) == 1
        finally:
            if created_user_id is not None:
                await repo.delete_user(created_user_id)
            await db.dispose()

    asyncio.run(_run())
