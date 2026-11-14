"""add broker connections and portfolio snapshots

Revision ID: 0002_add_broker_portfolio_snapshots
Revises: 0001_create_pgvector_rag
Create Date: 2026-11-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_add_broker_portfolio_snapshots"
down_revision: Union[str, Sequence[str], None] = "0001_create_pgvector_rag"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("hashed_password", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "broker_connections",
        sa.Column("connection_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("broker_name", sa.String(length=32), nullable=False),
        sa.Column("broker_account_id", sa.String(length=128), nullable=True),
        sa.Column("encrypted_access_token", sa.Text(), nullable=True),
        sa.Column("encryption_key_id", sa.String(length=64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_authenticated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("connection_id"),
        sa.UniqueConstraint("user_id", "broker_name", name="uq_broker_connections_user_broker"),
    )
    op.create_index("ix_broker_connections_user_id", "broker_connections", ["user_id"], unique=False)
    op.create_index("ix_broker_connections_user_status", "broker_connections", ["user_id", "status"], unique=False)

    op.create_table(
        "user_portfolios",
        sa.Column("portfolio_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("broker_connection_id", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("broker_name", sa.String(length=32), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["broker_connection_id"], ["broker_connections.connection_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("portfolio_id"),
        sa.UniqueConstraint("user_id", "name", name="uq_user_portfolios_user_name"),
    )
    op.create_index("ix_user_portfolios_user_id", "user_portfolios", ["user_id"], unique=False)
    op.create_index("ix_user_portfolios_broker_connection_id", "user_portfolios", ["broker_connection_id"], unique=False)

    op.create_table(
        "portfolio_snapshots",
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("portfolio_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("broker_connection_id", sa.String(length=64), nullable=True),
        sa.Column("broker_name", sa.String(length=32), nullable=False),
        sa.Column("broker_account_id", sa.String(length=128), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("cash_balance", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("valuation_available", sa.Boolean(), nullable=False),
        sa.Column("holdings_count", sa.Integer(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["user_portfolios.portfolio_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["broker_connection_id"], ["broker_connections.connection_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("portfolio_id", "idempotency_key", name="uq_portfolio_snapshots_portfolio_idempotency"),
    )
    op.create_index(
        "ix_portfolio_snapshots_portfolio_status_fetched",
        "portfolio_snapshots",
        ["portfolio_id", "status", "fetched_at"],
        unique=False,
    )
    op.create_index("ix_portfolio_snapshots_user_portfolio", "portfolio_snapshots", ["user_id", "portfolio_id"], unique=False)

    op.create_table(
        "portfolio_holdings",
        sa.Column("holding_id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("instrument_identifier", sa.String(length=128), nullable=False),
        sa.Column("isin", sa.String(length=32), nullable=True),
        sa.Column("quantity", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("average_purchase_price", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("last_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["portfolio_snapshots.snapshot_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("holding_id"),
        sa.UniqueConstraint(
            "snapshot_id",
            "symbol",
            "exchange",
            "instrument_identifier",
            name="uq_portfolio_holdings_snapshot_instrument",
        ),
    )
    op.create_index("ix_portfolio_holdings_snapshot_id", "portfolio_holdings", ["snapshot_id"], unique=False)
    op.create_index("ix_portfolio_holdings_symbol", "portfolio_holdings", ["symbol"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_portfolio_holdings_symbol", table_name="portfolio_holdings")
    op.drop_index("ix_portfolio_holdings_snapshot_id", table_name="portfolio_holdings")
    op.drop_table("portfolio_holdings")

    op.drop_index("ix_portfolio_snapshots_user_portfolio", table_name="portfolio_snapshots")
    op.drop_index("ix_portfolio_snapshots_portfolio_status_fetched", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")

    op.drop_index("ix_user_portfolios_broker_connection_id", table_name="user_portfolios")
    op.drop_index("ix_user_portfolios_user_id", table_name="user_portfolios")
    op.drop_table("user_portfolios")

    op.drop_index("ix_broker_connections_user_status", table_name="broker_connections")
    op.drop_index("ix_broker_connections_user_id", table_name="broker_connections")
    op.drop_table("broker_connections")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

