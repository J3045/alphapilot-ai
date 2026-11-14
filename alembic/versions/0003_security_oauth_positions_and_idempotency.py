"""add oauth states, portfolio positions, and partial snapshot idempotency index

Revision ID: 0003_sec_oauth_pos_idemp
Revises: 0002_add_broker_portfolio_snapshots
Create Date: 2026-11-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_sec_oauth_pos_idemp"
down_revision: Union[str, Sequence[str], None] = "0002_add_broker_portfolio_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "broker_oauth_states",
        sa.Column("state_id", sa.String(length=64), nullable=False),
        sa.Column("nonce_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("broker_name", sa.String(length=32), nullable=False),
        sa.Column("client_binding_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("state_id"),
        sa.UniqueConstraint("nonce_hash", name="uq_broker_oauth_states_nonce_hash"),
    )
    op.create_index("ix_broker_oauth_states_user_id", "broker_oauth_states", ["user_id"], unique=False)
    op.create_index(
        "ix_broker_oauth_states_user_broker_expires",
        "broker_oauth_states",
        ["user_id", "broker_name", "expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_broker_oauth_states_expires_consumed",
        "broker_oauth_states",
        ["expires_at", "consumed_at"],
        unique=False,
    )

    op.create_table(
        "portfolio_positions",
        sa.Column("position_id", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("instrument_identifier", sa.String(length=128), nullable=False),
        sa.Column("product", sa.String(length=32), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("average_price", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("last_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("pnl", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("unrealised", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("realised", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["snapshot_id"], ["portfolio_snapshots.snapshot_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("position_id"),
        sa.UniqueConstraint(
            "snapshot_id",
            "symbol",
            "exchange",
            "instrument_identifier",
            "product",
            name="uq_portfolio_positions_snapshot_instrument_product",
        ),
    )
    op.create_index("ix_portfolio_positions_snapshot_id", "portfolio_positions", ["snapshot_id"], unique=False)
    op.create_index("ix_portfolio_positions_symbol", "portfolio_positions", ["symbol"], unique=False)

    # Preserve caller idempotency_key in metadata for any legacy failed snapshots before replacing unique constraint.
    op.execute(
        sa.text(
            "UPDATE portfolio_snapshots "
            "SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{requested_idempotency_key}', to_jsonb(idempotency_key::text), true), "
            "idempotency_key = NULL "
            "WHERE status = 'failed' AND idempotency_key IS NOT NULL"
        )
    )
    op.drop_constraint(
        "uq_portfolio_snapshots_portfolio_idempotency",
        "portfolio_snapshots",
        type_="unique",
    )
    op.create_index(
        "uq_portfolio_snapshots_portfolio_idempotency_active",
        "portfolio_snapshots",
        ["portfolio_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL AND status <> 'failed'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_portfolio_snapshots_portfolio_idempotency_active",
        table_name="portfolio_snapshots",
    )
    op.create_unique_constraint(
        "uq_portfolio_snapshots_portfolio_idempotency",
        "portfolio_snapshots",
        ["portfolio_id", "idempotency_key"],
    )

    op.drop_index("ix_portfolio_positions_symbol", table_name="portfolio_positions")
    op.drop_index("ix_portfolio_positions_snapshot_id", table_name="portfolio_positions")
    op.drop_table("portfolio_positions")

    op.drop_index("ix_broker_oauth_states_expires_consumed", table_name="broker_oauth_states")
    op.drop_index("ix_broker_oauth_states_user_broker_expires", table_name="broker_oauth_states")
    op.drop_index("ix_broker_oauth_states_user_id", table_name="broker_oauth_states")
    op.drop_table("broker_oauth_states")

