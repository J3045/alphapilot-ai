# pyrefly: ignore[missing-import]
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings

DEFAULT_EMBEDDING_DIMENSION: int = settings.embedding_dimension
METADATA_COLUMN_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Base declarative class for AlphaPilot SQLAlchemy ORM models."""


class ResearchDocumentRecord(Base):
    """Persistent relational record for an ingested research or policy document."""

    __tablename__ = "research_documents"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        METADATA_COLUMN_TYPE,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunkRecord(Base):
    """Persistent text chunk record paired with its pgvector embedding."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
        Index("ix_document_chunks_embedding_model", "embedding_model"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(192), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("research_documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        METADATA_COLUMN_TYPE,
        nullable=False,
        default=dict,
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(DEFAULT_EMBEDDING_DIMENSION),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    document: Mapped[ResearchDocumentRecord] = relationship(
        back_populates="chunks",
    )


class UserRecord(Base):
    """Persistent AlphaPilot multi-user account owning broker connections and portfolios."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    broker_connections: Mapped[list["BrokerConnectionRecord"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    portfolios: Mapped[list["UserPortfolioRecord"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    oauth_states: Mapped[list["BrokerOAuthStateRecord"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class BrokerOAuthStateRecord(Base):
    """Persistent single-use OAuth state record preventing replay across workers and restarts."""

    __tablename__ = "broker_oauth_states"
    __table_args__ = (
        UniqueConstraint("nonce_hash", name="uq_broker_oauth_states_nonce_hash"),
        Index(
            "ix_broker_oauth_states_user_broker_expires",
            "user_id",
            "broker_name",
            "expires_at",
        ),
        Index(
            "ix_broker_oauth_states_expires_consumed",
            "expires_at",
            "consumed_at",
        ),
    )

    state_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    broker_name: Mapped[str] = mapped_column(String(32), nullable=False)
    client_binding_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped[UserRecord] = relationship(back_populates="oauth_states")


class BrokerConnectionRecord(Base):
    """Encrypted brokerage connection owned by a single AlphaPilot user."""

    __tablename__ = "broker_connections"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "broker_name",
            name="uq_broker_connections_user_broker",
        ),
        Index("ix_broker_connections_user_status", "user_id", "status"),
    )

    connection_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    broker_name: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_account_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    encrypted_access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    encryption_key_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    last_authenticated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped[UserRecord] = relationship(back_populates="broker_connections")


class UserPortfolioRecord(Base):
    """Logical brokerage portfolio linked to a user and optional broker connection."""

    __tablename__ = "user_portfolios"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "name",
            name="uq_user_portfolios_user_name",
        ),
    )

    portfolio_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    broker_connection_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("broker_connections.connection_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    broker_name: Mapped[str] = mapped_column(String(32), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped[UserRecord] = relationship(back_populates="portfolios")
    snapshots: Mapped[list["PortfolioSnapshotRecord"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PortfolioSnapshotRecord(Base):
    """Immutable historical point-in-time snapshot generated by manual portfolio refresh."""

    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        Index(
            "uq_portfolio_snapshots_portfolio_idempotency_active",
            "portfolio_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL AND status <> 'failed'"),
            sqlite_where=text("idempotency_key IS NOT NULL AND status != 'failed'"),
        ),
        Index(
            "ix_portfolio_snapshots_portfolio_status_fetched",
            "portfolio_id",
            "status",
            "fetched_at",
        ),
        Index("ix_portfolio_snapshots_user_portfolio", "user_id", "portfolio_id"),
    )

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    portfolio_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("user_portfolios.portfolio_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    broker_connection_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("broker_connections.connection_id", ondelete="SET NULL"),
        nullable=True,
    )
    broker_name: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_account_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cash_balance: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    valuation_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    holdings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    snapshot_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        METADATA_COLUMN_TYPE,
        nullable=False,
        default=dict,
    )
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    portfolio: Mapped[UserPortfolioRecord] = relationship(back_populates="snapshots")
    holdings: Mapped[list["PortfolioHoldingRecord"]] = relationship(
        back_populates="snapshot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    positions: Mapped[list["PortfolioPositionRecord"]] = relationship(
        back_populates="snapshot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PortfolioHoldingRecord(Base):
    """Immutable point-in-time normalized holding entry inside a PortfolioSnapshotRecord."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "symbol",
            "exchange",
            "instrument_identifier",
            name="uq_portfolio_holdings_snapshot_instrument",
        ),
        Index("ix_portfolio_holdings_symbol", "symbol"),
    )

    holding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("portfolio_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument_identifier: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    isin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    average_purchase_price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    last_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    holding_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        METADATA_COLUMN_TYPE,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    snapshot: Mapped[PortfolioSnapshotRecord] = relationship(back_populates="holdings")


class PortfolioPositionRecord(Base):
    """Immutable point-in-time normalized open/intraday position inside a PortfolioSnapshotRecord."""

    __tablename__ = "portfolio_positions"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "symbol",
            "exchange",
            "instrument_identifier",
            "product",
            name="uq_portfolio_positions_snapshot_instrument_product",
        ),
        Index("ix_portfolio_positions_symbol", "symbol"),
    )

    position_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("portfolio_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument_identifier: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    product: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    average_price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    last_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    unrealised: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    realised: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    position_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        METADATA_COLUMN_TYPE,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    snapshot: Mapped[PortfolioSnapshotRecord] = relationship(back_populates="positions")

