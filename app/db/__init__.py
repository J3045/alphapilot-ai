"""Database persistence layer for AlphaPilot AI (PostgreSQL + pgvector)."""

from app.db.broker_repository import (
    BrokerConnectionEntity,
    BrokerPersistenceRepository,
    DuplicateEntityError,
    EntityNotFoundError,
    InMemoryBrokerPersistenceRepository,
    PortfolioHoldingEntity,
    PortfolioSnapshotEntity,
    PostgresBrokerPersistenceRepository,
    SnapshotComparisonResult,
    UserEntity,
    UserPortfolioEntity,
    compare_portfolio_snapshots,
)
from app.db.models import (
    Base,
    BrokerConnectionRecord,
    DocumentChunkRecord,
    PortfolioHoldingRecord,
    PortfolioSnapshotRecord,
    ResearchDocumentRecord,
    UserPortfolioRecord,
    UserRecord,
)
from app.db.repository import PostgresVectorRepository
from app.db.session import (
    DatabaseManager,
    dispose_default_database_manager,
    get_default_database_manager,
    has_database_configuration,
    reset_default_database_manager,
)

__all__ = [
    "Base",
    "BrokerConnectionEntity",
    "BrokerConnectionRecord",
    "BrokerPersistenceRepository",
    "DatabaseManager",
    "DocumentChunkRecord",
    "DuplicateEntityError",
    "EntityNotFoundError",
    "InMemoryBrokerPersistenceRepository",
    "PortfolioHoldingEntity",
    "PortfolioHoldingRecord",
    "PortfolioSnapshotEntity",
    "PortfolioSnapshotRecord",
    "PostgresBrokerPersistenceRepository",
    "PostgresVectorRepository",
    "ResearchDocumentRecord",
    "SnapshotComparisonResult",
    "UserEntity",
    "UserPortfolioEntity",
    "UserPortfolioRecord",
    "UserRecord",
    "compare_portfolio_snapshots",
    "dispose_default_database_manager",
    "get_default_database_manager",
    "has_database_configuration",
    "reset_default_database_manager",
]

