"""Database repository FastAPI dependency shared across auth and API layers."""

from fastapi import HTTPException, Request, status

from app.db.broker_repository import (
    BrokerPersistenceRepository,
    PostgresBrokerPersistenceRepository,
)
from app.db.session import get_default_database_manager, has_database_configuration


def get_broker_repository(request: Request) -> BrokerPersistenceRepository:
    """Resolve persistent repository backed by the shared DatabaseManager (or test override)."""
    repo = getattr(request.app.state, "broker_repository", None)
    if repo is not None:
        return repo
    if not has_database_configuration():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL database_url is not configured for broker and portfolio persistence.",
        )
    db_manager = getattr(request.app.state, "db_manager", None) or get_default_database_manager()
    request.app.state.db_manager = db_manager
    repo = PostgresBrokerPersistenceRepository(db_manager)
    request.app.state.broker_repository = repo
    return repo

