# pyrefly: ignore[missing-import]
from contextlib import asynccontextmanager
import logging
import threading
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import normalize_async_database_url, settings
from app.logging_config import redact_sensitive_text
from app.services.retrieval_service import VectorStoreConnectionError, VectorStoreError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLAlchemy async engine, connection pooling, and clean session lifecycle."""

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        pool_timeout_seconds: float | None = None,
        echo: bool | None = None,
    ) -> None:
        raw_url = database_url if database_url is not None else settings.get_async_database_url()
        self._database_url = normalize_async_database_url(raw_url)
        self.redacted_url = redact_sensitive_text(self._database_url)
        self.pool_size = pool_size if pool_size is not None else settings.db_pool_size
        self.max_overflow = max_overflow if max_overflow is not None else settings.db_max_overflow
        self.pool_timeout_seconds = (
            pool_timeout_seconds
            if pool_timeout_seconds is not None
            else settings.db_pool_timeout_seconds
        )
        self.echo = echo if echo is not None else settings.db_echo
        self._disposed = False

        self._engine: AsyncEngine = create_async_engine(
            self._database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout_seconds,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=self.echo,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info(
            "database_pool_initialized",
            extra={
                "target": self.redacted_url,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout_seconds": self.pool_timeout_seconds,
            },
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an AsyncSession with rollback on error and clean closure."""
        if self._disposed:
            raise VectorStoreConnectionError("DatabaseManager has already been disposed.")

        async with self._session_factory() as db_session:
            try:
                yield db_session
            except Exception:
                await db_session.rollback()
                raise

    async def verify_connection_and_pgvector(self) -> None:
        """Verify PostgreSQL connectivity and that the pgvector extension is enabled."""
        if self._disposed:
            raise VectorStoreConnectionError("DatabaseManager has already been disposed.")

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                ext_result = await conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = :ext_name"),
                    {"ext_name": "vector"},
                )
                if ext_result.scalar_one_or_none() != "vector":
                    raise VectorStoreError(
                        "pgvector extension is not enabled in target database; run Alembic migrations first."
                    )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.error(
                "database_health_check_failed",
                extra={"target": self.redacted_url, "error_type": type(exc).__name__},
            )
            raise VectorStoreConnectionError(
                f"PostgreSQL connection or pgvector verification failed ({type(exc).__name__})."
            ) from exc

    async def dispose(self) -> None:
        """Dispose the SQLAlchemy async engine connection pool cleanly."""
        if self._disposed:
            return
        self._disposed = True
        await self._engine.dispose()
        logger.info("database_pool_disposed", extra={"target": self.redacted_url})


_DEFAULT_DB_LOCK = threading.Lock()
_DEFAULT_DB_MANAGER: DatabaseManager | None = None


def has_database_configuration() -> bool:
    """Return True when PostgreSQL connection settings are configured."""
    return (
        settings.database_url is not None
        and bool(settings.database_url.get_secret_value().strip())
    )


def get_default_database_manager(database_url: str | None = None) -> DatabaseManager:
    """Return the shared process-local DatabaseManager singleton (single connection pool)."""
    global _DEFAULT_DB_MANAGER
    if _DEFAULT_DB_MANAGER is None or _DEFAULT_DB_MANAGER._disposed:
        with _DEFAULT_DB_LOCK:
            if _DEFAULT_DB_MANAGER is None or _DEFAULT_DB_MANAGER._disposed:
                _DEFAULT_DB_MANAGER = DatabaseManager(database_url=database_url)
    return _DEFAULT_DB_MANAGER


def reset_default_database_manager() -> None:
    """Reset the shared DatabaseManager singleton reference (primarily for tests)."""
    global _DEFAULT_DB_MANAGER
    with _DEFAULT_DB_LOCK:
        _DEFAULT_DB_MANAGER = None


async def dispose_default_database_manager() -> None:
    """Asynchronously dispose and clear the shared DatabaseManager singleton."""
    global _DEFAULT_DB_MANAGER
    manager: DatabaseManager | None = None
    with _DEFAULT_DB_LOCK:
        manager = _DEFAULT_DB_MANAGER
        _DEFAULT_DB_MANAGER = None
    if manager is not None:
        await manager.dispose()

