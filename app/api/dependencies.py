"""FastAPI dependencies wiring database repositories, broker registry, encryption, and services."""

from fastapi import Depends, HTTPException, Request, status

from app.auth.dependencies import get_auth_service
from app.auth.service import AuthService
from app.db.broker_repository import (
    BrokerPersistenceRepository,
    DuplicateEntityError,
    EntityNotFoundError,
)
from app.db.dependencies import get_broker_repository
from app.integrations.brokers.base import BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConfigurationError,
    BrokerConnectionRevokedError,
    BrokerError,
    BrokerPermissionError,
    BrokerRateLimitError,
    BrokerResponseValidationError,
    BrokerTimeoutError,
    BrokerTokenExpiredError,
    BrokerUnavailableError,
    BrokerUnsupportedOperationError,
)
from app.integrations.brokers.zerodha import ZerodhaBrokerAdapter
from app.integrations.security.encryption import TokenEncryptionService
from app.services.broker_service import BrokerConnectionService
from app.services.portfolio_ingestion_service import PortfolioIngestionService
from app.services.resilience import sanitize_provider_error


def get_broker_registry(request: Request) -> BrokerRegistry:
    """Resolve BrokerRegistry singleton registering supported adapters cleanly."""
    registry = getattr(request.app.state, "broker_registry", None)
    if registry is None:
        registry = BrokerRegistry()
        registry.register(ZerodhaBrokerAdapter.from_settings())
        request.app.state.broker_registry = registry
    return registry


def get_token_encryption_service(request: Request) -> TokenEncryptionService | None:
    """Resolve optional test override or lazily construct env-backed AES-256-GCM encryption."""
    return getattr(request.app.state, "token_encryption_service", None)


def get_broker_connection_service(
    request: Request,
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
    registry: BrokerRegistry = Depends(get_broker_registry),
    auth_service: AuthService = Depends(get_auth_service),
) -> BrokerConnectionService:
    encryption = get_token_encryption_service(request)
    return BrokerConnectionService(
        repository=repo,
        registry=registry,
        auth_service=auth_service,
        encryption_service=encryption,
    )


def get_portfolio_ingestion_service(
    request: Request,
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
    registry: BrokerRegistry = Depends(get_broker_registry),
) -> PortfolioIngestionService:
    encryption = get_token_encryption_service(request)
    return PortfolioIngestionService(
        repository=repo,
        registry=registry,
        encryption_service=encryption,
    )


def translate_service_exception(exc: Exception) -> HTTPException:
    """Map domain, database, and sanitized broker exceptions to accurate HTTP status codes."""
    if isinstance(exc, EntityNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, DuplicateEntityError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    if isinstance(exc, BrokerConfigurationError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerTokenExpiredError):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Broker session expired (reauth_required). Please reconnect your brokerage account.",
        )
    if isinstance(exc, BrokerConnectionRevokedError):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerAuthenticationError):
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerPermissionError):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerRateLimitError):
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerTimeoutError):
        return HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerUnavailableError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerResponseValidationError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitize_provider_error(exc))
    if isinstance(exc, BrokerUnsupportedOperationError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, BrokerError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitize_provider_error(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected internal server error.")

