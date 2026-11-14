"""FastAPI dependency helpers resolving the authenticated AlphaPilot user."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.models import AuthenticatedUser
from app.auth.service import AuthConfigurationError, AuthError, AuthService
from app.db.broker_repository import BrokerPersistenceRepository
from app.db.dependencies import get_broker_repository

_bearer_scheme = HTTPBearer(auto_error=False)


def get_auth_service(request: Request) -> AuthService:
    """Resolve AuthService from app state (overridable in tests) or environment settings."""
    service = getattr(request.app.state, "auth_service", None)
    if service is not None:
        return service
    try:
        service = AuthService.from_settings()
    except AuthConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    request.app.state.auth_service = service
    return service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
) -> AuthenticatedUser:
    """Cryptographically verify bearer JWT AND confirm user exists and is active in DB."""
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_principal = auth_service.verify_access_token(credentials.credentials.strip())
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    db_user = await repo.get_user_by_id(token_principal.user_id)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user account does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )
    return AuthenticatedUser(
        user_id=db_user.user_id,
        email=db_user.email,
        is_active=True,
    )


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
) -> AuthenticatedUser | None:
    """Resolve authenticated user when Bearer header is provided; return None only when header is absent."""
    if credentials is None:
        return None
    return await get_current_user(credentials=credentials, auth_service=auth_service, repo=repo)

