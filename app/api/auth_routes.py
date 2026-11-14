"""Multi-user authentication endpoints issuing signed AlphaPilot bearer tokens."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_broker_repository
from app.auth.dependencies import get_auth_service
from app.auth.models import UserLoginRequest, UserRegisterRequest, UserTokenResponse
from app.auth.service import AuthService
from app.db.broker_repository import BrokerPersistenceRepository, DuplicateEntityError

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserTokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserRegisterRequest,
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserTokenResponse:
    """Create a user account using PBKDF2-HMAC-SHA256 hashing and issue an access token."""
    hashed_pw = auth_service.hash_password(payload.password)
    try:
        user = await repo.create_user(email=payload.email, hashed_password=hashed_pw)
    except DuplicateEntityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    token, expires_at = auth_service.issue_access_token(user_id=user.user_id, email=user.email)
    return UserTokenResponse(
        access_token=token,
        expires_at=expires_at,
        user_id=user.user_id,
        email=user.email,
    )


@router.post("/token", response_model=UserTokenResponse)
async def login_for_access_token(
    payload: UserLoginRequest,
    repo: BrokerPersistenceRepository = Depends(get_broker_repository),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserTokenResponse:
    """Verify user credentials and issue a signed bearer access token."""
    user = await repo.get_user_by_email(payload.email)
    if user is None or not auth_service.verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    token, expires_at = auth_service.issue_access_token(user_id=user.user_id, email=user.email)
    return UserTokenResponse(
        access_token=token,
        expires_at=expires_at,
        user_id=user.user_id,
        email=user.email,
    )

