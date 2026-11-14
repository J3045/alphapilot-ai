"""User-scoped Zerodha Kite Connect OAuth and brokerage connection management routes."""

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import RedirectResponse

from app.api.dependencies import get_broker_connection_service, translate_service_exception
from app.api.schemas import ZerodhaConnectRequest
from app.auth.dependencies import get_current_user, get_optional_current_user
from app.auth.models import AuthenticatedUser
from app.config import settings
from app.integrations.brokers.models import BrokerAuthorizationRequest, BrokerConnectionSummary
from app.services.broker_service import BrokerConnectionService

router = APIRouter(prefix="/brokers", tags=["Brokers"])

OAUTH_BINDING_COOKIE_NAME = "alphapilot_oauth_bind"
_OAUTH_CALLBACK_COOKIE_PATH = "/brokers/zerodha/callback"


def _append_connected_query(base_uri: str) -> str:
    separator = "&" if "?" in base_uri else "?"
    return f"{base_uri}{separator}broker=zerodha&status=connected"


@router.post("/zerodha/connect", response_model=BrokerAuthorizationRequest)
async def connect_zerodha(
    response: Response,
    payload: ZerodhaConnectRequest | None = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: BrokerConnectionService = Depends(get_broker_connection_service),
) -> BrokerAuthorizationRequest:
    """Initiate Zerodha Kite Connect v3 login consent URL bound cleanly to persistent state and HttpOnly cookie."""
    try:
        auth_req, binding_secret = await service.initiate_browser_connect(
            user_id=current_user.user_id,
            broker_name="zerodha",
            redirect_uri=payload.redirect_uri if payload else None,
        )
        response.set_cookie(
            key=OAUTH_BINDING_COOKIE_NAME,
            value=binding_secret,
            max_age=settings.broker_oauth_state_ttl_seconds,
            path=_OAUTH_CALLBACK_COOKIE_PATH,
            httponly=True,
            samesite="lax",
            secure=settings.oauth_cookie_secure,
        )
        return auth_req
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get(
    "/zerodha/callback",
    response_model=BrokerConnectionSummary,
    responses={
        302: {"description": "Optional browser frontend redirect when ZERODHA_POST_CALLBACK_REDIRECT_URI is set."},
    },
)
async def zerodha_callback(
    request: Request,
    response: Response,
    request_token: str | None = Query(default=None),
    state: str | None = Query(default=None),
    status: str | None = Query(default="success"),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
    service: BrokerConnectionService = Depends(get_broker_connection_service),
) -> BrokerConnectionSummary | RedirectResponse:
    """Validate browser cookie or Bearer-bound single-use state, exchange request_token, and persist connection."""
    binding_secret = request.cookies.get(OAUTH_BINDING_COOKIE_NAME)
    try:
        summary = await service.handle_callback(
            broker_name="zerodha",
            request_token=request_token,
            state=state,
            callback_status=status,
            user_id=current_user.user_id if current_user is not None else None,
            binding_secret=binding_secret,
        )
    except Exception as exc:
        raise translate_service_exception(exc) from exc

    redirect_base = (settings.zerodha_post_callback_redirect_uri or "").strip()
    if redirect_base:
        redirect_resp = RedirectResponse(
            url=_append_connected_query(redirect_base),
            status_code=302,
        )
        redirect_resp.delete_cookie(
            key=OAUTH_BINDING_COOKIE_NAME,
            path=_OAUTH_CALLBACK_COOKIE_PATH,
            httponly=True,
            samesite="lax",
            secure=settings.oauth_cookie_secure,
        )
        return redirect_resp

    response.delete_cookie(
        key=OAUTH_BINDING_COOKIE_NAME,
        path=_OAUTH_CALLBACK_COOKIE_PATH,
        httponly=True,
        samesite="lax",
        secure=settings.oauth_cookie_secure,
    )
    return summary


@router.get("/connections", response_model=list[BrokerConnectionSummary])
async def list_broker_connections(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: BrokerConnectionService = Depends(get_broker_connection_service),
) -> list[BrokerConnectionSummary]:
    """List sanitized brokerage connection metadata owned by the authenticated user."""
    try:
        return await service.list_connections(user_id=current_user.user_id)
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.delete("/connections/{connection_id}", response_model=BrokerConnectionSummary)
async def revoke_broker_connection(
    connection_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: BrokerConnectionService = Depends(get_broker_connection_service),
) -> BrokerConnectionSummary:
    """Revoke upstream broker session and wipe encrypted token material for the owner."""
    try:
        return await service.revoke_connection(
            user_id=current_user.user_id,
            connection_id=connection_id,
        )
    except Exception as exc:
        raise translate_service_exception(exc) from exc

