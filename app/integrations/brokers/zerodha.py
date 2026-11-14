"""Official Zerodha Kite Connect v3 read-only brokerage adapter."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import logging
import math
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr, ValidationError

from app.config import settings
from app.integrations.brokers.base import BrokerAdapter
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConfigurationError,
    BrokerError,
    BrokerPermissionError,
    BrokerRateLimitError,
    BrokerResponseValidationError,
    BrokerTimeoutError,
    BrokerTokenExpiredError,
    BrokerUnavailableError,
)
from app.integrations.brokers.models import (
    BrokerBalance,
    BrokerHoldingItem,
    BrokerHoldingsBatch,
    BrokerPositionItem,
    BrokerPositionsBatch,
    BrokerProfile,
    BrokerSessionToken,
    MalformedBrokerRecord,
    SupportedBroker,
)
from app.logging_config import redact_sensitive_text
from app.services.resilience import (
    WorkflowDeadline,
    WorkflowDeadlineExceededError,
    execute_with_retry_async,
    get_active_workflow_deadline,
)

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_KITE_VERSION = "3"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_broker_error(exc: Exception) -> bool:
    """Retry only transient network/timeout/rate-limit/5xx errors; never retry auth/4xx/validation."""
    return isinstance(exc, (BrokerTimeoutError, BrokerUnavailableError, BrokerRateLimitError))


def _parse_retry_after_seconds(headers: httpx.Headers | None) -> float | None:
    """Parse HTTP Retry-After header (numeric seconds or HTTP-date) safely."""
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None or not str(raw).strip():
        return None
    cleaned = str(raw).strip()
    try:
        secs = float(cleaned)
        if math.isfinite(secs) and secs > 0:
            return secs
        return None
    except ValueError:
        pass
    try:
        retry_dt = parsedate_to_datetime(cleaned)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
        if math.isfinite(delta) and delta > 0:
            return delta
    except Exception:
        return None
    return None


def compute_kite_session_expiry(now_utc: datetime | None = None) -> datetime:
    """Compute Kite Connect daily session token invalidation time (~07:30 IST next cycle)."""
    ref_utc = now_utc or datetime.now(timezone.utc)
    ref_ist = ref_utc.astimezone(_IST)
    cutoff_today = ref_ist.replace(hour=7, minute=30, second=0, microsecond=0)
    if ref_ist < cutoff_today:
        expiry_ist = cutoff_today
    else:
        expiry_ist = cutoff_today + timedelta(days=1)
    return expiry_ist.astimezone(timezone.utc)


def compute_kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Compute official Kite Connect SHA-256(api_key + request_token + api_secret) checksum."""
    raw = f"{api_key}{request_token}{api_secret}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ZerodhaBrokerAdapter(BrokerAdapter):
    """Async read-only adapter implementing Zerodha Kite Connect v3 REST protocol."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_base_url: str | None = None,
        login_base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        initial_backoff_seconds: float | None = None,
        max_backoff_seconds: float | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
        rng: Callable[[float, float], float] | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._api_secret = (api_secret or "").strip()
        self.api_base_url = (api_base_url or settings.zerodha_api_base_url).rstrip("/")
        self.login_base_url = (login_base_url or settings.zerodha_login_base_url).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.zerodha_timeout_seconds
        )
        self.max_retries = max_retries if max_retries is not None else settings.zerodha_max_retries
        self.initial_backoff_seconds = (
            initial_backoff_seconds
            if initial_backoff_seconds is not None
            else settings.retry_initial_backoff_seconds
        )
        self.max_backoff_seconds = (
            max_backoff_seconds
            if max_backoff_seconds is not None
            else settings.retry_max_backoff_seconds
        )
        self._sleep_fn = sleep_fn
        self._rng = rng
        self._external_client = http_client is not None
        self._http_client = http_client
        self._transport = transport

    @classmethod
    def from_settings(
        cls,
        *,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "ZerodhaBrokerAdapter":
        """Build ZerodhaBrokerAdapter from environment-backed Settings (no hardcoded secrets)."""
        api_key = settings.zerodha_api_key.get_secret_value() if settings.zerodha_api_key else None
        api_secret = (
            settings.zerodha_api_secret.get_secret_value() if settings.zerodha_api_secret else None
        )
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            api_base_url=settings.zerodha_api_base_url,
            login_base_url=settings.zerodha_login_base_url,
            timeout_seconds=settings.zerodha_timeout_seconds,
            max_retries=settings.zerodha_max_retries,
            http_client=http_client,
            transport=transport,
        )

    @property
    def broker_name(self) -> SupportedBroker:
        return "zerodha"

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise BrokerConfigurationError(
                "ZERODHA_API_KEY is not configured.",
                broker_name=self.broker_name,
            )
        return self._api_key

    def _require_credentials(self) -> tuple[str, str]:
        api_key = self._require_api_key()
        if not self._api_secret:
            raise BrokerConfigurationError(
                "ZERODHA_API_SECRET is not configured.",
                broker_name=self.broker_name,
            )
        return api_key, self._api_secret

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            client_kwargs: dict[str, Any] = {
                "base_url": self.api_base_url,
                "timeout": self.timeout_seconds,
                "headers": {"X-Kite-Version": _KITE_VERSION},
            }
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            self._http_client = httpx.AsyncClient(**client_kwargs)
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None and not self._external_client:
            await self._http_client.aclose()
            self._http_client = None

    def build_authorization_url(self, state: str, redirect_uri: str | None = None) -> str:
        """Build official Kite Connect v3 login URL wrapping CSRF state inside `redirect_params`."""
        api_key = self._require_api_key()
        if not state or not state.strip():
            raise BrokerAuthenticationError(
                "OAuth CSRF state parameter is required.",
                broker_name=self.broker_name,
            )

        if redirect_uri is not None and redirect_uri.strip():
            configured = (settings.zerodha_redirect_uri or "").strip()
            if not configured:
                raise BrokerAuthenticationError(
                    "Custom redirect_uri requires ZERODHA_REDIRECT_URI allowlist configuration.",
                    broker_name=self.broker_name,
                )
            if redirect_uri.strip() != configured:
                raise BrokerAuthenticationError(
                    "Provided redirect_uri does not match configured ZERODHA_REDIRECT_URI.",
                    broker_name=self.broker_name,
                )

        # Official Kite Connect passes custom callback parameters inside URL-encoded `redirect_params`.
        redirect_query = urlencode({"state": state.strip()})
        query_params = {
            "v": _KITE_VERSION,
            "api_key": api_key,
            "redirect_params": redirect_query,
        }
        return f"{self.login_base_url}?{urlencode(query_params)}"

    def _translate_kite_error(
        self,
        *,
        status_code: int,
        payload: Any,
        raw_text: str,
        is_token_exchange: bool = False,
        retry_after_seconds: float | None = None,
    ) -> BrokerError:
        """Translate official Kite JSON error envelopes or HTML edge responses into sanitized BrokerError."""
        if not isinstance(payload, dict):
            snippet = raw_text.lstrip()[:64].lower()
            if status_code >= 500 or snippet.startswith("<") or "html" in snippet:
                return BrokerUnavailableError(
                    f"Zerodha API gateway returned an unexpected non-JSON response (HTTP {status_code}).",
                    broker_name=self.broker_name,
                    status_code=status_code,
                )
            return BrokerResponseValidationError(
                f"Zerodha returned malformed error payload (HTTP {status_code}).",
                broker_name=self.broker_name,
                status_code=status_code,
            )

        error_type = str(payload.get("error_type") or "")
        message = str(payload.get("message") or f"Zerodha API error (HTTP {status_code})")

        if error_type == "TokenException" or status_code == 401:
            if is_token_exchange:
                return BrokerAuthenticationError(
                    "Zerodha session exchange failed: request token is invalid or expired.",
                    broker_name=self.broker_name,
                    error_type=error_type or "TokenException",
                    status_code=status_code,
                )
            return BrokerTokenExpiredError(
                "Zerodha session token has expired or been invalidated; reauthorization is required.",
                broker_name=self.broker_name,
                error_type=error_type or "TokenException",
                status_code=status_code,
            )

        if error_type == "PermissionException":
            return BrokerPermissionError(
                f"Zerodha API permission denied: {message}",
                broker_name=self.broker_name,
                error_type=error_type,
                status_code=status_code,
            )

        if error_type == "TooManyRequests" or status_code == 429:
            return BrokerRateLimitError(
                "Zerodha API rate limit exceeded. Please retry later.",
                broker_name=self.broker_name,
                error_type=error_type or "TooManyRequests",
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        if error_type == "InputException" or status_code == 400:
            if is_token_exchange:
                return BrokerAuthenticationError(
                    f"Zerodha login token verification failed: {message}",
                    broker_name=self.broker_name,
                    error_type=error_type or "InputException",
                    status_code=status_code,
                )
            return BrokerResponseValidationError(
                f"Zerodha rejected request input: {message}",
                broker_name=self.broker_name,
                error_type=error_type or "InputException",
                status_code=status_code,
            )

        if status_code == 403:
            return BrokerPermissionError(
                f"Zerodha access forbidden: {message}",
                broker_name=self.broker_name,
                error_type=error_type or "PermissionException",
                status_code=status_code,
            )

        if error_type in {"NetworkException", "DataException", "GeneralException"} or status_code >= 500:
            return BrokerUnavailableError(
                f"Zerodha service unavailable ({error_type or status_code}).",
                broker_name=self.broker_name,
                error_type=error_type,
                status_code=status_code,
            )

        return BrokerError(
            f"Zerodha API error ({error_type or status_code}): {message}",
            broker_name=self.broker_name,
            error_type=error_type,
            status_code=status_code,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        access_token: str | None = None,
        params: dict[str, Any] | None = None,
        form_data: dict[str, Any] | None = None,
        is_token_exchange: bool = False,
        deadline: WorkflowDeadline | None = None,
    ) -> Any:
        """Execute an HTTP request against Kite Connect using shared exponential backoff, jitter, and deadline control."""
        api_key = self._require_api_key()
        headers: dict[str, str] = {"X-Kite-Version": _KITE_VERSION}
        if access_token is not None:
            cleaned_token = access_token.strip()
            if not cleaned_token:
                raise BrokerAuthenticationError(
                    "Broker access token is missing.",
                    broker_name=self.broker_name,
                )
            headers["Authorization"] = f"token {api_key}:{cleaned_token}"

        client = self._get_client()
        effective_deadline = (
            deadline
            or get_active_workflow_deadline()
            or WorkflowDeadline(timeout_seconds=settings.zerodha_workflow_deadline_seconds)
        )

        async def _attempt() -> Any:
            try:
                response = await client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    params=params,
                    data=form_data,
                )
            except httpx.TimeoutException as exc:
                raise BrokerTimeoutError(
                    f"Zerodha request timed out calling {path}.",
                    broker_name=self.broker_name,
                ) from exc
            except httpx.HTTPError as exc:
                raise BrokerUnavailableError(
                    f"Network failure calling Zerodha API endpoint {path}.",
                    broker_name=self.broker_name,
                ) from exc

            try:
                parsed_json = response.json()
            except ValueError:
                parsed_json = None

            retry_after = _parse_retry_after_seconds(response.headers)

            if response.status_code >= 400:
                raise self._translate_kite_error(
                    status_code=response.status_code,
                    payload=parsed_json,
                    raw_text=response.text,
                    is_token_exchange=is_token_exchange,
                    retry_after_seconds=retry_after,
                )

            if not isinstance(parsed_json, dict):
                raise BrokerResponseValidationError(
                    f"Zerodha API returned invalid non-JSON response on {path}.",
                    broker_name=self.broker_name,
                    status_code=response.status_code,
                )

            if parsed_json.get("status") != "success":
                raise self._translate_kite_error(
                    status_code=response.status_code,
                    payload=parsed_json,
                    raw_text=response.text,
                    is_token_exchange=is_token_exchange,
                    retry_after_seconds=retry_after,
                )

            if "data" not in parsed_json:
                raise BrokerResponseValidationError(
                    f"Zerodha API response on {path} is missing the 'data' envelope field.",
                    broker_name=self.broker_name,
                    status_code=response.status_code,
                )

            return parsed_json["data"]

        try:
            return await execute_with_retry_async(
                _attempt,
                max_retries=self.max_retries,
                initial_backoff_seconds=self.initial_backoff_seconds,
                max_backoff_seconds=self.max_backoff_seconds,
                retryable_predicate=_is_retryable_broker_error,
                sleep_fn=self._sleep_fn,
                rng=self._rng,
                deadline=effective_deadline,
            )
        except WorkflowDeadlineExceededError as exc:
            raise BrokerTimeoutError(
                f"Zerodha workflow deadline exceeded calling {path}.",
                broker_name=self.broker_name,
                status_code=504,
            ) from exc

    async def exchange_request_token(self, request_token: str) -> BrokerSessionToken:
        """Exchange one-time Kite Connect `request_token` for an authenticated daily session token."""
        cleaned_request_token = (request_token or "").strip()
        if not cleaned_request_token:
            raise BrokerAuthenticationError(
                "Zerodha callback did not contain a valid request_token.",
                broker_name=self.broker_name,
            )

        api_key, api_secret = self._require_credentials()
        checksum = compute_kite_checksum(api_key, cleaned_request_token, api_secret)

        data = await self._request(
            "POST",
            "/session/token",
            form_data={
                "api_key": api_key,
                "request_token": cleaned_request_token,
                "checksum": checksum,
            },
            is_token_exchange=True,
        )
        if not isinstance(data, dict):
            raise BrokerResponseValidationError(
                "Zerodha session exchange returned unexpected data structure.",
                broker_name=self.broker_name,
            )

        access_token = str(data.get("access_token") or "").strip()
        broker_user_id = str(data.get("user_id") or "").strip()
        if not access_token or not broker_user_id:
            raise BrokerResponseValidationError(
                "Zerodha session exchange response missing 'access_token' or 'user_id'.",
                broker_name=self.broker_name,
            )

        login_time: datetime | None = None
        raw_login_time = data.get("login_time")
        if isinstance(raw_login_time, str) and raw_login_time.strip():
            try:
                login_time = datetime.fromisoformat(raw_login_time.replace(" ", "T"))
                if login_time.tzinfo is None:
                    login_time = login_time.replace(tzinfo=_IST).astimezone(timezone.utc)
            except ValueError:
                login_time = None

        return BrokerSessionToken(
            broker_name=self.broker_name,
            access_token=SecretStr(access_token),
            broker_account_id=broker_user_id,
            broker_user_name=data.get("user_name"),
            email=data.get("email"),
            exchanges=[str(x) for x in (data.get("exchanges") or [])],
            products=[str(x) for x in (data.get("products") or [])],
            login_time=login_time,
            expires_at=compute_kite_session_expiry(),
        )

    async def revoke_session(self, access_token: str) -> None:
        """Invalidate the Kite Connect access token via `DELETE /session/token`."""
        api_key = self._require_api_key()
        try:
            await self._request(
                "DELETE",
                "/session/token",
                params={"api_key": api_key, "access_token": access_token},
            )
        except BrokerTokenExpiredError:
            # Already invalid or expired on Zerodha side; local revocation can proceed safely.
            logger.info("zerodha_token_already_expired_during_revoke")

    async def get_profile(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerProfile:
        """Fetch authenticated Zerodha user profile from `GET /user/profile`."""
        data = await self._request(
            "GET",
            "/user/profile",
            access_token=access_token,
            deadline=deadline,
        )
        if not isinstance(data, dict):
            raise BrokerResponseValidationError(
                "Zerodha profile response is not a JSON object.",
                broker_name=self.broker_name,
            )
        broker_account_id = str(data.get("user_id") or "").strip()
        if not broker_account_id:
            raise BrokerResponseValidationError(
                "Zerodha profile response missing required 'user_id'.",
                broker_name=self.broker_name,
            )
        return BrokerProfile(
            broker_name=self.broker_name,
            broker_account_id=broker_account_id,
            user_name=data.get("user_name") or data.get("user_shortname"),
            email=data.get("email"),
            user_type=data.get("user_type"),
            exchanges=[str(x) for x in (data.get("exchanges") or [])],
            products=[str(x) for x in (data.get("products") or [])],
        )

    def _parse_holdings_rows(self, data: list[Any]) -> BrokerHoldingsBatch:
        """Parse Zerodha holdings rows individually so one malformed row does not discard valid rows."""
        holdings: list[BrokerHoldingItem] = []
        malformed: list[MalformedBrokerRecord] = []
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                malformed.append(
                    MalformedBrokerRecord(
                        record_type="holding",
                        index=idx,
                        symbol="UNKNOWN",
                        reason=f"Zerodha holding entry at index {idx} is not an object.",
                    )
                )
                continue
            raw_symbol = str(row.get("tradingsymbol") or "UNKNOWN").strip().upper()[:64] or "UNKNOWN"
            raw_exchange = str(row.get("exchange") or "NSE").strip().upper()[:32] or "NSE"
            try:
                holdings.append(
                    BrokerHoldingItem(
                        symbol=str(row.get("tradingsymbol") or ""),
                        exchange=str(row.get("exchange") or "NSE"),
                        instrument_identifier=(
                            str(row["instrument_token"])
                            if row.get("instrument_token") is not None
                            else None
                        ),
                        isin=row.get("isin"),
                        product=row.get("product"),
                        quantity=row.get("quantity"),
                        t1_quantity=row.get("t1_quantity", 0),
                        average_price=row.get("average_price"),
                        last_price=row.get("last_price"),
                        pnl=row.get("pnl"),
                        currency="INR",
                        raw_metadata={
                            k: row.get(k)
                            for k in ("authorised_quantity", "collateral_quantity", "discrepancy")
                            if k in row
                        },
                    )
                )
            except (ValidationError, ValueError, TypeError) as exc:
                malformed.append(
                    MalformedBrokerRecord(
                        record_type="holding",
                        index=idx,
                        symbol=raw_symbol,
                        exchange=raw_exchange,
                        reason=redact_sensitive_text(str(exc))[:500],
                    )
                )
        return BrokerHoldingsBatch(items=holdings, malformed_records=malformed)

    def _parse_positions_rows(self, net_rows: list[Any]) -> BrokerPositionsBatch:
        """Parse Zerodha net positions rows individually into valid items and sanitized diagnostics."""
        positions: list[BrokerPositionItem] = []
        malformed: list[MalformedBrokerRecord] = []
        for idx, row in enumerate(net_rows):
            if not isinstance(row, dict):
                malformed.append(
                    MalformedBrokerRecord(
                        record_type="position",
                        index=idx,
                        symbol="UNKNOWN",
                        reason=f"Zerodha net position entry at index {idx} is not an object.",
                    )
                )
                continue
            raw_symbol = str(row.get("tradingsymbol") or "UNKNOWN").strip().upper()[:64] or "UNKNOWN"
            raw_exchange = str(row.get("exchange") or "NSE").strip().upper()[:32] or "NSE"
            try:
                positions.append(
                    BrokerPositionItem(
                        symbol=str(row.get("tradingsymbol") or ""),
                        exchange=str(row.get("exchange") or "NSE"),
                        instrument_identifier=(
                            str(row["instrument_token"])
                            if row.get("instrument_token") is not None
                            else None
                        ),
                        product=row.get("product"),
                        quantity=row.get("quantity"),
                        average_price=row.get("average_price"),
                        last_price=row.get("last_price"),
                        pnl=row.get("pnl"),
                        unrealised=row.get("unrealised"),
                        realised=row.get("realised"),
                        currency="INR",
                    )
                )
            except (ValidationError, ValueError, TypeError) as exc:
                malformed.append(
                    MalformedBrokerRecord(
                        record_type="position",
                        index=idx,
                        symbol=raw_symbol,
                        exchange=raw_exchange,
                        reason=redact_sensitive_text(str(exc))[:500],
                    )
                )
        return BrokerPositionsBatch(items=positions, malformed_records=malformed)

    async def get_holdings_batch(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerHoldingsBatch:
        """Fetch equity demat + T+1 holdings preserving valid rows alongside malformed diagnostics."""
        data = await self._request(
            "GET",
            "/portfolio/holdings",
            access_token=access_token,
            deadline=deadline,
        )
        if not isinstance(data, list):
            raise BrokerResponseValidationError(
                "Zerodha holdings response is not a list.",
                broker_name=self.broker_name,
            )
        return self._parse_holdings_rows(data)

    async def get_holdings(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> list[BrokerHoldingItem]:
        """Fetch equity demat + T+1 holdings strictly (raises on any malformed row)."""
        batch = await self.get_holdings_batch(access_token, deadline=deadline)
        if batch.malformed_records:
            first = batch.malformed_records[0]
            raise BrokerResponseValidationError(
                f"Zerodha holding record at index {first.index} failed strict schema validation.",
                broker_name=self.broker_name,
            )
        return batch.items

    async def get_positions_batch(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerPositionsBatch:
        """Fetch consolidated `net` positions preserving valid rows alongside malformed diagnostics."""
        data = await self._request(
            "GET",
            "/portfolio/positions",
            access_token=access_token,
            deadline=deadline,
        )
        if not isinstance(data, dict):
            raise BrokerResponseValidationError(
                "Zerodha positions response is not a JSON object.",
                broker_name=self.broker_name,
            )

        net_rows = data.get("net", [])
        if not isinstance(net_rows, list):
            raise BrokerResponseValidationError(
                "Zerodha positions response 'net' field is not a list.",
                broker_name=self.broker_name,
            )
        return self._parse_positions_rows(net_rows)

    async def get_positions(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> list[BrokerPositionItem]:
        """Fetch consolidated `net` positions strictly (raises on any malformed row)."""
        batch = await self.get_positions_batch(access_token, deadline=deadline)
        if batch.malformed_records:
            first = batch.malformed_records[0]
            raise BrokerResponseValidationError(
                f"Zerodha net position record at index {first.index} failed validation.",
                broker_name=self.broker_name,
            )
        return batch.items

    async def get_balance(
        self,
        access_token: str,
        *,
        deadline: WorkflowDeadline | None = None,
    ) -> BrokerBalance:
        """Fetch equity margin and available cash balance from `GET /user/margins`."""
        data = await self._request(
            "GET",
            "/user/margins",
            access_token=access_token,
            deadline=deadline,
        )
        if not isinstance(data, dict):
            raise BrokerResponseValidationError(
                "Zerodha margins response is not a JSON object.",
                broker_name=self.broker_name,
            )

        equity = data.get("equity")
        if not isinstance(equity, dict):
            raise BrokerResponseValidationError(
                "Zerodha margins response missing 'equity' segment.",
                broker_name=self.broker_name,
            )

        available = equity.get("available")
        if not isinstance(available, dict):
            raise BrokerResponseValidationError(
                "Zerodha equity margins missing 'available' breakdown.",
                broker_name=self.broker_name,
            )
        utilised = equity.get("utilised") or {}
        if not isinstance(utilised, dict):
            raise BrokerResponseValidationError(
                "Zerodha equity margins 'utilised' field is malformed.",
                broker_name=self.broker_name,
            )

        raw_cash = available.get("live_balance")
        if raw_cash is None:
            raw_cash = available.get("cash")
        if raw_cash is None:
            raise BrokerResponseValidationError(
                "Zerodha equity margins missing available cash balance.",
                broker_name=self.broker_name,
            )

        try:
            return BrokerBalance(
                available_cash=raw_cash,
                net_balance=equity.get("net"),
                collateral=available.get("collateral", 0),
                utilised_debits=utilised.get("debits", 0),
                currency="INR",
                segment="equity",
                raw_metadata={
                    "enabled": equity.get("enabled"),
                    "cash": available.get("cash"),
                    "live_balance": available.get("live_balance"),
                    "adhoc_margin": available.get("adhoc_margin"),
                },
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise BrokerResponseValidationError(
                "Zerodha equity margins contained invalid numeric values.",
                broker_name=self.broker_name,
            ) from exc
