"""Application service managing user-owned brokerage OAuth handshakes, encryption, and revocation."""

from datetime import datetime, timezone
import logging

from app.auth.service import (
    AuthError,
    AuthService,
    generate_oauth_binding_secret,
    hash_oauth_token,
)
from app.db.broker_repository import (
    BrokerConnectionEntity,
    BrokerPersistenceRepository,
    EntityNotFoundError,
    OAuthStateRejectedError,
)
from app.integrations.brokers.base import BrokerRegistry
from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerConfigurationError,
    BrokerError,
)
from app.integrations.brokers.models import (
    BrokerAuthorizationRequest,
    BrokerConnectionSummary,
    SupportedBroker,
)
from app.integrations.security.encryption import TokenEncryptionError, TokenEncryptionService

logger = logging.getLogger(__name__)


def _to_connection_summary(entity: BrokerConnectionEntity) -> BrokerConnectionSummary:
    return BrokerConnectionSummary(
        connection_id=entity.connection_id,
        broker_name=entity.broker_name,  # type: ignore[arg-type]
        broker_account_id=entity.broker_account_id,
        status=entity.status,
        token_expires_at=entity.token_expires_at,
        last_authenticated_at=entity.last_authenticated_at,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


class BrokerConnectionService:
    """Coordinates user-scoped broker authorization, encrypted token storage, and revocation."""

    def __init__(
        self,
        *,
        repository: BrokerPersistenceRepository,
        registry: BrokerRegistry,
        auth_service: AuthService,
        encryption_service: TokenEncryptionService | None = None,
    ) -> None:
        self._repo = repository
        self._registry = registry
        self._auth = auth_service
        self._encryption = encryption_service

    def _get_encryption(self) -> TokenEncryptionService:
        if self._encryption is None:
            self._encryption = TokenEncryptionService.from_settings()
        return self._encryption

    async def initiate_browser_connect(
        self,
        *,
        user_id: str,
        broker_name: SupportedBroker,
        redirect_uri: str | None = None,
        binding_secret: str | None = None,
    ) -> tuple[BrokerAuthorizationRequest, str]:
        """Create persistent single-use OAuth state bound to user + HttpOnly browser secret and return `(request, binding_secret)`."""
        db_user = await self._repo.get_user_by_id(user_id)
        if db_user is None or not db_user.is_active:
            raise BrokerAuthenticationError(
                "Authenticated user account is missing or inactive.",
                broker_name=broker_name,
            )

        adapter = self._registry.get(broker_name)
        raw_binding = (binding_secret or "").strip() or generate_oauth_binding_secret()
        state, claims, expires_at = self._auth.create_oauth_state_with_claims(
            user_id=user_id,
            broker_name=adapter.broker_name,
        )
        authorization_url = adapter.build_authorization_url(
            state=state,
            redirect_uri=redirect_uri,
        )
        try:
            await self._repo.create_oauth_state_record(
                nonce_hash=hash_oauth_token(claims.nonce),
                user_id=user_id,
                broker_name=adapter.broker_name,
                client_binding_hash=hash_oauth_token(raw_binding),
                expires_at=expires_at,
            )
        except OAuthStateRejectedError as exc:
            raise BrokerAuthenticationError(str(exc), broker_name=adapter.broker_name) from exc

        logger.info(
            "broker_connect_initiated",
            extra={"user_id": user_id, "broker_name": adapter.broker_name},
        )
        auth_req = BrokerAuthorizationRequest(
            broker_name=adapter.broker_name,
            authorization_url=authorization_url,
            state_expires_at=expires_at,
        )
        return auth_req, raw_binding

    async def initiate_connect(
        self,
        *,
        user_id: str,
        broker_name: SupportedBroker,
        redirect_uri: str | None = None,
        binding_secret: str | None = None,
    ) -> BrokerAuthorizationRequest:
        """Build signed CSRF-bound broker login URL persisted for single-use callback verification."""
        auth_req, _ = await self.initiate_browser_connect(
            user_id=user_id,
            broker_name=broker_name,
            redirect_uri=redirect_uri,
            binding_secret=binding_secret,
        )
        return auth_req

    async def handle_callback(
        self,
        *,
        broker_name: SupportedBroker,
        request_token: str | None,
        state: str | None,
        callback_status: str | None = "success",
        user_id: str | None = None,
        binding_secret: str | None = None,
    ) -> BrokerConnectionSummary:
        """Validate browser-bound or Bearer-bound persistent OAuth state, consume once, exchange token, and persist connection."""
        adapter = self._registry.get(broker_name)
        expected_user_id = user_id.strip() if user_id and user_id.strip() else None
        clean_binding = binding_secret.strip() if binding_secret and binding_secret.strip() else None

        if expected_user_id is None and clean_binding is None:
            raise BrokerAuthenticationError(
                "OAuth callback requires either the initiating browser binding cookie or Bearer authorization.",
                broker_name=adapter.broker_name,
            )

        if not state or not state.strip():
            raise BrokerAuthenticationError(
                "Missing OAuth CSRF state parameter on broker callback.",
                broker_name=adapter.broker_name,
            )

        try:
            claims = self._auth.verify_oauth_state_token(
                state.strip(),
                expected_broker_name=adapter.broker_name,
                expected_user_id=expected_user_id,
            )
        except AuthError as exc:
            raise BrokerAuthenticationError(
                f"Invalid or expired OAuth state parameter: {exc}",
                broker_name=adapter.broker_name,
            ) from exc

        try:
            consumed = await self._repo.consume_oauth_state_record(
                nonce_hash=hash_oauth_token(claims.nonce),
                broker_name=adapter.broker_name,
                expected_user_id=expected_user_id,
                client_binding_hash=hash_oauth_token(clean_binding) if clean_binding else None,
            )
        except OAuthStateRejectedError as exc:
            raise BrokerAuthenticationError(
                str(exc),
                broker_name=adapter.broker_name,
            ) from exc

        resolved_user_id = consumed.user_id

        if callback_status and callback_status.strip().lower() != "success":
            raise BrokerAuthenticationError(
                f"Broker authorization did not succeed (status={callback_status}).",
                broker_name=adapter.broker_name,
            )

        if not request_token or not request_token.strip():
            raise BrokerAuthenticationError(
                "Missing request_token on broker callback.",
                broker_name=adapter.broker_name,
            )

        session_token = await adapter.exchange_request_token(request_token.strip())
        encryption = self._get_encryption()
        encrypted_blob, key_id = encryption.encrypt(
            session_token.access_token.get_secret_value(),
            user_id=resolved_user_id,
            broker_name=adapter.broker_name,
        )

        now = datetime.now(timezone.utc)
        saved = await self._repo.upsert_broker_connection(
            user_id=resolved_user_id,
            broker_name=adapter.broker_name,
            broker_account_id=session_token.broker_account_id,
            encrypted_access_token=encrypted_blob,
            encryption_key_id=key_id,
            token_expires_at=session_token.expires_at,
            status="active",
            last_authenticated_at=now,
        )

        logger.info(
            "broker_connection_activated",
            extra={
                "user_id": resolved_user_id,
                "connection_id": saved.connection_id,
                "broker_name": saved.broker_name,
                "encryption_key_id": key_id,
            },
        )
        return _to_connection_summary(saved)

    async def list_connections(self, *, user_id: str) -> list[BrokerConnectionSummary]:
        """Return sanitized metadata for all brokerage connections belonging to `user_id`."""
        connections = await self._repo.list_broker_connections(user_id=user_id)
        return [_to_connection_summary(conn) for conn in connections]

    async def revoke_connection(
        self,
        *,
        user_id: str,
        connection_id: str,
    ) -> BrokerConnectionSummary:
        """Invalidate upstream broker session best-effort and wipe local encrypted token material."""
        conn = await self._repo.get_broker_connection(connection_id, user_id=user_id)
        if conn is None:
            raise EntityNotFoundError("Broker connection not found for authenticated user.")

        if conn.status != "revoked" and conn.encrypted_access_token:
            try:
                encryption = self._get_encryption()
                access_token = encryption.decrypt(
                    conn.encrypted_access_token,
                    user_id=user_id,
                    broker_name=conn.broker_name,
                )
                adapter = self._registry.get(conn.broker_name)
                await adapter.revoke_session(access_token)
            except (BrokerError, TokenEncryptionError, BrokerConfigurationError) as exc:
                # Proceed with wiping local encrypted secrets even if upstream revocation fails/expired.
                logger.warning(
                    "broker_upstream_revoke_best_effort_warning",
                    extra={
                        "user_id": user_id,
                        "connection_id": connection_id,
                        "broker_name": conn.broker_name,
                        "error_type": type(exc).__name__,
                    },
                )

        revoked = await self._repo.revoke_broker_connection(connection_id, user_id=user_id)
        assert revoked is not None
        logger.info(
            "broker_connection_revoked",
            extra={
                "user_id": user_id,
                "connection_id": connection_id,
                "broker_name": revoked.broker_name,
            },
        )
        return _to_connection_summary(revoked)

