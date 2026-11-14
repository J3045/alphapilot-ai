"""Pydantic schemas for multi-user authentication and OAuth CSRF state claims."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRegisterRequest(BaseModel):
    """Registration payload for creating an AlphaPilot AI user account."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if "@" not in cleaned or "." not in cleaned.split("@", 1)[-1]:
            raise ValueError("A valid email address is required.")
        return cleaned


class UserLoginRequest(BaseModel):
    """Credential login payload for issuing a signed AlphaPilot bearer access token."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserTokenResponse(BaseModel):
    """Bearer access token response without exposing user password hashes."""

    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user_id: str
    email: str


class AuthenticatedUser(BaseModel):
    """Authenticated user identity resolved from bearer token claims."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    email: str
    is_active: bool = True


class OAuthStateClaims(BaseModel):
    """Verified signed CSRF state claims binding a broker login attempt to a single user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    broker_name: str
    nonce: str
    issued_at: datetime
    expires_at: datetime

