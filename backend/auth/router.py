from datetime import datetime, timedelta
import json
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.calendar_credentials import decrypt_calendar_credentials, encrypt_calendar_credentials
from backend.auth.dependencies import get_current_user
from backend.config import settings
from backend.auth.google import verify_google_id_token
from backend.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_refresh_token,
)
from backend.db.models import GoogleCalendarCredential, RefreshToken, User
from backend.db.session import get_db
from backend.timezone_utils import now_ist_naive, to_ist_naive
from planner_calendar import SCOPES


router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(
        min_length=10,
        description="Google OAuth ID token obtained on the client side.",
    )


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    avatar: str


class AuthResponse(BaseModel):
    access_token: str
    user: UserOut


class LogoutResponse(BaseModel):
    success: bool


class CalendarOAuthUrlResponse(BaseModel):
    authorization_url: str


class CalendarConnectRequest(BaseModel):
    authorization_code: str = Field(min_length=8)
    redirect_uri: str | None = None


class CalendarConnectionResponse(BaseModel):
    connected: bool
    email: str | None = None
    google_id: str | None = None


def _istnow() -> datetime:
    # DB DateTime columns in this project are stored as naive local timestamps (IST).
    return now_ist_naive()


def _to_ist_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return to_ist_naive(value)


def _cookie_max_age_seconds(days: int) -> int:
    return int(days * 24 * 60 * 60)


def _serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar": user.avatar_url,
    }


def _issue_tokens(db: Session, user: User) -> tuple[str, str]:
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(refresh_token),
            expires_at=_istnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRY_DAYS),
        )
    )
    db.commit()
    return access_token, refresh_token


def _set_refresh_cookie(response: JSONResponse, refresh_token: str) -> None:
    same_site = settings.COOKIE_SAMESITE if settings.COOKIE_SAMESITE in {"lax", "strict", "none"} else "lax"
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=same_site,
        max_age=_cookie_max_age_seconds(settings.REFRESH_TOKEN_EXPIRY_DAYS),
        path="/",
    )


def _revoke_refresh_token(db: Session, refresh_token: str | None) -> None:
    token = str(refresh_token or "").strip()
    if not token:
        return

    token_hash = hash_refresh_token(token)
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    if row is None:
        return

    row.revoked_at = _istnow()
    db.commit()


def _google_identity_from_access_token(access_token: str) -> tuple[str, str]:
    token = str(access_token or "").strip()
    if not token:
        return "", ""

    params = urllib.parse.urlencode({"access_token": token})
    url = f"{settings.GOOGLE_TOKENINFO_ENDPOINT}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return "", ""
    except Exception:
        return "", ""

    if not isinstance(payload, dict):
        return "", ""

    google_id = str(payload.get("sub") or payload.get("user_id") or "").strip()
    email = str(payload.get("email") or "").strip().lower()
    return google_id, email


def _resolve_calendar_redirect_uri(explicit_redirect_uri: str | None = None) -> str:
    direct = str(explicit_redirect_uri or "").strip()
    if direct:
        return direct

    configured = str(settings.GOOGLE_CALENDAR_REDIRECT_URI or "").strip()
    if configured:
        return configured

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Calendar redirect URI is required",
    )


def _load_calendar_credential_row(db: Session, user_id: str) -> GoogleCalendarCredential | None:
    return db.scalar(select(GoogleCalendarCredential).where(GoogleCalendarCredential.user_id == user_id))


@router.post(
    "/google",
    response_model=AuthResponse,
    summary="Sign in with Google",
    description=(
        "Validates a Google ID token, creates or updates the local user record, "
        "returns an access token, and sets a secure HTTP-only refresh token cookie."
    ),
    response_description="Authenticated user profile and short-lived access token.",
)
def auth_google(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    claims = verify_google_id_token(payload.id_token)
    google_id = claims.get("google_id")
    email = claims.get("email")

    if not google_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google token missing identity claims")

    user = db.scalar(select(User).where(User.google_id == google_id))
    if user is None:
        user = db.scalar(select(User).where(User.email == email))

    if user is None:
        user = User(
            google_id=google_id,
            email=email,
            name=claims.get("name") or email.split("@")[0],
            avatar_url=claims.get("avatar") or "",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.google_id = google_id
        user.email = email
        user.name = claims.get("name") or user.name
        user.avatar_url = claims.get("avatar") or user.avatar_url
        db.commit()
        db.refresh(user)

    access_token, refresh_token = _issue_tokens(db, user)

    response = JSONResponse(content={"access_token": access_token, "user": _serialize_user(user)})
    _set_refresh_cookie(response, refresh_token)
    return response


@router.post(
    "/refresh",
    response_model=AuthResponse,
    summary="Refresh access token",
    description=(
        "Reads the refresh token from the configured cookie, validates it, rotates it, "
        "and returns a fresh access token plus a new refresh cookie."
    ),
    response_description="New authenticated session payload.",
)
def refresh_auth(request: Request, db: Session = Depends(get_db)):
    refresh_token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")

    payload = decode_token(refresh_token, expected_type="refresh")
    user_id = str(payload.get("sub"))

    token_row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
    if token_row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is invalid")

    if token_row.revoked_at is not None or _to_ist_naive(token_row.expires_at) <= _istnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is invalid")

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    token_row.revoked_at = _istnow()
    db.commit()

    access_token, new_refresh_token = _issue_tokens(db, user)

    response = JSONResponse(content={"access_token": access_token, "user": _serialize_user(user)})
    _set_refresh_cookie(response, new_refresh_token)
    return response


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout current session",
    description="Revokes the refresh token tied to the current cookie and clears the cookie from the client.",
    response_description="Logout confirmation payload.",
)
def logout(request: Request, db: Session = Depends(get_db)):
    _revoke_refresh_token(db, request.cookies.get(settings.REFRESH_COOKIE_NAME))

    response = JSONResponse(content={"success": True})
    response.delete_cookie(settings.REFRESH_COOKIE_NAME, path="/")
    return response


@router.get(
    "/google/calendar/authorization-url",
    response_model=CalendarOAuthUrlResponse,
    summary="Get Google Calendar OAuth URL",
)
def get_google_calendar_authorization_url(
    redirect_uri: str | None = None,
    current_user: User = Depends(get_current_user),
):
    del current_user

    try:
        from google_auth_oauthlib.flow import Flow
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth libraries are not available",
        ) from exc

    resolved_redirect_uri = _resolve_calendar_redirect_uri(redirect_uri)
    flow = Flow.from_client_secrets_file(settings.GOOGLE_OAUTH_CREDENTIALS_PATH, scopes=SCOPES)
    flow.redirect_uri = resolved_redirect_uri
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return {"authorization_url": authorization_url}


@router.post(
    "/google/calendar/connect",
    response_model=CalendarConnectionResponse,
    summary="Connect Google Calendar",
)
def connect_google_calendar(
    payload: CalendarConnectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        from google_auth_oauthlib.flow import Flow
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth libraries are not available",
        ) from exc

    resolved_redirect_uri = _resolve_calendar_redirect_uri(payload.redirect_uri)
    flow = Flow.from_client_secrets_file(settings.GOOGLE_OAUTH_CREDENTIALS_PATH, scopes=SCOPES)
    flow.redirect_uri = resolved_redirect_uri

    try:
        flow.fetch_token(code=payload.authorization_code)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google Calendar authorization failed: {exc}",
        ) from exc

    credentials_payload = dict(json.loads(flow.credentials.to_json()) or {})
    google_id, email = _google_identity_from_access_token(flow.credentials.token)

    encrypted = encrypt_calendar_credentials(credentials_payload)
    row = _load_calendar_credential_row(db, current_user.id)
    if row is None:
        row = GoogleCalendarCredential(
            user_id=current_user.id,
            encrypted_credentials=encrypted,
            scopes_json=list(credentials_payload.get("scopes") or []),
            google_account_id=google_id,
            google_account_email=email,
        )
        db.add(row)
    else:
        row.encrypted_credentials = encrypted
        row.scopes_json = list(credentials_payload.get("scopes") or [])
        row.google_account_id = google_id
        row.google_account_email = email

    db.commit()

    return {
        "connected": True,
        "email": email or None,
        "google_id": google_id or None,
    }


@router.get(
    "/google/calendar/status",
    response_model=CalendarConnectionResponse,
    summary="Get Google Calendar connection status",
)
def get_google_calendar_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = _load_calendar_credential_row(db, current_user.id)
    if row is None:
        return {"connected": False, "email": None, "google_id": None}

    # Defensive read ensures encrypted payload is valid.
    try:
        _ = decrypt_calendar_credentials(row.encrypted_credentials)
    except Exception:
        return {"connected": False, "email": None, "google_id": None}

    return {
        "connected": True,
        "email": row.google_account_email or None,
        "google_id": row.google_account_id or None,
    }


@router.delete(
    "/google/calendar/disconnect",
    response_model=CalendarConnectionResponse,
    summary="Disconnect Google Calendar",
)
def disconnect_google_calendar(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = _load_calendar_credential_row(db, current_user.id)
    if row is not None:
        db.delete(row)
        db.commit()

    return {"connected": False, "email": None, "google_id": None}
