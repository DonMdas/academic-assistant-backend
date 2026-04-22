import json
import urllib.error
import urllib.parse
import urllib.request

from fastapi import HTTPException, status

from backend.config import settings


def verify_google_id_token(id_token: str) -> dict:
    token = str(id_token or "").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="id_token is required")

    params = urllib.parse.urlencode({"id_token": token})
    url = f"{settings.GOOGLE_TOKENINFO_ENDPOINT}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google token validation failed",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach Google token validation service",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token payload")

    if payload.get("error_description"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(payload.get("error_description")))

    aud = str(payload.get("aud") or "").strip()
    if settings.GOOGLE_CLIENT_ID and aud != settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google token audience mismatch")

    email_verified = str(payload.get("email_verified", "false")).strip().lower()
    if email_verified not in {"true", "1", "yes"}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google account email is not verified")

    return {
        "google_id": str(payload.get("sub") or "").strip(),
        "email": str(payload.get("email") or "").strip().lower(),
        "name": str(payload.get("name") or "").strip(),
        "avatar": str(payload.get("picture") or "").strip(),
    }
