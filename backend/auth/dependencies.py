from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.auth.jwt import decode_token
from backend.db.models import User
from backend.db.session import get_db


http_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    token: str | None = Query(default=None, min_length=1),
    db: Session = Depends(get_db),
) -> User:
    access_token = str(credentials.credentials).strip() if credentials is not None else ""
    if not access_token:
        access_token = str(token or "").strip()

    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token is required")

    payload = decode_token(access_token, expected_type="access")
    user = db.get(User, str(payload.get("sub")))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
