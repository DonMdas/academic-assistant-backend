import base64
import hashlib
import json

from cryptography.fernet import Fernet

from backend.config import settings


def _fernet() -> Fernet:
    seed = str(settings.JWT_SECRET).encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(seed).digest())
    return Fernet(key)


def encrypt_calendar_credentials(payload: dict) -> str:
    serialized = json.dumps(dict(payload or {}), separators=(",", ":")).encode("utf-8")
    return _fernet().encrypt(serialized).decode("utf-8")


def decrypt_calendar_credentials(ciphertext: str) -> dict:
    token = str(ciphertext or "").strip()
    if not token:
        return {}

    raw = _fernet().decrypt(token.encode("utf-8"))
    data = json.loads(raw.decode("utf-8"))
    return dict(data or {})
