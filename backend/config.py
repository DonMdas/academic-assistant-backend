import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "*") -> list[str]:
    raw = _env_text(name, default)
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _env_required(name: str) -> str:
    raw = os.getenv(name)
    value = str(raw or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class Settings:
    APP_NAME = _env_text("APP_NAME", "Academic Assistant API")
    APP_VERSION = _env_text("APP_VERSION", "0.1.0")

    APP_DATABASE_URL = _env_text("APP_DATABASE_URL", "sqlite:///./backend_app.db")
    LEGACY_DB_PATH = _env_text("LEGACY_DB_PATH", "documents.db")

    UPLOAD_ROOT = _env_text("UPLOAD_ROOT", "uploads")
    SCHEDULE_INDEX_DIR_TEMPLATE = _env_text(
        "SCHEDULE_INDEX_DIR_TEMPLATE",
        "indexes/schedules/{schedule_id}",
    )

    JWT_SECRET = _env_required("JWT_SECRET")
    JWT_ALGORITHM = _env_text("JWT_ALGORITHM", "HS256")
    JWT_EXPIRY_MINUTES = max(5, _env_int("JWT_EXPIRY_MINUTES", 60))
    REFRESH_TOKEN_EXPIRY_DAYS = max(1, _env_int("REFRESH_TOKEN_EXPIRY_DAYS", 7))
    REFRESH_COOKIE_NAME = _env_text("REFRESH_COOKIE_NAME", "refresh_token")
    COOKIE_SECURE = _env_flag("COOKIE_SECURE", False)

    GOOGLE_CLIENT_ID = _env_text("GOOGLE_CLIENT_ID", "")
    GOOGLE_OAUTH_CREDENTIALS_PATH = _env_text("GOOGLE_OAUTH_CREDENTIALS_PATH", "credentials.json")
    GOOGLE_CALENDAR_REDIRECT_URI = _env_text("GOOGLE_CALENDAR_REDIRECT_URI", "")
    GOOGLE_TOKENINFO_ENDPOINT = _env_text(
        "GOOGLE_TOKENINFO_ENDPOINT",
        "https://oauth2.googleapis.com/tokeninfo",
    )

    SESSION_EMBEDDING_THRESHOLD = _env_float("SESSION_EMBEDDING_THRESHOLD", 0.65)
    SESSION_KEYWORD_OVERLAP_MIN = _env_float("SESSION_KEYWORD_OVERLAP_MIN", 0.30)

    BRIEFING_MAX_TOKENS = max(256, _env_int("BRIEFING_MAX_TOKENS", 2048))
    BRIEFING_MODEL = _env_text("BRIEFING_MODEL", "gemma3:12b")

    CHAT_CONTEXT_TURNS = max(1, _env_int("CHAT_CONTEXT_TURNS", 8))
    ALLOWED_ORIGINS = _env_csv("ALLOWED_ORIGINS", "http://localhost:3000")


settings = Settings()


def schedule_index_dir(schedule_id: str) -> Path:
    relative = settings.SCHEDULE_INDEX_DIR_TEMPLATE.format(schedule_id=schedule_id)
    return Path(relative)


def schedule_merged_index_base(schedule_id: str) -> Path:
    return schedule_index_dir(schedule_id) / "merged"
