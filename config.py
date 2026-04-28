# config.py

import os

from dotenv import load_dotenv


load_dotenv()

DIRECT_TO_GEMMA_CHAR_LIMIT = 50000


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)

    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _env_text(name, default=""):
    value = os.getenv(name)
    if value is None:
        return str(default)
    return str(value).strip()

def _normalize_ollama_base_url(value: str | None) -> str:
    default_url = "http://127.0.0.1:11434"

    if not value:
        return default_url

    value = value.strip().rstrip("/")

    if value.startswith("http://https://"):
        value = value[len("http://"):]
    elif value.startswith("https://http://"):
        value = value[len("https://"):]

    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"

    return value


# Default Ollama host is local; override with OLLAMA_BASE_URL for remote devices.
OLLAMA_BASE_URL = _normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL"))
_raw_backup_ollama_url = os.getenv("OLLAMA_BACKUP_BASE_URL")
OLLAMA_BACKUP_BASE_URL = (
    _normalize_ollama_base_url(_raw_backup_ollama_url)
    if _raw_backup_ollama_url
    else ""
)

# Planner backend for Gemma orchestration: "gemini" or "ollama".
GEMMA_BACKEND = _env_text("GEMMA_BACKEND", "gemini").lower() or "gemini"

# Default Gemma model names per backend.
GEMMA_GEMINI_MODEL = _env_text("GEMMA_GEMINI_MODEL", "gemma-4-26b-a4b-it")
GEMMA_OLLAMA_MODEL = _env_text("GEMMA_OLLAMA_MODEL", "gemma4:e4b")
QWEN_MODEL = _env_text("QWEN_MODEL", "qwen2.5:7b")

# Disable reasoning/thinking mode for all Ollama models by default.
DISABLE_OLLAMA_THINKING = _env_flag("DISABLE_OLLAMA_THINKING", True)

# Keep legacy segmented chunking disabled unless explicitly enabled.
USE_LEGACY_SEGMENT_CHUNKING = _env_flag("USE_LEGACY_SEGMENT_CHUNKING", False)

FILTER_THRESHOLDS = {
    "slides": {"min_chars": 20, "min_lines": 2},
    "textbook": {"min_chars": 150, "min_lines": 6},
    "notes": {"min_chars": 80, "min_lines": 3},
}

WINDOW_CONFIG = {
    "slides": {"window_size": 6, "overlap": 2},
    "textbook": {"window_size": 3, "overlap": 1},
    "notes": {"window_size": 4, "overlap": 1},
}