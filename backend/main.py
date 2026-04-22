from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth.router import router as auth_router
from backend.chat.router import router as chat_router
from backend.config import settings
from backend.db.session import init_db
from backend.documents.router import router as documents_router
from backend.operations.router import router as operations_router
from backend.plans.router import router as plans_router
from backend.schedules.router import router as schedules_router
from backend.sessions.router import router as sessions_router


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI layer for schedules, ingestion, planning, and RAG-backed study sessions.",
)


if settings.ALLOWED_ORIGINS:
    allow_all = "*" in settings.ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else settings.ALLOWED_ORIGINS,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get(
    "/healthz",
    tags=["system"],
    summary="Health check",
    description="Liveness probe endpoint used by clients and deployment platforms to verify the API is running.",
    response_description="Basic service status payload.",
)
def health() -> dict:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(operations_router)
app.include_router(schedules_router)
app.include_router(documents_router)
app.include_router(plans_router)
app.include_router(chat_router)
app.include_router(sessions_router)
