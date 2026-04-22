import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.session import Base
from backend.timezone_utils import now_ist_naive


def _uuid() -> str:
    return str(uuid.uuid4())


def _istnow() -> datetime:
    return now_ist_naive()


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    google_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    avatar_url: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)

    schedules: Mapped[list["Schedule"]] = relationship(back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GoogleCalendarCredential(Base):
    __tablename__ = "google_calendar_credentials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    encrypted_credentials: Mapped[str] = mapped_column(Text)
    scopes_json: Mapped[list] = mapped_column("scopes", JSON, default=list)
    google_account_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    google_account_email: Mapped[str] = mapped_column(String(320), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, onupdate=_istnow)


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    index_path: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, onupdate=_istnow)

    user: Mapped[User] = relationship(back_populates="schedules")
    documents: Mapped[list["Document"]] = relationship(back_populates="schedule")


class Document(Base):
    __tablename__ = "documents_api"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    ingest_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    strategy: Mapped[str] = mapped_column(String(32), default="rag")
    doc_type: Mapped[str] = mapped_column(String(32), default="notes")
    ingest_report: Mapped[dict] = mapped_column(JSON, default=dict)
    pipeline_doc_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    index_path: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)

    schedule: Mapped[Schedule] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks_api"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents_api.id"), index=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    embedding_path: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class StudyPlan(Base):
    __tablename__ = "study_plans_api"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"), index=True)
    sessions_payload: Mapped[list] = mapped_column("sessions", JSON, default=list)
    constraints_json: Mapped[dict] = mapped_column("constraints", JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, onupdate=_istnow)
    

class StudySession(Base):
    __tablename__ = "study_sessions_api"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(ForeignKey("study_plans_api.id"), index=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"), index=True)
    session_number: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str] = mapped_column(String(255), default="Study Session")
    scheduled_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    start_time: Mapped[time | None] = mapped_column(nullable=True)
    end_time: Mapped[time | None] = mapped_column(nullable=True)
    focus_chunks_json: Mapped[list] = mapped_column("focus_chunks", JSON, default=list)
    prerequisites_json: Mapped[list] = mapped_column("prerequisites", JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="upcoming", index=True)
    generated_briefing: Mapped[str] = mapped_column(Text, default="")
    briefing_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, onupdate=_istnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[list] = mapped_column("sources", JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, index=True)


class SessionChatMessage(Base):
    __tablename__ = "session_chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("study_sessions_api.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[list] = mapped_column("sources", JSON, default=list)
    retrieval_path: Mapped[str] = mapped_column(String(32), default="rag_fallback")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_istnow, index=True)
