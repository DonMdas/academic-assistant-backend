import os
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.config import schedule_merged_index_base
from backend.db.models import ChatMessage, Chunk, Schedule
from backend.documents.service import rebuild_schedule_index_from_chunks
from backend.llm import generate_answer_with_chunks


def serialize_chat_message(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "schedule_id": message.schedule_id,
        "user_id": message.user_id,
        "role": message.role,
        "content": message.content,
        "sources": message.sources_json or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _index_exists(index_base: str) -> bool:
    return os.path.exists(f"{index_base}.index") and os.path.exists(f"{index_base}.pkl")


def resolve_schedule_index_path(db: Session, schedule: Schedule) -> str:
    candidates = []

    if schedule.index_path:
        candidates.append(schedule.index_path)

    candidates.append(str(schedule_merged_index_base(schedule.id)))
    candidates.append(f"indexes/sessions/{schedule.id}")

    for candidate in candidates:
        if candidate and _index_exists(candidate):
            schedule.index_path = candidate
            db.commit()
            return candidate

    rebuilt = rebuild_schedule_index_from_chunks(db, schedule)
    schedule.index_path = rebuilt
    db.commit()

    if rebuilt and _index_exists(rebuilt):
        return rebuilt

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No schedule index found. Ingest at least one document first.",
    )


def _format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        retrieval = chunk.get("retrieval", {}) if isinstance(chunk, dict) else {}
        sources.append(
            {
                "chunk_id": metadata.get("chunk_id") or metadata.get("legacy_chunk_id"),
                "source_doc_id": chunk.get("source_doc_id") or metadata.get("source_doc_id"),
                "topic": metadata.get("topic"),
                "pages": metadata.get("pages", []),
                "score": retrieval.get("score"),
                "retrieval": retrieval,
            }
        )
    return sources


def retrieve_schedule_chunks(
    db: Session,
    schedule: Schedule,
    message: str,
    mode: str,
    max_minutes: int | None = None,
) -> tuple[list[dict], list[dict], str]:
    index_path = resolve_schedule_index_path(db, schedule)

    try:
        from rag_engine import RAGEngine

        rag = RAGEngine(index_path=index_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load RAG index: {exc}",
        ) from exc

    mode_value = str(mode or "qa").strip().lower()

    if mode_value == "plan":
        chunks = rag.retrieve_hybrid(
            message,
            top_k=8,
            rerank_with_qwen=True,
            rerank_top_n=6,
        )
    elif mode_value == "beginner":
        chunks = rag.retrieve(message, filters={"complexity": "beginner"})
    elif mode_value == "time_budget":
        budget = int(max_minutes or 120)
        chunks = rag.retrieve_with_time_budget(message, max_total_time=max(15, budget))
    else:
        chunks = rag.retrieve_hybrid(
            message,
            top_k=5,
            rerank_with_qwen=True,
            rerank_top_n=5,
        )

    sources = _format_sources(chunks)
    return chunks, sources, index_path


def answer_schedule_message(message: str, chunks: list[dict], mode: str) -> str:
    mode_value = "plan" if str(mode).lower() in {"plan", "time_budget"} else "qa"
    return generate_answer_with_chunks(message, chunks, mode=mode_value)


def save_schedule_chat_turn(
    db: Session,
    schedule_id: str,
    user_id: str,
    message: str,
    answer: str,
    sources: list[dict],
) -> None:
    db.add(
        ChatMessage(
            schedule_id=schedule_id,
            user_id=user_id,
            role="user",
            content=message,
            sources_json=[],
        )
    )
    db.add(
        ChatMessage(
            schedule_id=schedule_id,
            user_id=user_id,
            role="assistant",
            content=answer,
            sources_json=sources,
        )
    )
    db.commit()


def get_schedule_chat_history(
    db: Session,
    schedule_id: str,
    user_id: str,
    limit: int,
    offset: int,
) -> list[dict]:
    rows = db.scalars(
        select(ChatMessage)
        .where(
            ChatMessage.schedule_id == schedule_id,
            ChatMessage.user_id == user_id,
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()

    return [serialize_chat_message(row) for row in rows]


def clear_schedule_chat_history(db: Session, schedule_id: str, user_id: str) -> int:
    result = db.execute(
        delete(ChatMessage).where(
            ChatMessage.schedule_id == schedule_id,
            ChatMessage.user_id == user_id,
        )
    )
    db.commit()
    return int(result.rowcount or 0)


def schedule_has_chunks(db: Session, schedule_id: str) -> bool:
    row = db.scalar(select(Chunk.id).where(Chunk.schedule_id == schedule_id).limit(1))
    return row is not None
