import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.db.session import get_db
from backend.llm import generate_answer_with_chunks, generate_text
from backend.operations.logs import operation_logs
from backend.sessions.briefing import (
    briefing_stream_id,
    run_briefing_background,
)
from backend.sessions.retrieval import two_stage_retrieve
from backend.sessions.service import (
    build_sidebar,
    get_user_study_session_or_404,
    list_session_chat_history,
    mark_session_completed,
    mark_session_started,
    save_session_chat_turn,
)
from backend.sse.utils import format_sse, sse_manager
from backend.sessions.service import get_current_scheduled_session  

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User message for the active study session.")
    history: list[dict] = Field(default_factory=list, description="Optional previous turns supplied by client.")


def _stream_chunks(text: str, size: int = 140) -> list[str]:
    value = str(text or "")
    if not value:
        return [""]

    chunks = []
    index = 0
    while index < len(value):
        chunks.append(value[index : index + size])
        index += size
    return chunks


def _build_local_session_prompt(study_session_title: str, question: str, chunks: list[dict]) -> str:
    context = []
    for item in chunks:
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        topic = str(meta.get("topic") or "General").strip()
        focus_points = list(meta.get("focus_points") or [])
        focus_label = ", ".join(focus_points) if focus_points else "general focus"
        content = str(item.get("text") or "").strip()
        context.append(f"Topic: {topic}\nFocus points: {focus_label}\nContent: {content}")

    body = "\n\n".join(context)
    return f"""
You are helping during a focused study session on: {study_session_title}.
Use only the material below unless absolutely necessary.
Be concise, clear, and aligned to this session scope.

Question:
{question}

Session material:
{body}
""".strip()


@router.post(
    "/{session_id}/start",
    summary="Start study session",
    description=(
        "Marks a materialized study session as active, initializes its briefing stream, "
        "and schedules background briefing generation."
    ),
    response_description="Updated session state and briefing stream URL.",
)
def start_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="session_briefing",
        message="Session start requested",
        metadata={"session_id": session_id},
    )

    study_session = get_user_study_session_or_404(db, current_user.id, session_id)
    operation_logs.append(operation_id, "Session ownership validated")
    payload = mark_session_started(db, study_session)
    operation_logs.append(operation_id, "Session marked active")

    stream_id = briefing_stream_id(session_id)
    sse_manager.create_or_reset(stream_id)
    background_tasks.add_task(run_briefing_background, session_id, operation_id, current_user.id)
    operation_logs.append(operation_id, "Briefing background task queued")

    payload["briefing_stream_url"] = f"/sessions/{session_id}/briefing/stream"
    payload["operation_id"] = operation_id
    return payload


@router.get(
    "/{session_id}/briefing/stream",
    summary="Stream session briefing",
    description=(
        "Streams briefing generation events for the session via Server-Sent Events. "
        "Emits status/delta/done events and can replay stored briefing text when available."
    ),
    response_description="SSE stream for briefing generation and completion.",
)
def stream_briefing(
    session_id: str,
    operation_id: str | None = Query(default=None, description="Optional operation ID from start session response."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_session = get_user_study_session_or_404(db, current_user.id, session_id)
    stream_id = briefing_stream_id(session_id)

    async def event_stream():
        if operation_id:
            yield format_sse("operation", {"operation_id": operation_id})

        if sse_manager.exists(stream_id):
            async for row in sse_manager.stream(stream_id):
                yield row
            return

        if study_session.generated_briefing:
            for part in _stream_chunks(study_session.generated_briefing, size=260):
                yield format_sse("delta", {"text": part})
                await asyncio.sleep(0)
            yield format_sse("done", {"status": study_session.briefing_status or "done"})
            return

        yield format_sse("status", {"status": study_session.briefing_status})
        yield format_sse("done", {"status": study_session.briefing_status})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/{session_id}/chat",
    summary="Session-focused chat (SSE)",
    description=(
        "Runs two-stage retrieval (session-local first, schedule fallback), generates an answer, "
        "persists both turns, and streams the response as SSE."
    ),
    response_description="SSE stream containing sources, answer chunks, and completion event.",
)
def session_chat(
    session_id: str,
    payload: SessionChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="session_chat",
        message="Session chat request started",
        metadata={"session_id": session_id},
    )

    study_session = get_user_study_session_or_404(db, current_user.id, session_id)
    operation_logs.append(operation_id, "Session ownership validated")

    try:
        retrieval = two_stage_retrieve(db, study_session, payload.message)

        chunks = list(retrieval.get("chunks") or [])
        sources = list(retrieval.get("sources") or [])
        retrieval_path = str(retrieval.get("retrieval_path") or "rag_fallback")
        operation_logs.append(
            operation_id,
            "Retrieval completed",
            metadata={
                "retrieval_path": retrieval_path,
                "source_count": len(sources),
                "chunk_count": len(chunks),
            },
        )

        if retrieval_path == "local_chunk":
            prompt = _build_local_session_prompt(study_session.title, payload.message, chunks)
            answer = generate_text(prompt, max_tokens=900, temperature=0.15)
        else:
            answer = generate_answer_with_chunks(payload.message, chunks, mode="qa")
        operation_logs.append(operation_id, "Generated session answer", metadata={"answer_chars": len(answer)})

        save_session_chat_turn(
            db=db,
            study_session=study_session,
            user_id=current_user.id,
            question=payload.message,
            answer=answer,
            sources=sources,
            retrieval_path=retrieval_path,
        )
    except Exception as exc:
        operation_logs.fail(operation_id, f"Session chat failed: {exc}")
        raise

    async def event_stream():
        try:
            yield format_sse("operation", {"operation_id": operation_id})
            yield format_sse(
                "sources",
                {
                    "sources": sources,
                    "retrieval_path": retrieval_path,
                    "max_similarity": retrieval.get("max_similarity"),
                    "keyword_overlap": retrieval.get("keyword_overlap"),
                },
            )
            for piece in _stream_chunks(answer):
                yield format_sse("delta", {"text": piece})
                await asyncio.sleep(0)
            yield format_sse("done", {"text": answer, "retrieval_path": retrieval_path, "operation_id": operation_id})
            operation_logs.succeed(operation_id, "Session chat stream completed")
        except Exception as exc:
            operation_logs.fail(operation_id, f"Session chat streaming failed: {exc}")
            yield format_sse("error", {"message": str(exc), "operation_id": operation_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get(
    "/{session_id}/chat/history",
    summary="Get session chat history",
    description="Returns paginated chat history for one study session.",
    response_description="Paginated array of session chat messages.",
)
def session_chat_history(
    session_id: str,
    limit: int = Query(30, ge=1, le=200, description="Maximum messages to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_user_study_session_or_404(db, current_user.id, session_id)
    return list_session_chat_history(db, session_id, current_user.id, limit, offset)


@router.post(
    "/{session_id}/complete",
    summary="Complete study session",
    description="Marks the session as completed.",
    response_description="Updated session state payload.",
)
def complete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_session = get_user_study_session_or_404(db, current_user.id, session_id)
    return mark_session_completed(db, study_session)


@router.get(
    "/{session_id}/sidebar",
    summary="Get session sidebar context",
    description="Returns prerequisites and a preview of upcoming sessions for sidebar UI panels.",
    response_description="Session sidebar payload.",
)
def session_sidebar(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_session = get_user_study_session_or_404(db, current_user.id, session_id)
    return build_sidebar(db, study_session)



@router.post(
    "/auto-start",
    summary="Check and auto-start current session",
    description="Finds a session scheduled for right now and activates it.",
)
def auto_start_current_session(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Find if there is a session scheduled for right now
    study_session = get_current_scheduled_session(db, current_user.id)
    
    if not study_session:
        return {"active_session": None}

    # 2. Mark it active and start the briefing (reusing existing logic)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="session_auto_start",
        message="Session auto-started on login",
        metadata={"session_id": study_session.id},
    )

    payload = mark_session_started(db, study_session)
    stream_id = briefing_stream_id(study_session.id)
    sse_manager.create_or_reset(stream_id)
    background_tasks.add_task(run_briefing_background, study_session.id, operation_id, current_user.id)

    payload["briefing_stream_url"] = f"/sessions/{study_session.id}/briefing/stream"
    payload["operation_id"] = operation_id
    
    return {"active_session": payload}