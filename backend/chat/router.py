import asyncio

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.chat.service import (
    answer_schedule_message,
    clear_schedule_chat_history,
    get_schedule_chat_history,
    retrieve_schedule_chunks,
    save_schedule_chat_turn,
)
from backend.db.models import User
from backend.db.session import get_db
from backend.operations.logs import operation_logs
from backend.schedules.service import get_schedule_or_404
from backend.sse.utils import format_sse


router = APIRouter(prefix="/schedules/{schedule_id}/chat", tags=["chat"])


class ScheduleChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User question or prompt.")
    mode: str = Field(default="qa", description="Retrieval mode: qa, plan, beginner, or time_budget.")
    history: list[dict] = Field(default_factory=list, description="Optional prior turns from the client.")
    max_minutes: int | None = Field(default=None, description="Time budget hint for time_budget mode.")


def _chunks_for_stream(text: str, size: int = 140) -> list[str]:
    value = str(text or "")
    if not value:
        return [""]

    out = []
    cursor = 0
    while cursor < len(value):
        out.append(value[cursor : cursor + size])
        cursor += size
    return out


@router.post(
    "",
    summary="Schedule chat (SSE)",
    description=(
        "Retrieves schedule-level context, generates an answer, persists both user and assistant turns, "
        "and streams response chunks as Server-Sent Events."
    ),
    response_description="SSE stream with sources, delta, and done events.",
)
def schedule_chat(
    schedule_id: str,
    payload: ScheduleChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)

    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="schedule_chat",
        message="Schedule chat request started",
        metadata={"schedule_id": schedule.id, "mode": payload.mode},
    )

    try:
        operation_logs.append(operation_id, "Retrieving relevant schedule chunks")
        chunks, sources, _index_path = retrieve_schedule_chunks(
            db=db,
            schedule=schedule,
            message=payload.message,
            mode=payload.mode,
            max_minutes=payload.max_minutes,
        )
        operation_logs.append(
            operation_id,
            "Retrieved schedule chunks",
            metadata={"source_count": len(list(sources or []))},
        )

        answer = answer_schedule_message(payload.message, chunks, payload.mode)
        operation_logs.append(
            operation_id,
            "Generated response text",
            metadata={"answer_chars": len(str(answer or ""))},
        )
        save_schedule_chat_turn(db, schedule.id, current_user.id, payload.message, answer, sources)
    except Exception as exc:
        operation_logs.fail(operation_id, f"Schedule chat failed: {exc}")
        raise

    async def event_stream():
        try:
            yield format_sse("operation", {"operation_id": operation_id})
            yield format_sse("sources", {"sources": sources})
            for piece in _chunks_for_stream(answer):
                yield format_sse("delta", {"text": piece})
                await asyncio.sleep(0)
            yield format_sse("done", {"text": answer, "operation_id": operation_id})
            operation_logs.succeed(operation_id, "Schedule chat stream completed")
        except Exception as exc:
            operation_logs.fail(operation_id, f"Schedule chat streaming failed: {exc}")
            yield format_sse("error", {"message": str(exc), "operation_id": operation_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get(
    "/history",
    summary="Get schedule chat history",
    description="Returns paginated schedule chat history for the current user.",
    response_description="Paginated array of chat messages.",
)
def history(
    schedule_id: str,
    limit: int = Query(30, ge=1, le=200, description="Maximum messages to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return get_schedule_chat_history(db, schedule_id, current_user.id, limit=limit, offset=offset)


@router.delete(
    "/history",
    summary="Clear schedule chat history",
    description="Deletes all schedule chat messages for the current user.",
    response_description="Count of deleted messages.",
)
def clear_history(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    deleted = clear_schedule_chat_history(db, schedule_id, current_user.id)
    return {"deleted": deleted}
