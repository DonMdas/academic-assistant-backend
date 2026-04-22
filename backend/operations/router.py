import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.operations.logs import operation_logs
from backend.sse.utils import format_sse


router = APIRouter(prefix="/operations", tags=["operations"])


@router.get(
    "/{operation_id}/logs",
    summary="Get operation logs",
    description=(
        "Returns incremental operation logs for a user-scoped long-running action. "
        "Use offset-based polling from frontend clients."
    ),
    response_description="Operation log snapshot with entries and next offset.",
)
def get_operation_logs(
    operation_id: str,
    offset: int = Query(0, ge=0, description="Log offset to resume polling from."),
    limit: int = Query(200, ge=1, le=500, description="Maximum log entries to return."),
    current_user: User = Depends(get_current_user),
):
    payload = operation_logs.get_logs(operation_id=operation_id, user_id=current_user.id, offset=offset, limit=limit)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    return payload


@router.get(
    "/{operation_id}/stream",
    summary="Stream operation logs (SSE)",
    description=(
        "Streams live operation logs using Server-Sent Events. "
        "Useful for showing real-time planner progress in frontend UIs."
    ),
    response_description="SSE stream of operation log entries and terminal done event.",
)
async def stream_operation_logs(
    operation_id: str,
    offset: int = Query(0, ge=0, description="Log offset to resume stream from."),
    current_user: User = Depends(get_current_user),
):
    user_id = str(current_user.id)
    initial = operation_logs.get_logs(operation_id=operation_id, user_id=user_id, offset=offset, limit=1)
    if initial is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")

    async def _event_stream():
        next_offset = max(0, int(offset or 0))
        yield format_sse(
            "state",
            {
                "operation_id": operation_id,
                "status": initial.get("status", "running"),
                "next_offset": next_offset,
                "done": bool(initial.get("done", False)),
            },
        )

        while True:
            payload = operation_logs.get_logs(
                operation_id=operation_id,
                user_id=user_id,
                offset=next_offset,
                limit=200,
            )
            if payload is None:
                yield format_sse("error", {"message": "operation_not_found"})
                break

            entries = list(payload.get("entries") or [])
            for row in entries:
                yield format_sse("log", row)

            next_offset = int(payload.get("next_offset") or next_offset)
            done = bool(payload.get("done", False))

            yield format_sse(
                "state",
                {
                    "operation_id": operation_id,
                    "status": payload.get("status", "running"),
                    "next_offset": next_offset,
                    "done": done,
                    "total_entries": int(payload.get("total_entries") or 0),
                },
            )

            if done:
                yield format_sse(
                    "done",
                    {
                        "operation_id": operation_id,
                        "status": payload.get("status", "done"),
                        "next_offset": next_offset,
                        "total_entries": int(payload.get("total_entries") or 0),
                    },
                )
                break

            await asyncio.sleep(0.75)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)
