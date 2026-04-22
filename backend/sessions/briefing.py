import asyncio
from datetime import datetime

from backend.config import settings
from backend.db.models import Chunk, StudySession
from backend.db.session import SessionLocal
from backend.llm import generate_text
from backend.operations.logs import operation_logs
from backend.sse.utils import sse_manager
from backend.timezone_utils import today_ist


def briefing_stream_id(session_id: str) -> str:
    return f"briefing:{session_id}"


def _estimate_duration_minutes(study_session: StudySession) -> int:
    if study_session.start_time and study_session.end_time:
        local_date = today_ist()
        start_dt = datetime.combine(local_date, study_session.start_time)
        end_dt = datetime.combine(local_date, study_session.end_time)
        delta = int((end_dt - start_dt).total_seconds() // 60)
        if delta > 0:
            return delta
    return 60


def _build_briefing_prompt(study_session: StudySession, chunk_rows: list[Chunk]) -> str:
    focus_points = []
    for item in list(study_session.focus_chunks_json or []):
        topic = str(item.get("topic") or "").strip()
        points = list(item.get("focus_points") or [])
        if topic:
            focus_points.append(f"{topic}: {', '.join(points) if points else 'general focus'}")

    focus_block = "\n".join(f"- {row}" for row in focus_points) or "- Consolidate core concepts"
    prerequisites = list(study_session.prerequisites_json or [])
    prereq_block = "\n".join(f"- {item}" for item in prerequisites) or "- None"

    ordered_content = []
    for index, chunk in enumerate(chunk_rows, 1):
        topic = str((chunk.metadata_json or {}).get("topic") or f"Chunk {index}").strip()
        text = str(chunk.content or "").strip()
        ordered_content.append(f"[{index}] Topic: {topic}\n{text}")

    material = "\n\n".join(ordered_content)
    duration = _estimate_duration_minutes(study_session)

    return f"""
You are a tutor. Generate a detailed, flowing explanation for this focused study session.

Session title: {study_session.title}
Session number: {study_session.session_number}
Estimated duration (minutes): {duration}

Focus points:
{focus_block}

Prerequisites:
{prereq_block}

Use this structure:
1) Overview
2) Core concepts
3) Worked examples
4) Summary

Study material:
{material}
""".strip()


def _split_stream_parts(text: str, max_chars: int = 280) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return [""]

    out = []
    cursor = 0
    while cursor < len(value):
        out.append(value[cursor : cursor + max_chars])
        cursor += max_chars
    return out


async def generate_session_briefing(
    session_id: str,
    operation_id: str | None = None,
    user_id: str | None = None,
) -> None:
    stream_id = briefing_stream_id(session_id)
    db = SessionLocal()

    try:
        if operation_id and user_id:
            operation_logs.start(
                operation_id=operation_id,
                user_id=user_id,
                kind="session_briefing",
                message="Briefing generation started",
                metadata={"session_id": session_id},
            )

        study_session = db.get(StudySession, session_id)
        if study_session is None:
            await sse_manager.publish(stream_id, "error", {"message": "study_session_not_found"})
            if operation_id:
                operation_logs.fail(operation_id, "Study session not found")
            return

        if operation_id:
            operation_logs.append(operation_id, "Loaded study session context")

        focus_ids = []
        for item in list(study_session.focus_chunks_json or []):
            chunk_id = str(item.get("chunk_id") or "").strip()
            if chunk_id:
                focus_ids.append(chunk_id)

        if focus_ids:
            rows = db.query(Chunk).filter(Chunk.id.in_(focus_ids)).all()
            by_id = {row.id: row for row in rows}
            chunk_rows = [by_id[cid] for cid in focus_ids if cid in by_id]
        else:
            chunk_rows = []

        if operation_id:
            operation_logs.append(operation_id, "Prepared focus chunk context", metadata={"chunk_count": len(chunk_rows)})

        prompt = _build_briefing_prompt(study_session, chunk_rows)
        if operation_id:
            operation_logs.append(operation_id, "Sending briefing prompt to LLM")

        text = generate_text(prompt, max_tokens=settings.BRIEFING_MAX_TOKENS, temperature=0.2)
        if operation_id:
            operation_logs.append(operation_id, "Received briefing text", metadata={"chars": len(text)})

        for part in _split_stream_parts(text):
            await sse_manager.publish(stream_id, "delta", {"text": part})

        study_session.generated_briefing = text
        study_session.briefing_status = "done"
        db.commit()

        await sse_manager.publish(stream_id, "done", {"status": "done"})
        if operation_id:
            operation_logs.succeed(operation_id, "Briefing generation completed")
    except Exception as exc:
        study_session = db.get(StudySession, session_id)
        if study_session is not None:
            study_session.briefing_status = "failed"
            db.commit()
        await sse_manager.publish(stream_id, "error", {"message": str(exc)})
        if operation_id:
            operation_logs.fail(operation_id, f"Briefing generation failed: {exc}")
    finally:
        await sse_manager.close(stream_id)
        db.close()


def run_briefing_background(
    session_id: str,
    operation_id: str | None = None,
    user_id: str | None = None,
) -> None:
    asyncio.run(generate_session_briefing(session_id, operation_id=operation_id, user_id=user_id))
