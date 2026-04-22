from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Schedule, SessionChatMessage, StudySession
from backend.timezone_utils import now_ist_naive


def _focus_topics_from_session(study_session: StudySession) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()
    for item in list(study_session.focus_chunks_json or []):
        topic = str((item or {}).get("topic") or "").strip()
        if topic and topic not in seen:
            seen.add(topic)
            topics.append(topic)
    return topics


def check_and_auto_start_session(db: Session, user_id: str) -> dict | None:
    # Session date/time checks use backend-local IST to match scheduled slots.
    now = now_ist_naive()
    current_date = now.date()
    current_time = now.time()

    # Find an upcoming session scheduled for right now
    study_session = db.scalar(
        select(StudySession)
        .join(Schedule, Schedule.id == StudySession.schedule_id)
        .where(
            Schedule.user_id == user_id,
            StudySession.scheduled_date == current_date,
            StudySession.start_time <= current_time,
            StudySession.end_time >= current_time,
            StudySession.status == "upcoming"
        )
    )

    if study_session:
        # Re-use your existing logic to mark it active
        return mark_session_started(db, study_session)
    
    return None


def get_current_scheduled_session(db: Session, user_id: str) -> StudySession | None:
    now = now_ist_naive()
    return db.scalar(
        select(StudySession)
        .join(Schedule, Schedule.id == StudySession.schedule_id)
        .where(
            Schedule.user_id == user_id,
            StudySession.scheduled_date == now.date(),
            StudySession.start_time <= now.time(),
            StudySession.end_time >= now.time(),
            StudySession.status == "upcoming"
        )
    )
    
    
def get_user_study_session_or_404(db: Session, user_id: str, session_id: str) -> StudySession:
    study_session = db.scalar(
        select(StudySession)
        .join(Schedule, Schedule.id == StudySession.schedule_id)
        .where(
            StudySession.id == session_id,
            Schedule.user_id == user_id,
        )
    )
    if study_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study session not found")
    return study_session


def serialize_study_session(study_session: StudySession) -> dict:
    return {
        "id": study_session.id,
        "plan_id": study_session.plan_id,
        "schedule_id": study_session.schedule_id,
        "session_number": study_session.session_number,
        "title": study_session.title,
        "scheduled_date": study_session.scheduled_date.isoformat() if study_session.scheduled_date else None,
        "start_time": study_session.start_time.isoformat() if study_session.start_time else None,
        "end_time": study_session.end_time.isoformat() if study_session.end_time else None,
        "focus_topics": _focus_topics_from_session(study_session),
        "status": study_session.status,
        "briefing_status": study_session.briefing_status,
    }


def mark_session_started(db: Session, study_session: StudySession) -> dict:
    study_session.status = "active"
    if study_session.briefing_status in {"pending", "failed"}:
        study_session.briefing_status = "generating"
    db.commit()
    db.refresh(study_session)
    return serialize_study_session(study_session)


def mark_session_completed(db: Session, study_session: StudySession) -> dict:
    study_session.status = "completed"
    db.commit()
    db.refresh(study_session)
    return serialize_study_session(study_session)


def save_session_chat_turn(
    db: Session,
    study_session: StudySession,
    user_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    retrieval_path: str,
) -> None:
    db.add(
        SessionChatMessage(
            session_id=study_session.id,
            user_id=user_id,
            role="user",
            content=question,
            sources_json=[],
            retrieval_path=retrieval_path,
        )
    )
    db.add(
        SessionChatMessage(
            session_id=study_session.id,
            user_id=user_id,
            role="assistant",
            content=answer,
            sources_json=sources,
            retrieval_path=retrieval_path,
        )
    )
    db.commit()


def list_session_chat_history(
    db: Session,
    session_id: str,
    user_id: str,
    limit: int,
    offset: int,
) -> list[dict]:
    rows = db.scalars(
        select(SessionChatMessage)
        .where(
            SessionChatMessage.session_id == session_id,
            SessionChatMessage.user_id == user_id,
        )
        .order_by(SessionChatMessage.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()

    return [
        {
            "id": row.id,
            "session_id": row.session_id,
            "role": row.role,
            "content": row.content,
            "sources": row.sources_json or [],
            "retrieval_path": row.retrieval_path,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def build_sidebar(db: Session, study_session: StudySession) -> dict:
    upcoming = db.scalars(
        select(StudySession)
        .where(
            StudySession.schedule_id == study_session.schedule_id,
            StudySession.session_number > study_session.session_number,
        )
        .order_by(StudySession.session_number.asc())
        .limit(3)
    ).all()

    return {
        "prerequisites": list(study_session.prerequisites_json or []),
        "upcoming_sessions": [
            {
                "id": row.id,
                "session_number": row.session_number,
                "title": row.title,
                "scheduled_date": row.scheduled_date.isoformat() if row.scheduled_date else None,
                "status": row.status,
            }
            for row in upcoming
        ],
    }
