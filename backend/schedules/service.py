from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Document, Schedule, StudyPlan


def serialize_schedule(schedule: Schedule) -> dict:
    return {
        "id": schedule.id,
        "name": schedule.name,
        "description": schedule.description,
        "status": schedule.status,
        "index_path": schedule.index_path,
        "created_at": schedule.created_at.isoformat() if schedule.created_at else None,
        "updated_at": schedule.updated_at.isoformat() if schedule.updated_at else None,
    }


def get_schedule_or_404(db: Session, user_id: str, schedule_id: str) -> Schedule:
    schedule = db.scalar(
        select(Schedule).where(
            Schedule.id == schedule_id,
            Schedule.user_id == user_id,
        )
    )
    if schedule is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
    return schedule


def list_user_schedules(db: Session, user_id: str) -> list[dict]:
    rows = db.scalars(
        select(Schedule)
        .where(Schedule.user_id == user_id)
        .order_by(Schedule.updated_at.desc())
    ).all()
    return [serialize_schedule(row) for row in rows]


def build_schedule_detail(db: Session, schedule: Schedule) -> dict:
    documents = db.scalars(
        select(Document)
        .where(Document.schedule_id == schedule.id)
        .order_by(Document.created_at.desc())
    ).all()

    latest_plan = db.scalar(
        select(StudyPlan)
        .where(StudyPlan.schedule_id == schedule.id)
        .order_by(StudyPlan.updated_at.desc())
    )

    detail = serialize_schedule(schedule)
    detail["documents"] = [
        {
            "id": doc.id,
            "filename": doc.filename,
            "ingest_status": doc.ingest_status,
            "strategy": doc.strategy,
            "doc_type": doc.doc_type,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in documents
    ]

    if latest_plan is None:
        detail["active_plan_summary"] = None
    else:
        detail["active_plan_summary"] = {
            "id": latest_plan.id,
            "status": latest_plan.status,
            "session_count": len(list(latest_plan.sessions_payload or [])),
            "updated_at": latest_plan.updated_at.isoformat() if latest_plan.updated_at else None,
        }

    return detail
