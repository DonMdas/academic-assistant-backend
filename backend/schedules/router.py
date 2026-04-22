from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.db.models import Schedule, User
from backend.db.session import get_db
from backend.schedules.service import (
    build_schedule_detail,
    get_schedule_or_404,
    list_user_schedules,
    serialize_schedule,
)


router = APIRouter(prefix="/schedules", tags=["schedules"])


class ScheduleCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255, description="Human-readable schedule title.")
    description: str = Field(default="", description="Optional notes or context for this schedule.")


class SchedulePatchRequest(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated schedule title.",
    )
    description: str | None = Field(default=None, description="Updated schedule description.")


@router.get(
    "",
    summary="List schedules",
    description="Returns schedules owned by the authenticated user, ordered by most recently updated.",
    response_description="Array of user schedule records.",
)
def list_schedules(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_user_schedules(db, current_user.id)


@router.post(
    "",
    summary="Create schedule",
    description="Creates a new active schedule for the current user.",
    response_description="Created schedule record.",
)
def create_schedule(
    payload: ScheduleCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = Schedule(
        user_id=current_user.id,
        name=payload.name.strip(),
        description=(payload.description or "").strip(),
        status="active",
    )
    db.add(schedule)
    db.commit()
    db.refresh(schedule)
    return serialize_schedule(schedule)


@router.get(
    "/{schedule_id}",
    summary="Get schedule detail",
    description=(
        "Returns one schedule with enriched data, including document ingestion summaries "
        "and the latest plan snapshot if present."
    ),
    response_description="Detailed schedule payload.",
)
def get_schedule(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)
    return build_schedule_detail(db, schedule)


@router.patch(
    "/{schedule_id}",
    summary="Update schedule",
    description="Applies partial updates to schedule name and/or description.",
    response_description="Updated schedule record.",
)
def patch_schedule(
    schedule_id: str,
    payload: SchedulePatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)

    if payload.name is not None:
        schedule.name = payload.name.strip()
    if payload.description is not None:
        schedule.description = payload.description.strip()

    db.commit()
    db.refresh(schedule)
    return serialize_schedule(schedule)


@router.delete(
    "/{schedule_id}",
    summary="Archive schedule",
    description="Soft-deletes a schedule by switching its status to archived.",
    response_description="Archive status payload.",
)
def delete_schedule(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    schedule = get_schedule_or_404(db, current_user.id, schedule_id)
    schedule.status = "archived"
    db.commit()
    db.refresh(schedule)
    return {"id": schedule.id, "status": schedule.status}
