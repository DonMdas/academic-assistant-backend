import threading

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.db.models import StudyPlan, User
from backend.db.session import SessionLocal, get_db
from backend.operations.logs import operation_logs
from backend.plans.service import (
    confirm_plan,
    delete_all_plans,
    delete_plan,
    generate_draft_plan,
    get_current_plan,
    get_materialized_session_detail,
    list_plans_for_schedule,
    list_materialized_sessions,
    revise_plan_with_feedback,
    sync_plan_sessions_to_calendar,
    unsync_plan_sessions_from_calendar,
)
from backend.schedules.service import get_schedule_or_404


router = APIRouter(prefix="/schedules/{schedule_id}/plan", tags=["plans"])


def _format_planner_log_line(entry: dict, *, include_metadata: bool = False) -> str:
    message = str(dict(entry or {}).get("message") or "").strip() or "(no message)"
    if message.startswith("[Planner]"):
        line = message
    else:
        line = f"[Planner] {message}"

    if not include_metadata:
        return line

    metadata = dict(dict(entry or {}).get("metadata") or {})
    if not metadata:
        return line

    parts: list[str] = []
    for key in sorted(metadata.keys()):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={str(value)}")

    if not parts:
        return line

    return f"{line} | {', '.join(parts[:8])}"


def _run_generate_plan_background(
    *,
    operation_id: str,
    user_id: str,
    schedule_id: str,
    constraints: dict,
    user_feedback: str,
    feedback_history: list[str],
) -> None:
    db = SessionLocal()
    try:
        get_schedule_or_404(db, user_id, schedule_id)
        operation_logs.append(operation_id, "Background plan generation job started")

        plan = generate_draft_plan(
            db,
            schedule_id,
            constraints,
            user_id=user_id,
            operation_id=operation_id,
            user_feedback=user_feedback,
            feedback_history=feedback_history,
        )
        operation_logs.succeed(
            operation_id,
            "Draft plan generated",
            metadata={
                "plan_id": str(plan.get("id") or ""),
                "session_count": int(plan.get("session_count") or 0),
                "needs_user_feedback": bool(dict(plan.get("review") or {}).get("user_feedback_requested", False)),
            },
        )
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan generation failed: {exc}")
    finally:
        db.close()


def _run_get_plan_background(
    *,
    operation_id: str,
    user_id: str,
    schedule_id: str,
) -> None:
    db = SessionLocal()
    try:
        get_schedule_or_404(db, user_id, schedule_id)
        operation_logs.append(operation_id, "Background latest-plan fetch job started")

        plan = get_current_plan(db, schedule_id)
        if plan is None:
            operation_logs.succeed(
                operation_id,
                "No plan found for schedule",
                metadata={"found": False, "schedule_id": schedule_id},
            )
            return

        review = dict(plan.get("review") or {})
        operation_logs.succeed(
            operation_id,
            "Latest plan fetched",
            metadata={
                "found": True,
                "schedule_id": schedule_id,
                "plan_id": str(plan.get("id") or ""),
                "status": str(plan.get("status") or ""),
                "session_count": int(plan.get("session_count") or 0),
                "needs_user_feedback": bool(review.get("user_feedback_requested", False)),
                "updated_at": str(plan.get("updated_at") or ""),
            },
        )
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan fetch failed: {exc}")
    finally:
        db.close()


def _queue_get_plan_async_operation(*, user_id: str, schedule_id: str) -> dict:
    operation_id = operation_logs.start(
        user_id=user_id,
        kind="plan_get",
        message="Latest plan fetch queued",
        metadata={"schedule_id": schedule_id, "mode": "async"},
    )

    operation_logs.append(operation_id, "Queued background latest-plan fetch job")

    worker = threading.Thread(
        target=_run_get_plan_background,
        kwargs={
            "operation_id": operation_id,
            "user_id": user_id,
            "schedule_id": schedule_id,
        },
        daemon=True,
    )
    worker.start()

    return {
        "operation_id": operation_id,
        "kind": "plan_get",
        "status": "running",
        "schedule_id": schedule_id,
    }


def _run_revise_plan_background(
    *,
    operation_id: str,
    user_id: str,
    schedule_id: str,
    feedback: str,
    plan_id: str | None,
    feedback_history: list[str],
) -> None:
    db = SessionLocal()
    try:
        get_schedule_or_404(db, user_id, schedule_id)
        operation_logs.append(operation_id, "Background plan revision job started")

        plan = revise_plan_with_feedback(
            db,
            schedule_id,
            feedback,
            user_id=user_id,
            plan_id=plan_id,
            operation_id=operation_id,
            feedback_history=feedback_history,
        )

        operation_logs.succeed(
            operation_id,
            "Plan revision completed",
            metadata={
                "plan_id": str(plan.get("id") or ""),
                "session_count": int(plan.get("session_count") or 0),
                "needs_user_feedback": bool(dict(plan.get("review") or {}).get("user_feedback_requested", False)),
            },
        )
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan revision failed: {exc}")
    finally:
        db.close()


def _queue_patch_plan_async_operation(
    *,
    user_id: str,
    schedule_id: str,
    feedback: str,
    plan_id: str | None,
    feedback_history: list[str],
) -> dict:
    operation_id = operation_logs.start(
        user_id=user_id,
        kind="plan_revise",
        message="Plan revision queued",
        metadata={"schedule_id": schedule_id, "mode": "async"},
    )

    operation_logs.append(
        operation_id,
        "Queued background plan revision job",
        metadata={
            "has_plan_id": bool(plan_id),
            "feedback_history_count": len(feedback_history),
        },
    )

    worker = threading.Thread(
        target=_run_revise_plan_background,
        kwargs={
            "operation_id": operation_id,
            "user_id": user_id,
            "schedule_id": schedule_id,
            "feedback": feedback,
            "plan_id": plan_id,
            "feedback_history": feedback_history,
        },
        daemon=True,
    )
    worker.start()

    return {
        "operation_id": operation_id,
        "kind": "plan_revise",
        "status": "running",
        "schedule_id": schedule_id,
    }


class PlanGenerateRequest(BaseModel):
    constraints: dict = Field(
        default_factory=dict,
        description="Scheduling and planning constraints (date range, cadence, calendar mode, etc.).",
    )
    user_feedback: str | None = Field(
        default=None,
        description="Optional immediate feedback to apply right after draft generation.",
    )
    feedback_history: list[str] = Field(
        default_factory=list,
        description="Optional prior feedback turns to preserve revision context.",
    )


class PlanPatchRequest(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Optional explicit plan ID to revise. If omitted, backend revises the latest plan.",
    )
    feedback: str = Field(
        min_length=1,
        description="Natural-language feedback used to revise the selected plan (or the latest plan when omitted).",
    )
    feedback_history: list[str] = Field(
        default_factory=list,
        description="Optional earlier feedback turns to carry forward into this revision.",
    )


class PlanCalendarSyncRequest(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Optional explicit plan ID. If omitted, backend uses active plan first, then latest draft/latest plan.",
    )
    calendar_id: str | None = Field(
        default=None,
        description="Optional calendar ID override. Defaults to plan constraints calendar_id or 'primary'.",
    )
    timezone_name: str | None = Field(
        default=None,
        description="Optional timezone override. Defaults to plan constraints timezone_name or Asia/Kolkata (IST).",
    )
    skip_existing: bool = Field(
        default=True,
        description="Skips slots already marked as synced to avoid duplicate event creation.",
    )

class PlanCalendarUnsyncRequest(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Optional explicit plan ID. If omitted, backend uses active plan first, then latest draft/latest plan.",
    )
    calendar_id: str | None = Field(
        default=None,
        description="Optional calendar ID override. Defaults to plan constraints calendar_id or 'primary'.",
    )
    timezone_name: str | None = Field(
        default=None,
        description="Optional timezone override. Defaults to plan constraints timezone_name or Asia/Kolkata (IST).",
    )


class PlanConfirmRequest(BaseModel):
    plan_id: str | None = Field(
        default=None,
        description="Optional explicit plan ID to confirm. If omitted, backend confirms the latest plan.",
    )
    sync_calendar: bool = Field(
        default=False,
        description="When true, creates Google Calendar events for the confirmed plan in the same request.",
    )
    calendar_id: str | None = Field(
        default=None,
        description="Optional calendar ID override for confirm-time sync.",
    )
    timezone_name: str | None = Field(
        default=None,
        description="Optional timezone override for confirm-time sync.",
    )
    skip_existing: bool = Field(
        default=True,
        description="Skips already synced slots when confirm-time sync is enabled.",
    )


@router.post(
    "/generate",
    summary="Generate draft plan",
    description=(
        "Builds a fresh draft plan for the schedule using ingested chunks, constraints, "
        "calendar availability, and internal review signals."
    ),
    response_description="Generated draft plan payload.",
)
def generate_plan(
    schedule_id: str,
    payload: PlanGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    has_existing_plan = (
        db.scalar(
            select(StudyPlan.id)
            .where(StudyPlan.schedule_id == schedule_id)
            .limit(1)
        )
        is not None
    )
    effective_user_feedback = str(payload.user_feedback or "").strip() if has_existing_plan else ""
    effective_feedback_history = list(payload.feedback_history or []) if has_existing_plan else []

    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_generate",
        message="Plan generation started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        operation_logs.append(
            operation_id,
            "Building draft plan from constraints",
            metadata={
                "constraint_keys": sorted(list(payload.constraints.keys())),
                "has_user_feedback": bool(effective_user_feedback),
                "feedback_history_count": len(effective_feedback_history),
            },
        )
        plan = generate_draft_plan(
            db,
            schedule_id,
            payload.constraints,
            user_id=current_user.id,
            operation_id=operation_id,
            user_feedback=effective_user_feedback,
            feedback_history=effective_feedback_history,
        )
        operation_logs.succeed(
            operation_id,
            "Draft plan generated",
            metadata={
                "session_count": int(plan.get("session_count") or 0),
                "needs_user_feedback": bool(dict(plan.get("review") or {}).get("user_feedback_requested", False)),
            },
        )
        return {**plan, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan generation failed: {exc}")
        raise


@router.post(
    "/generate-async",
    summary="Start draft plan generation (async)",
    description=(
        "Starts draft plan generation in a background worker and returns operation_id immediately. "
        "Use /operations/{operation_id}/logs or /operations/{operation_id}/stream to track live progress."
    ),
    response_description="Operation handle for live progress tracking.",
)
def generate_plan_async(
    schedule_id: str,
    payload: PlanGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    has_existing_plan = (
        db.scalar(
            select(StudyPlan.id)
            .where(StudyPlan.schedule_id == schedule_id)
            .limit(1)
        )
        is not None
    )
    effective_user_feedback = str(payload.user_feedback or "").strip() if has_existing_plan else ""
    effective_feedback_history = list(payload.feedback_history or []) if has_existing_plan else []

    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_generate",
        message="Plan generation queued",
        metadata={"schedule_id": schedule_id, "mode": "async"},
    )

    operation_logs.append(
        operation_id,
        "Queued background planner job",
        metadata={
            "constraint_keys": sorted(list(payload.constraints.keys())),
            "has_user_feedback": bool(effective_user_feedback),
            "feedback_history_count": len(effective_feedback_history),
        },
    )

    worker = threading.Thread(
        target=_run_generate_plan_background,
        kwargs={
            "operation_id": operation_id,
            "user_id": current_user.id,
            "schedule_id": schedule_id,
            "constraints": dict(payload.constraints or {}),
            "user_feedback": effective_user_feedback,
            "feedback_history": effective_feedback_history,
        },
        daemon=True,
    )
    worker.start()

    return {
        "operation_id": operation_id,
        "kind": "plan_generate",
        "status": "running",
        "schedule_id": schedule_id,
    }


@router.get(
    "/logs",
    summary="Get planner logs in terminal format",
    description=(
        "Returns planner operation logs as terminal-style lines (for example, '[Planner] ...'). "
        "Use operation_id from /generate or /generate-async response."
    ),
    response_description="Planner log snapshot with formatted lines and plain-text log block.",
)
def get_plan_logs(
    schedule_id: str,
    operation_id: str = Query(..., min_length=1, description="Operation ID returned by plan generation/revision endpoints."),
    offset: int = Query(0, ge=0, description="Log offset to resume from."),
    limit: int = Query(200, ge=1, le=500, description="Maximum entries to return."),
    include_metadata: bool = Query(False, description="When true, appends metadata key/value pairs to each line."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)

    payload = operation_logs.get_logs(
        operation_id=str(operation_id).strip(),
        user_id=current_user.id,
        offset=offset,
        limit=limit,
    )
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")

    lines = [
        _format_planner_log_line(entry, include_metadata=include_metadata)
        for entry in list(payload.get("entries") or [])
    ]

    return {
        **payload,
        "schedule_id": schedule_id,
        "format": "planner_text",
        "lines": lines,
        "text": "\n".join(lines),
    }


@router.get(
    "",
    summary="Get latest plan",
    description="Returns the latest plan for the schedule, regardless of status.",
    response_description="Current plan payload or null when absent.",
)
def get_plan(
    schedule_id: str,
    async_mode: bool = Query(
        default=False,
        alias="async",
        description=(
            "When true, queues latest-plan retrieval and returns operation_id immediately. "
            "Use /operations/{operation_id}/logs or /operations/{operation_id}/stream, then call this "
            "endpoint without async to fetch the full payload."
        ),
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    if async_mode:
        return _queue_get_plan_async_operation(user_id=current_user.id, schedule_id=schedule_id)
    return get_current_plan(db, schedule_id)


@router.get(
    "/async",
    summary="Get latest plan (async)",
    description=(
        "Queues latest-plan retrieval in a background worker and returns operation_id immediately. "
        "Use /operations/{operation_id}/logs or /operations/{operation_id}/stream to track progress, "
        "then call /schedules/{schedule_id}/plan for the full payload."
    ),
    response_description="Operation handle for async latest-plan retrieval.",
)
def get_plan_async(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return _queue_get_plan_async_operation(user_id=current_user.id, schedule_id=schedule_id)


@router.get(
    "/all",
    summary="List all plans",
    description="Returns all plans for the schedule, ordered by most recently updated first.",
    response_description="Array of plan payloads.",
)
def list_schedule_plans(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return list_plans_for_schedule(db, schedule_id)


@router.delete(
    "/{plan_id}",
    summary="Delete one plan",
    description=(
        "Deletes a specific plan for the schedule and removes dependent materialized sessions "
        "and session chat records associated with those sessions."
    ),
    response_description="Deletion summary for the specified plan.",
)
def delete_schedule_plan(
    schedule_id: str,
    plan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_delete",
        message="Plan deletion started",
        metadata={"schedule_id": schedule_id, "plan_id": plan_id},
    )

    try:
        operation_logs.append(operation_id, "Deleting plan and dependent session records")
        payload = delete_plan(db, schedule_id, plan_id)
        operation_logs.succeed(
            operation_id,
            "Plan deletion completed",
            metadata={
                "deleted_materialized_sessions": int(payload.get("deleted_materialized_sessions") or 0),
                "deleted_session_chat_messages": int(payload.get("deleted_session_chat_messages") or 0),
            },
        )
        return {**payload, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan deletion failed: {exc}")
        raise


@router.delete(
    "",
    summary="Delete all plans",
    description=(
        "Deletes all plans for the schedule and removes all dependent materialized sessions "
        "and session chat records associated with those sessions."
    ),
    response_description="Deletion summary for all schedule plans.",
)
def delete_schedule_plan_collection(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_delete_all",
        message="Bulk plan deletion started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        operation_logs.append(operation_id, "Deleting all plans and dependent session records")
        payload = delete_all_plans(db, schedule_id)
        operation_logs.succeed(
            operation_id,
            "Bulk plan deletion completed",
            metadata={
                "deleted_plans": int(payload.get("deleted_plans") or 0),
                "deleted_materialized_sessions": int(payload.get("deleted_materialized_sessions") or 0),
                "deleted_session_chat_messages": int(payload.get("deleted_session_chat_messages") or 0),
            },
        )
        return {**payload, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Bulk plan deletion failed: {exc}")
        raise


@router.patch(
    "",
    summary="Revise plan with feedback",
    description=(
        "Applies natural-language feedback to the selected plan (or latest plan) and regenerates sessions. "
        "Set async=true to queue background revision and return operation_id immediately."
    ),
    response_description="Updated draft plan payload.",
)
def patch_plan(
    schedule_id: str,
    payload: PlanPatchRequest,
    async_mode: bool = Query(
        default=False,
        alias="async",
        description=(
            "When true, queues plan revision and returns operation_id immediately. "
            "Use /operations/{operation_id}/logs or /operations/{operation_id}/stream to track progress."
        ),
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    normalized_plan_id = str(payload.plan_id or "").strip() or None
    feedback_history = list(payload.feedback_history or [])

    if async_mode:
        return _queue_patch_plan_async_operation(
            user_id=current_user.id,
            schedule_id=schedule_id,
            feedback=payload.feedback,
            plan_id=normalized_plan_id,
            feedback_history=feedback_history,
        )

    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_revise",
        message="Plan revision started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        operation_logs.append(
            operation_id,
            "Applying feedback to regenerate plan",
            metadata={
                "has_plan_id": bool(normalized_plan_id),
                "feedback_history_count": len(feedback_history),
            },
        )
        plan = revise_plan_with_feedback(
            db,
            schedule_id,
            payload.feedback,
            user_id=current_user.id,
            plan_id=normalized_plan_id,
            operation_id=operation_id,
            feedback_history=feedback_history,
        )
        operation_logs.succeed(
            operation_id,
            "Plan revision completed",
            metadata={
                "session_count": int(plan.get("session_count") or 0),
                "needs_user_feedback": bool(dict(plan.get("review") or {}).get("user_feedback_requested", False)),
            },
        )
        return {**plan, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan revision failed: {exc}")
        raise


@router.patch(
    "/patch-async",
    summary="Start plan revision with feedback (async)",
    description=(
        "Starts plan revision in a background worker and returns operation_id immediately. "
        "Use /operations/{operation_id}/logs or /operations/{operation_id}/stream to track live progress."
    ),
    response_description="Operation handle for async plan revision.",
)
def patch_plan_async(
    schedule_id: str,
    payload: PlanPatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    normalized_plan_id = str(payload.plan_id or "").strip() or None
    feedback_history = list(payload.feedback_history or [])
    return _queue_patch_plan_async_operation(
        user_id=current_user.id,
        schedule_id=schedule_id,
        feedback=payload.feedback,
        plan_id=normalized_plan_id,
        feedback_history=feedback_history,
    )


@router.post(
    "/confirm",
    summary="Confirm plan",
    description=(
        "Promotes the selected/latest plan to active status and materializes study sessions "
        "for session-focused workflows."
    ),
    response_description="Confirmation and materialization summary.",
)
def confirm_schedule_plan(
    schedule_id: str,
    payload: PlanConfirmRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_confirm",
        message="Plan confirmation started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        options = payload or PlanConfirmRequest()
        confirmed = confirm_plan(
            db,
            schedule_id,
            plan_id=str(options.plan_id or "").strip() or None,
        )
        sync_error = ""
        sync_report = None

        if options.sync_calendar:
            operation_logs.append(
                operation_id,
                "Syncing confirmed sessions to Google Calendar",
                metadata={
                    "calendar_id_override": bool(str(options.calendar_id or "").strip()),
                    "timezone_override": bool(str(options.timezone_name or "").strip()),
                    "skip_existing": bool(options.skip_existing),
                },
            )
            try:
                sync_report = sync_plan_sessions_to_calendar(
                    db,
                    schedule_id,
                    user_id=current_user.id,
                    plan_id=str(confirmed.get("plan_id") or "").strip() or None,
                    calendar_id=str(options.calendar_id or "").strip() or None,
                    timezone_name=str(options.timezone_name or "").strip() or None,
                    skip_existing=bool(options.skip_existing),
                    expected_google_id=str(current_user.google_id or "").strip() or None,
                    expected_email=str(current_user.email or "").strip().lower() or None,
                )
            except Exception as exc:
                sync_error = str(exc)
                operation_logs.append(
                    operation_id,
                    "Calendar sync failed after confirmation",
                    level="error",
                    metadata={"error": sync_error},
                )

        success_metadata = {
            "materialized_sessions": int(confirmed.get("materialized_sessions") or 0),
        }
        if sync_report is not None:
            success_metadata.update(
                {
                    "calendar_created": int(sync_report.get("created") or 0),
                    "calendar_failed": int(sync_report.get("failed") or 0),
                    "calendar_skipped_existing": int(sync_report.get("skipped_existing") or 0),
                }
            )
        if sync_error:
            success_metadata["calendar_sync_error"] = sync_error

        operation_logs.succeed(
            operation_id,
            "Plan confirmed and sessions materialized",
            metadata=success_metadata,
        )

        response = {**confirmed, "operation_id": operation_id}
        if sync_report is not None:
            response["calendar_sync"] = sync_report
        if sync_error:
            response["calendar_sync_error"] = sync_error
        return response
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan confirmation failed: {exc}")
        raise


@router.post(
    "/sync-calendar",
    summary="Create calendar events for plan sessions",
    description=(
        "Creates Google Calendar events from plan sessions and persists per-slot sync status "
        "(calendar_event_id/calendar_status) inside the plan payload."
    ),
    response_description="Calendar sync report with per-slot results.",
)
def sync_plan_calendar(
    schedule_id: str,
    payload: PlanCalendarSyncRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_sync_calendar",
        message="Plan calendar sync started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        operation_logs.append(
            operation_id,
            "Creating calendar events from plan sessions",
            metadata={
                "has_plan_id": bool(str(payload.plan_id or "").strip()),
                "calendar_id_override": bool(str(payload.calendar_id or "").strip()),
                "timezone_override": bool(str(payload.timezone_name or "").strip()),
                "skip_existing": bool(payload.skip_existing),
            },
        )
        report = sync_plan_sessions_to_calendar(
            db,
            schedule_id,
            user_id=current_user.id,
            plan_id=str(payload.plan_id or "").strip() or None,
            calendar_id=str(payload.calendar_id or "").strip() or None,
            timezone_name=str(payload.timezone_name or "").strip() or None,
            skip_existing=bool(payload.skip_existing),
            expected_google_id=str(current_user.google_id or "").strip() or None,
            expected_email=str(current_user.email or "").strip().lower() or None,
        )
        operation_logs.succeed(
            operation_id,
            "Plan calendar sync completed",
            metadata={
                "created": int(report.get("created") or 0),
                "failed": int(report.get("failed") or 0),
                "skipped_existing": int(report.get("skipped_existing") or 0),
            },
        )
        return {**report, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan calendar sync failed: {exc}")
        raise


@router.post(
    "/unsync-calendar",
    summary="Delete calendar events for plan sessions",
    description=(
        "Deletes Google Calendar events linked to plan sessions and updates per-slot sync metadata "
        "without deleting the plan from the database."
    ),
    response_description="Calendar unsync report with per-slot deletion results.",
)
def unsync_plan_calendar(
    schedule_id: str,
    payload: PlanCalendarUnsyncRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    operation_id = operation_logs.start(
        user_id=current_user.id,
        kind="plan_unsync_calendar",
        message="Plan calendar unsync started",
        metadata={"schedule_id": schedule_id},
    )

    try:
        operation_logs.append(
            operation_id,
            "Deleting calendar events linked to plan sessions",
            metadata={
                "has_plan_id": bool(str(payload.plan_id or "").strip()),
                "calendar_id_override": bool(str(payload.calendar_id or "").strip()),
                "timezone_override": bool(str(payload.timezone_name or "").strip()),
            },
        )
        report = unsync_plan_sessions_from_calendar(
            db,
            schedule_id,
            user_id=current_user.id,
            plan_id=str(payload.plan_id or "").strip() or None,
            calendar_id=str(payload.calendar_id or "").strip() or None,
            timezone_name=str(payload.timezone_name or "").strip() or None,
            expected_google_id=str(current_user.google_id or "").strip() or None,
            expected_email=str(current_user.email or "").strip().lower() or None,
        )
        operation_logs.succeed(
            operation_id,
            "Plan calendar unsync completed",
            metadata={
                "deleted": int(report.get("deleted") or 0),
                "failed": int(report.get("failed") or 0),
                "skipped_not_synced": int(report.get("skipped_not_synced") or 0),
            },
        )
        return {**report, "operation_id": operation_id}
    except Exception as exc:
        operation_logs.fail(operation_id, f"Plan calendar unsync failed: {exc}")
        raise


@router.get(
    "/sessions",
    summary="List materialized sessions",
    description="Lists study sessions created when a plan is confirmed.",
    response_description="Array of materialized session records.",
)
def list_plan_sessions(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return list_materialized_sessions(db, schedule_id)


@router.get(
    "/sessions/{session_id}",
    summary="Get materialized session detail",
    description=(
        "Returns one materialized study session with focus chunks, prerequisites, "
        "briefing state, and upcoming-session context."
    ),
    response_description="Detailed materialized session payload.",
)
def get_plan_session(
    schedule_id: str,
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_schedule_or_404(db, current_user.id, schedule_id)
    return get_materialized_session_detail(db, schedule_id, session_id)
