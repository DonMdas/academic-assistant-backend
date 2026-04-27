import re
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.auth.calendar_credentials import decrypt_calendar_credentials, encrypt_calendar_credentials
from backend.config import settings
from backend.db.models import Chunk, Document, GoogleCalendarCredential, SessionChatMessage, StudyPlan, StudySession
from backend.operations.logs import operation_logs
from backend.timezone_utils import today_ist
from db import DocumentDB
from planner_ai import (
    GemmaPlannerAgent,
    _apply_qwen_escalation_if_needed,
    _plan_snapshot,
    _post_review_refinement_reasons,
    apply_chunk_prerequisite_updates,
    apply_constraint_updates,
    qwen_review_plan,
    reset_planner_runtime_log_sink,
    set_planner_runtime_log_sink,
)
from planner_calendar import GoogleCalendarService
from planner_common import (
    ChunkRecord,
    DEFAULT_CONSTRAINTS,
    GEMMA_PLANNER_MAX_TURNS,
    _constraints_diff,
    _coverage_diff,
    _locked_constraint_keys,
    _normalize_chunk_hints,
    _normalize_constraints,
    _normalize_difficulty,
    _normalize_int,
    _normalize_text_list,
    _normalize_timezone_name,
    _parse_date_str,
    _resolve_timezone,
)
from planner_scheduling import _build_slot_description, _build_slot_title, build_schedule_data, load_session_chunks
from planner_toolbox import PlanningToolbox
from sqlalchemy.orm.attributes import flag_modified

_CORE_PLAN_KEYS = set(DEFAULT_CONSTRAINTS.keys()).union(
    {"start_date", "end_date", "timezone_name", "calendar_id", "calendar_mode"}
)


class NullCalendarService:
    def __init__(self, timezone_name: str) -> None:
        self.timezone_name = timezone_name
        self.tzinfo = _resolve_timezone(timezone_name)

    def list_events(self, calendar_id, start_dt, end_dt):
        return []


def _parse_iso(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _resolve_date_range(constraints: dict) -> tuple[date, date]:
    start_date = _parse_date_str(constraints.get("start_date")) or today_ist()
    end_date = _parse_date_str(constraints.get("end_date")) or (start_date + timedelta(days=14))

    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="constraints.end_date must be on or after start_date",
        )

    return start_date, end_date


def _op_log(
    operation_id: str | None,
    message: str,
    metadata: dict[str, Any] | None = None,
    level: str = "info",
) -> None:
    if not operation_id:
        return
    operation_logs.append(operation_id=operation_id, message=message, metadata=metadata, level=level)


def _build_planner_runtime_sink(operation_id: str | None):
    op_id = str(operation_id or "").strip()

    def _sink(line: str) -> None:
        if not op_id:
            return
        text = str(line or "").strip()
        if not text:
            return
        if text.startswith("[Planner]"):
            text = text[len("[Planner]") :].strip()
        _op_log(op_id, text)

    return _sink


def _strip_to_core_plan_keys(constraints: dict | None) -> dict:
    source = dict(constraints or {})
    out = {}
    for key in _CORE_PLAN_KEYS:
        if key in source:
            out[key] = source.get(key)
    return out


def _coerce_feedback_history(feedback_history: list[str] | None) -> list[str]:
    out: list[str] = []
    for row in list(feedback_history or []):
        text = str(row or "").strip()
        if not text:
            continue
        out.append(text)
    return out


def _is_affirmative_feedback(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    tokens = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead", "proceed"}
    if normalized in tokens:
        return True
    return any(token in normalized for token in ["yes", "do it", "go ahead", "proceed", "sure", "okay", "ok"])


def _is_negative_feedback(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    tokens = {"no", "n", "nope", "nah", "do not", "don't", "dont", "keep"}
    if normalized in tokens:
        return True
    return any(token in normalized for token in ["no", "do not", "don't", "dont", "keep as", "leave as"])


def _extract_daily_limit_from_clarification(question: str) -> int | None:
    text = str(question or "").strip().lower()
    if not text:
        return None

    for pattern in [
        r"\bupdate(?:\s+\w+){0,8}\s+to\s+(\d{1,4})\s*minutes?\b",
        r"\((\d{1,4})\s*minutes?\)",
        r"\b(\d{1,4})\s*minutes?\b",
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except Exception:
            continue
        if 30 <= value <= 720:
            return value
    return None


def _extract_minutes_value_from_text(text: str) -> int | None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None

    compound = re.search(
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)\s*(?:and\s*)?(\d+(?:\.\d+)?)?\s*(minutes?|mins?|min|m)?\b",
        lowered,
    )
    if compound:
        try:
            hours = float(compound.group(1))
            mins = float(compound.group(3)) if compound.group(3) and compound.group(4) else 0.0
            total = int(round(hours * 60 + mins))
            if 30 <= total <= 720:
                return total
        except Exception:
            pass

    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m)\b", lowered)
    if not match:
        return None

    try:
        amount = float(match.group(1))
    except Exception:
        return None

    unit = str(match.group(2) or "").strip().lower()
    minutes = int(round(amount * 60)) if unit.startswith("h") else int(round(amount))
    if 30 <= minutes <= 720:
        return minutes
    return None


def _extract_direct_daily_override_from_feedback(feedback_text: str, constraints: dict) -> int | None:
    text = str(feedback_text or "").strip().lower()
    if not text:
        return None

    # Direct override intent: "set/update/change ... to 75 mins"
    if not re.search(r"\b(?:set|update|change|make)\b", text):
        return None
    if not re.search(r"\bto\b", text):
        return None

    minutes = _extract_minutes_value_from_text(text)
    if minutes is None:
        return None

    if re.search(r"\b(?:daily|per\s*day|a\s*day|study\s*limit|session\s*time)\b", text):
        return minutes

    if re.search(r"\b(?:slot|session\s*length|min\s*slot|max\s*slot)\b", text):
        return None

    # If existing constraints clearly track a daily study limit request, treat this as daily override.
    additional = str(dict(constraints or {}).get("additional_constraints") or "").strip().lower()
    current_daily = dict(constraints or {}).get("daily_max_minutes")
    if current_daily is not None and re.search(r"\b(?:daily|per\s*day|session\s*time)\b", additional):
        return minutes

    return None


def _apply_pending_clarification_to_revision(
    constraints: dict,
    feedback_text: str,
    feedback_history: list[str],
) -> tuple[dict, list[str], dict[str, Any]]:
    next_constraints = dict(constraints or {})
    next_history = list(feedback_history or [])
    applied_updates: dict[str, Any] = {}

    question = str(next_constraints.get("clarification_question") or "").strip()
    if not question:
        question = str(next_constraints.get("feedback_prompt") or "").strip()
    if not question:
        return next_constraints, next_history, applied_updates

    question_context = f"Planner clarification: {question}"
    if question_context not in next_history:
        next_history.append(question_context)

    response_text = str(feedback_text or "").strip()
    if response_text:
        response_context = f"User response: {response_text}"
        if response_context not in next_history:
            next_history.append(response_context)

    lowered_question = question.lower()
    if "daily" in lowered_question and "minute" in lowered_question:
        if _is_affirmative_feedback(response_text):
            requested = _extract_daily_limit_from_clarification(question)
            if requested is not None:
                next_constraints["daily_max_minutes"] = requested
                applied_updates["daily_max_minutes"] = requested
        elif _is_negative_feedback(response_text):
            applied_updates["daily_max_minutes"] = next_constraints.get("daily_max_minutes")

    if response_text:
        next_constraints["clarification_question"] = ""
        next_constraints["feedback_prompt"] = ""
        next_constraints["feedback_source"] = ""
        next_constraints["user_feedback_requested"] = False

    return next_constraints, next_history, applied_updates


def _extract_constraint_updates(gemma_feedback: dict | None) -> dict:
    updates: dict[str, Any] = {}
    payload = dict(gemma_feedback or {})

    model_updates = payload.get("updated_constraints", {})
    if isinstance(model_updates, dict):
        updates.update(model_updates)

    tool_updates = payload.get("_tool_constraint_updates", {})
    if isinstance(tool_updates, dict):
        updates.update(tool_updates)

    return updates


def _extract_pending_clarification_question(gemma_feedback: dict | None) -> str:
    payload = dict(gemma_feedback or {})

    direct_question = str(payload.get("clarification_question") or "").strip()
    if direct_question:
        return direct_question

    history = list(payload.get("_history") or [])
    for row in reversed(history):
        if not isinstance(row, dict):
            continue

        assistant_action = dict(row.get("assistant_action") or {})
        if str(assistant_action.get("action") or "").strip().lower() != "ask_user":
            continue

        question = str(assistant_action.get("question") or "").strip()
        user_answer = str(row.get("user_answer") or "").strip()
        if question and not user_answer:
            return question

    return ""


def _build_schedule_once(
    *,
    chunks: list[ChunkRecord],
    constraints: dict,
    start_date: date,
    end_date: date,
    tzinfo,
    calendar_service,
    calendar_id: str,
) -> tuple[list[dict], dict, dict, date, dict]:
    buffer_days = _normalize_int(constraints.get("buffer_days"), 1, minimum=0, maximum=14)
    schedule_end = end_date - timedelta(days=buffer_days)
    if schedule_end < start_date:
        schedule_end = start_date

    schedule_data = build_schedule_data(
        chunks=chunks,
        constraints=constraints,
        start_date=start_date,
        end_date=end_date,
        schedule_end=schedule_end,
        tzinfo=tzinfo,
        calendar_service=calendar_service,
        calendar_id=calendar_id,
    )

    next_constraints = _normalize_constraints(dict(schedule_data.get("constraints", constraints)))
    slots = list(schedule_data.get("slots", []))
    coverage = dict(schedule_data.get("coverage", {}))
    return slots, next_constraints, coverage, schedule_end, schedule_data


def _prepare_plan_run_context(
    db: Session,
    schedule_id: str,
    constraints: dict,
    operation_id: str | None,
    user_id: str | None = None,
) -> dict:
    core_constraints = _strip_to_core_plan_keys(constraints)

    merged = dict(DEFAULT_CONSTRAINTS)
    merged.update(core_constraints)
    normalized_constraints = _normalize_constraints(merged)

    start_date, end_date = _resolve_date_range(merged)

    timezone_name = _normalize_timezone_name(merged.get("timezone_name") or "Asia/Kolkata")
    calendar_id = str(merged.get("calendar_id") or "primary").strip() or "primary"
    calendar_mode = "real"

    if calendar_mode == "real":
        try:
            if str(user_id or "").strip():
                credential_payload = _load_calendar_credentials_for_user(db, str(user_id).strip())
                calendar_service = GoogleCalendarService(
                    timezone_name=timezone_name,
                    credentials_info=credential_payload,
                    allow_local_oauth=False,
                )
            else:
                calendar_service = GoogleCalendarService(timezone_name=timezone_name)
            calendar_warning = ""
        except Exception as exc:
            calendar_service = NullCalendarService(timezone_name=timezone_name)
            calendar_warning = f"Calendar disabled for this run: {exc}"
            _op_log(
                operation_id,
                "Calendar service unavailable; falling back to offline calendar mode",
                metadata={"warning": calendar_warning},
                level="warning",
            )
    else:
        calendar_service = NullCalendarService(timezone_name=timezone_name)
        calendar_warning = ""

    chunks = _load_chunks_for_schedule(db, schedule_id)
    if not chunks:
        ingest_summary = _document_ingest_summary(db, schedule_id)
        detail = (
            "No ingested chunks found for this schedule. Upload and ingest documents first. "
            f"Document status counts: {ingest_summary}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    return {
        "constraints": normalized_constraints,
        "start_date": start_date,
        "end_date": end_date,
        "timezone_name": timezone_name,
        "calendar_id": calendar_id,
        "calendar_mode": calendar_mode,
        "calendar_warning": calendar_warning,
        "calendar_service": calendar_service,
        "chunks": chunks,
        "tzinfo": calendar_service.tzinfo,
    }


def _needs_user_feedback(qwen_feedback: dict | None) -> bool:
    payload = dict(qwen_feedback or {})
    severity = _normalize_int(payload.get("severity"), 1, minimum=0, maximum=3)
    approval_ready = bool(payload.get("approval_ready", False))
    return severity >= 2 or (not approval_ready)


def _suggest_feedback_prompt(qwen_feedback: dict | None) -> str:
    payload = dict(qwen_feedback or {})
    suggestions = _normalize_text_list(payload.get("suggested_adjustments", []))
    if suggestions:
        return suggestions[0]

    risks = _normalize_text_list(payload.get("risks", []))
    if risks:
        return f"Please share feedback to address: {risks[0]}"

    return "Share any constraints you want adjusted (days, timing, pace, or blocked dates)."


def _build_review_payload(constraints: dict) -> dict:
    qwen_feedback = dict(constraints.get("qwen_feedback") or {})
    severity = _normalize_int(qwen_feedback.get("severity"), 1, minimum=0, maximum=3)
    approval_ready = bool(qwen_feedback.get("approval_ready", False))
    user_feedback_requested = bool(constraints.get("user_feedback_requested", False))
    feedback_prompt = str(constraints.get("feedback_prompt") or "").strip() or None
    feedback_source = str(constraints.get("feedback_source") or "").strip() or None
    clarification_question = str(constraints.get("clarification_question") or "").strip() or None

    return {
        "qwen_feedback": qwen_feedback,
        "severity": severity,
        "approval_ready": approval_ready,
        "user_feedback_requested": user_feedback_requested,
        "feedback_prompt": feedback_prompt,
        "feedback_source": feedback_source,
        "clarification_question": clarification_question,
        "clarification_requested": bool(clarification_question),
    }


def _serialize_plan(plan: StudyPlan) -> dict:
    sessions = list(plan.sessions_payload or [])
    constraints = dict(plan.constraints_json or {})
    return {
        "id": plan.id,
        "schedule_id": plan.schedule_id,
        "status": plan.status,
        "constraints": constraints,
        "review": _build_review_payload(constraints),
        "sessions": sessions,
        "session_count": len(sessions),
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "updated_at": plan.updated_at.isoformat() if plan.updated_at else None,
    }


def _load_chunks_for_schedule(db: Session, schedule_id: str) -> list[ChunkRecord]:
    rows = db.scalars(
        select(Chunk)
        .where(Chunk.schedule_id == schedule_id)
        .order_by(Chunk.created_at.asc())
    ).all()

    if rows:
        out: list[ChunkRecord] = []
        for row in rows:
            metadata = dict(row.metadata_json or {})
            out.append(
                ChunkRecord(
                    chunk_id=str(row.id),
                    doc_id=str(row.document_id),
                    topic=str(metadata.get("topic") or "Unknown Topic").strip(),
                    subtopics=_normalize_text_list(metadata.get("subtopics")),
                    summary=str(metadata.get("summary") or "").strip(),
                    difficulty=_normalize_difficulty(metadata.get("complexity")),
                    prerequisites=_normalize_text_list(metadata.get("prerequisites")),
                    scheduling_hints=_normalize_chunk_hints(metadata.get("scheduling_hints")),
                    estimated_time=_normalize_int(metadata.get("estimated_time"), 30, minimum=10, maximum=240),
                    content=str(row.content or ""),
                )
            )
        return out

    legacy_db = DocumentDB(settings.LEGACY_DB_PATH)
    return load_session_chunks(legacy_db, schedule_id)


def _document_ingest_summary(db: Session, schedule_id: str) -> dict:
    rows = db.scalars(select(Document).where(Document.schedule_id == schedule_id)).all()
    summary = {
        "total": len(rows),
        "pending": 0,
        "processing": 0,
        "done": 0,
        "failed": 0,
        "other": 0,
    }

    for row in rows:
        status_key = str(row.ingest_status or "other").strip().lower() or "other"
        if status_key in summary:
            summary[status_key] += 1
        else:
            summary["other"] += 1
    return summary


def _run_planner_pipeline(
    db: Session,
    schedule_id: str,
    constraints: dict,
    *,
    user_id: str | None = None,
    operation_id: str | None = None,
    run_main_passes: bool = True,
    user_feedback: str = "",
    feedback_history: list[str] | None = None,
) -> tuple[list, dict, dict]:
    run_context = _prepare_plan_run_context(
        db=db,
        schedule_id=schedule_id,
        constraints=constraints,
        operation_id=operation_id,
        user_id=user_id,
    )

    current_constraints = dict(run_context["constraints"])
    start_date = run_context["start_date"]
    end_date = run_context["end_date"]
    timezone_name = run_context["timezone_name"]
    calendar_id = run_context["calendar_id"]
    calendar_mode = run_context["calendar_mode"]
    calendar_warning = run_context["calendar_warning"]
    calendar_service = run_context["calendar_service"]
    tzinfo = run_context["tzinfo"]
    chunks = list(run_context["chunks"])

    locked_constraint_keys = _locked_constraint_keys(current_constraints)
    if str(user_feedback or "").strip():
        # Explicit revision feedback should be allowed to adjust minute caps.
        locked_constraint_keys.discard("daily_max_minutes")
        locked_constraint_keys.discard("min_slot_minutes")
        locked_constraint_keys.discard("max_slot_minutes")

    slots: list[dict] = []
    coverage: dict[str, Any] = {}
    schedule_end = start_date

    gemma_feedback: dict[str, Any] = {}
    qwen_feedback: dict[str, Any] = {}
    gemma_feedback_history: list[dict] = []
    qwen_feedback_history: list[dict] = []
    round_change_log: list[dict] = []
    post_review_reasons: list[str] = []
    feedback_constraint_updates: dict[str, Any] = {}
    gemma_clarification_question = ""

    gemma_agent = None
    gemma_warning = ""
    try:
        gemma_agent = GemmaPlannerAgent()
        _op_log(operation_id, "Gemma planner agent initialized", metadata={"backend": getattr(gemma_agent, "backend", "unknown")})
    except Exception as exc:
        gemma_warning = str(exc)
        _op_log(
            operation_id,
            "Gemma planner unavailable; using deterministic planner path",
            metadata={"warning": gemma_warning},
            level="warning",
        )

    pass_count = 2 if run_main_passes else 1
    for round_idx in range(pass_count):
        round_number = round_idx + 1
        phase_name = f"main_pass_{round_number}" if run_main_passes else "baseline_build"

        _op_log(
            operation_id,
            f"Building schedule ({phase_name})",
            metadata={"round": round_number, "buffer_days": current_constraints.get("buffer_days")},
        )

        before_constraints = dict(current_constraints)
        before_coverage = dict(coverage)

        slots, current_constraints, coverage, schedule_end, schedule_data = _build_schedule_once(
            chunks=chunks,
            constraints=current_constraints,
            start_date=start_date,
            end_date=end_date,
            tzinfo=tzinfo,
            calendar_service=calendar_service,
            calendar_id=calendar_id,
        )

        _op_log(
            operation_id,
            f"Schedule built ({phase_name})",
            metadata={
                "slots": len(slots),
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
            },
        )

        if not run_main_passes:
            round_change_log.append(
                {
                    "phase": phase_name,
                    "constraint_diff": _constraints_diff(before_constraints, current_constraints),
                    "coverage_diff": _coverage_diff(before_coverage, coverage),
                }
            )
            break

        if gemma_agent is None:
            round_change_log.append(
                {
                    "phase": phase_name,
                    "constraint_diff": _constraints_diff(before_constraints, current_constraints),
                    "coverage_diff": _coverage_diff(before_coverage, coverage),
                    "note": "gemma_unavailable",
                }
            )
            break

        session_context = {
            "round": round_number,
            "session_id": schedule_id,
            "locked_constraints": sorted(list(locked_constraint_keys)),
            "changes_since_last_round": round_change_log[-1] if round_change_log else {},
            "round_change_log": round_change_log[-5:],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "schedule_end": schedule_end.isoformat(),
            },
            "chunk_count": len(chunks),
            "busy_event_count": len(list(schedule_data.get("busy_events", []) or [])),
            "free_block_count": len(list(schedule_data.get("free_blocks", []) or [])),
            "constraints": current_constraints,
            "coverage": {
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
                "partial_chunks": coverage.get("partially_covered_chunks"),
            },
        }

        toolbox = PlanningToolbox(chunks=chunks, slots=slots, coverage=coverage, constraints=current_constraints)

        _op_log(operation_id, f"Running Gemma planner loop ({phase_name})", metadata={"max_turns": GEMMA_PLANNER_MAX_TURNS})
        gemma_feedback = gemma_agent.run(
            session_context=session_context,
            toolbox=toolbox,
            max_turns=GEMMA_PLANNER_MAX_TURNS,
            allow_user_input=False,
        )
        gemma_feedback_history.append(dict(gemma_feedback or {}))
        question = _extract_pending_clarification_question(gemma_feedback)
        if question:
            gemma_clarification_question = question

        updated_constraints = apply_constraint_updates(
            current=current_constraints,
            updates=_extract_constraint_updates(gemma_feedback),
            locked_keys=locked_constraint_keys,
        )

        changed = updated_constraints != current_constraints
        needs_regeneration = bool(dict(gemma_feedback or {}).get("needs_regeneration", False))
        current_constraints = updated_constraints

        main_entry = {
            "phase": phase_name,
            "constraint_diff": _constraints_diff(before_constraints, current_constraints),
            "coverage_diff": _coverage_diff(before_coverage, coverage),
        }
        if not main_entry["constraint_diff"] and not main_entry["coverage_diff"]:
            main_entry["note"] = "no_material_changes"
        round_change_log.append(main_entry)

        _op_log(
            operation_id,
            f"Gemma {phase_name} completed",
            metadata={
                "needs_regeneration": needs_regeneration,
                "constraint_changes": len(main_entry["constraint_diff"]),
                "reason": str(dict(gemma_feedback or {}).get("reason") or ""),
            },
        )

        if not needs_regeneration and not changed:
            break
            
        if round_idx == (pass_count - 1):
            # Force a final rebuild if the very last pass updated constraints
            _op_log(operation_id, f"Final round changed constraints, syncing slots before proceeding")
            slots, current_constraints, coverage, schedule_end, schedule_data = _build_schedule_once(
                chunks=chunks,
                constraints=current_constraints,
                start_date=start_date,
                end_date=end_date,
                tzinfo=tzinfo,
                calendar_service=calendar_service,
                calendar_id=calendar_id,
            )
            break

        _op_log(operation_id, "Gemma requested schedule regeneration", metadata={"round": round_number})

    feedback_text = str(user_feedback or "").strip()
    feedback_history_rows = _coerce_feedback_history(feedback_history)
    skip_pre_feedback_review = bool(feedback_text)

    if not skip_pre_feedback_review:
        review_payload = {
            "phase": "api_plan_generate" if run_main_passes else "api_plan_revise",
            "session_id": schedule_id,
            "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=current_constraints),
        }

        _op_log(operation_id, "Running Qwen review", metadata={"phase": review_payload.get("phase")})
        qwen_feedback = qwen_review_plan(review_payload)
        qwen_feedback_history.append(dict(qwen_feedback or {}))
        _op_log(
            operation_id,
            "Qwen review completed",
            metadata={
                "severity": _normalize_int(dict(qwen_feedback or {}).get("severity"), 1, minimum=0, maximum=3),
                "approval_ready": bool(dict(qwen_feedback or {}).get("approval_ready", False)),
            },
        )

        escalation_result = _apply_qwen_escalation_if_needed(
            qwen_feedback=qwen_feedback,
            constraints=current_constraints,
            locked_constraint_keys=locked_constraint_keys,
            chunks=chunks,
            start_date=start_date,
            end_date=end_date,
            schedule_end=schedule_end,
            tzinfo=tzinfo,
            calendar_service=calendar_service,
            calendar_id=calendar_id,
        )
        if escalation_result.get("escalated"):
            before_constraints = dict(current_constraints)
            before_coverage = dict(coverage)

            current_constraints = dict(escalation_result.get("constraints", current_constraints))
            escalated_slots = escalation_result.get("slots")
            if escalated_slots is not None:
                slots = list(escalated_slots)
            escalated_coverage = escalation_result.get("coverage")
            if escalated_coverage is not None:
                coverage = dict(escalated_coverage)
            qwen_feedback = dict(escalation_result.get("qwen_feedback", qwen_feedback))
            schedule_end = escalation_result.get("schedule_end", schedule_end)
            qwen_feedback_history.append(dict(qwen_feedback or {}))

            round_change_log.append(
                {
                    "phase": "qwen_severity_3_reset",
                    "constraint_diff": _constraints_diff(before_constraints, current_constraints),
                    "coverage_diff": _coverage_diff(before_coverage, coverage),
                }
            )
            _op_log(operation_id, "Applied Qwen severity-3 escalation reset", metadata={"severity": 3}, level="warning")

        post_review_reasons = _post_review_refinement_reasons(qwen_feedback, coverage)
        if post_review_reasons:
            _op_log(
                operation_id,
                "Qwen flagged post-review issues",
                metadata={"reasons": post_review_reasons[:5]},
                level="warning",
            )

    if post_review_reasons and gemma_agent is not None:
        review_context = {
            "phase": "post_qwen_review_refinement",
            "session_id": schedule_id,
            "locked_constraints": sorted(list(locked_constraint_keys)),
            "changes_since_last_round": round_change_log[-1] if round_change_log else {},
            "round_change_log": round_change_log[-5:],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "schedule_end": schedule_end.isoformat(),
            },
            "constraints": current_constraints,
            "coverage": {
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
                "partial_chunks": coverage.get("partially_covered_chunks"),
            },
            "qwen_feedback": qwen_feedback,
        }

        review_toolbox = PlanningToolbox(chunks=chunks, slots=slots, coverage=coverage, constraints=current_constraints)
        post_review_gemma_feedback = gemma_agent.run(
            session_context=review_context,
            toolbox=review_toolbox,
            max_turns=max(2, GEMMA_PLANNER_MAX_TURNS - 1),
            allow_user_input=False,
        )
        gemma_feedback_history.append(dict(post_review_gemma_feedback or {}))
        question = _extract_pending_clarification_question(post_review_gemma_feedback)
        if question:
            gemma_clarification_question = question

        refined_constraints = apply_constraint_updates(
            current=current_constraints,
            updates=_extract_constraint_updates(post_review_gemma_feedback),
            locked_keys=locked_constraint_keys,
        )

        before_constraints = dict(current_constraints)
        before_coverage = dict(coverage)
        should_regenerate = bool(dict(post_review_gemma_feedback or {}).get("needs_regeneration", False))
        constraints_changed = refined_constraints != current_constraints

        if should_regenerate or constraints_changed:
            current_constraints = refined_constraints
            slots, current_constraints, coverage, schedule_end, _ = _build_schedule_once(
                chunks=chunks,
                constraints=current_constraints,
                start_date=start_date,
                end_date=end_date,
                tzinfo=tzinfo,
                calendar_service=calendar_service,
                calendar_id=calendar_id,
            )

            review_payload = {
                "phase": "post_qwen_refinement_review",
                "session_id": schedule_id,
                "gemma_feedback": {
                    "final_summary": dict(post_review_gemma_feedback or {}).get("final_summary", ""),
                    "reason": dict(post_review_gemma_feedback or {}).get("reason", ""),
                    "needs_regeneration": dict(post_review_gemma_feedback or {}).get("needs_regeneration", False),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=current_constraints),
            }
            qwen_feedback = qwen_review_plan(review_payload)
            qwen_feedback_history.append(dict(qwen_feedback or {}))

        post_review_entry = {
            "phase": "post_qwen_refinement",
            "constraint_diff": _constraints_diff(before_constraints, current_constraints),
            "coverage_diff": _coverage_diff(before_coverage, coverage),
        }
        if not post_review_entry["constraint_diff"] and not post_review_entry["coverage_diff"]:
            post_review_entry["note"] = "no_material_changes"
        round_change_log.append(post_review_entry)

        gemma_feedback = dict(post_review_gemma_feedback or {})
        _op_log(
            operation_id,
            "Post-Qwen Gemma refinement completed",
            metadata={"regenerated": bool(should_regenerate or constraints_changed)},
        )
    elif post_review_reasons and gemma_agent is None:
        _op_log(operation_id, "Skipped post-Qwen refinement because Gemma is unavailable", level="warning")

    if feedback_text:
        if feedback_text not in feedback_history_rows:
            feedback_history_rows.append(feedback_text)

        _op_log(operation_id, "Applying user feedback revision pass", metadata={"feedback_round": 1})

        user_revision_feedback: dict[str, Any] = {}
        if gemma_agent is not None:
            user_revision_context = {
                "phase": "user_feedback_revision",
                "feedback_round": 1,
                "session_id": schedule_id,
                "locked_constraints": sorted(list(locked_constraint_keys)),
                "user_feedback": feedback_text,
                "feedback_history": list(feedback_history_rows),
                "changes_since_last_round": round_change_log[-1] if round_change_log else {},
                "round_change_log": round_change_log[-5:],
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "schedule_end": schedule_end.isoformat(),
                },
                "constraints": current_constraints,
                "coverage": {
                    "coverage_pct": coverage.get("coverage_pct"),
                    "uncovered_chunks": coverage.get("uncovered_chunks"),
                    "partial_chunks": coverage.get("partially_covered_chunks"),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=current_constraints),
            }

            user_revision_toolbox = PlanningToolbox(
                chunks=chunks,
                slots=slots,
                coverage=coverage,
                constraints=current_constraints,
            )
            user_revision_feedback = dict(
                gemma_agent.run(
                    session_context=user_revision_context,
                    toolbox=user_revision_toolbox,
                    max_turns=max(2, GEMMA_PLANNER_MAX_TURNS - 1),
                    allow_user_input=False,
                )
                or {}
            )
            gemma_feedback_history.append(dict(user_revision_feedback))
            question = _extract_pending_clarification_question(user_revision_feedback)
            if question:
                gemma_clarification_question = question

        feedback_constraint_updates = _extract_constraint_updates(user_revision_feedback)

        revised_constraints = apply_constraint_updates(
            current=current_constraints,
            updates=feedback_constraint_updates,
            locked_keys=locked_constraint_keys,
        )

        before_constraints = dict(current_constraints)
        before_coverage = dict(coverage)
        chunk_updates = user_revision_feedback.get("updated_chunk_prerequisites", [])
        chunks_changed = apply_chunk_prerequisite_updates(chunks, chunk_updates)
        constraints_changed = revised_constraints != current_constraints

        if constraints_changed:
            current_constraints = revised_constraints

        if constraints_changed or chunks_changed or bool(user_revision_feedback.get("needs_regeneration", False)):
            slots, current_constraints, coverage, schedule_end, _ = _build_schedule_once(
                chunks=chunks,
                constraints=current_constraints,
                start_date=start_date,
                end_date=end_date,
                tzinfo=tzinfo,
                calendar_service=calendar_service,
                calendar_id=calendar_id,
            )

            review_payload = {
                "phase": "user_feedback_round_1_qwen_review",
                "session_id": schedule_id,
                "gemma_feedback": {
                    "final_summary": user_revision_feedback.get("final_summary", ""),
                    "reason": user_revision_feedback.get("reason", ""),
                    "needs_regeneration": user_revision_feedback.get("needs_regeneration", False),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=current_constraints),
                "user_feedback": feedback_text,
                "feedback_history": list(feedback_history_rows),
            }
            qwen_feedback = qwen_review_plan(review_payload)
            qwen_feedback_history.append(dict(qwen_feedback or {}))

            escalation_result = _apply_qwen_escalation_if_needed(
                qwen_feedback=qwen_feedback,
                constraints=current_constraints,
                locked_constraint_keys=locked_constraint_keys,
                chunks=chunks,
                start_date=start_date,
                end_date=end_date,
                schedule_end=schedule_end,
                tzinfo=tzinfo,
                calendar_service=calendar_service,
                calendar_id=calendar_id,
            )
            if escalation_result.get("escalated"):
                current_constraints = dict(escalation_result.get("constraints", current_constraints))
                escalated_slots = escalation_result.get("slots")
                if escalated_slots is not None:
                    slots = list(escalated_slots)
                escalated_coverage = escalation_result.get("coverage")
                if escalated_coverage is not None:
                    coverage = dict(escalated_coverage)
                qwen_feedback = dict(escalation_result.get("qwen_feedback", qwen_feedback))
                schedule_end = escalation_result.get("schedule_end", schedule_end)
                qwen_feedback_history.append(dict(qwen_feedback or {}))

        if not qwen_feedback_history:
            review_payload = {
                "phase": "user_feedback_round_1_qwen_review",
                "session_id": schedule_id,
                "gemma_feedback": {
                    "final_summary": user_revision_feedback.get("final_summary", ""),
                    "reason": user_revision_feedback.get("reason", ""),
                    "needs_regeneration": user_revision_feedback.get("needs_regeneration", False),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=current_constraints),
                "user_feedback": feedback_text,
                "feedback_history": list(feedback_history_rows),
            }
            _op_log(operation_id, "Running Qwen review", metadata={"phase": review_payload.get("phase")})
            qwen_feedback = qwen_review_plan(review_payload)
            qwen_feedback_history.append(dict(qwen_feedback or {}))
            _op_log(
                operation_id,
                "Qwen review completed",
                metadata={
                    "severity": _normalize_int(dict(qwen_feedback or {}).get("severity"), 1, minimum=0, maximum=3),
                    "approval_ready": bool(dict(qwen_feedback or {}).get("approval_ready", False)),
                },
            )

            escalation_result = _apply_qwen_escalation_if_needed(
                qwen_feedback=qwen_feedback,
                constraints=current_constraints,
                locked_constraint_keys=locked_constraint_keys,
                chunks=chunks,
                start_date=start_date,
                end_date=end_date,
                schedule_end=schedule_end,
                tzinfo=tzinfo,
                calendar_service=calendar_service,
                calendar_id=calendar_id,
            )
            if escalation_result.get("escalated"):
                current_constraints = dict(escalation_result.get("constraints", current_constraints))
                escalated_slots = escalation_result.get("slots")
                if escalated_slots is not None:
                    slots = list(escalated_slots)
                escalated_coverage = escalation_result.get("coverage")
                if escalated_coverage is not None:
                    coverage = dict(escalated_coverage)
                qwen_feedback = dict(escalation_result.get("qwen_feedback", qwen_feedback))
                schedule_end = escalation_result.get("schedule_end", schedule_end)
                qwen_feedback_history.append(dict(qwen_feedback or {}))

        feedback_entry = {
            "phase": "user_feedback_round_1",
            "constraint_diff": _constraints_diff(before_constraints, current_constraints),
            "coverage_diff": _coverage_diff(before_coverage, coverage),
        }
        if not feedback_entry["constraint_diff"] and not feedback_entry["coverage_diff"]:
            feedback_entry["note"] = "no_material_changes"
        round_change_log.append(feedback_entry)

        gemma_feedback = dict(user_revision_feedback or {})
        _op_log(
            operation_id,
            "User feedback revision completed",
            metadata={
                "constraint_changes": len(feedback_entry["constraint_diff"]),
                "coverage_changed": bool(feedback_entry["coverage_diff"]),
            },
        )

    stored_constraints = dict(current_constraints)
    stored_constraints["start_date"] = start_date.isoformat()
    stored_constraints["end_date"] = end_date.isoformat()
    stored_constraints["timezone_name"] = timezone_name
    stored_constraints["calendar_id"] = calendar_id
    stored_constraints["calendar_mode"] = calendar_mode
    if calendar_warning:
        stored_constraints["calendar_warning"] = calendar_warning

    stored_constraints["qwen_feedback"] = qwen_feedback

    model_notes: dict[str, Any] = {
        "gemma": gemma_feedback,
        "qwen": qwen_feedback,
        "gemma_feedback_history": gemma_feedback_history,
        "qwen_feedback_history": qwen_feedback_history,
        "round_change_log": round_change_log[-12:],
        "post_review_reasons": post_review_reasons,
    }
    if feedback_text:
        model_notes["user_feedback"] = feedback_text
        model_notes["feedback_history"] = list(feedback_history_rows)
    if feedback_constraint_updates:
        model_notes["feedback_constraint_updates"] = dict(feedback_constraint_updates)
    if gemma_warning:
        model_notes["gemma_warning"] = gemma_warning
    if gemma_clarification_question:
        model_notes["clarification_question"] = gemma_clarification_question
    stored_constraints["model_notes"] = model_notes

    if gemma_clarification_question:
        stored_constraints["user_feedback_requested"] = True
        stored_constraints["feedback_prompt"] = gemma_clarification_question
        stored_constraints["feedback_source"] = "gemma_clarification"
        stored_constraints["clarification_question"] = gemma_clarification_question
        _op_log(
            operation_id,
            "Gemma requested user clarification before confirmation",
            metadata={"question": gemma_clarification_question},
        )
    elif _needs_user_feedback(qwen_feedback):
        stored_constraints["user_feedback_requested"] = True
        stored_constraints["feedback_prompt"] = _suggest_feedback_prompt(qwen_feedback)
        stored_constraints["feedback_source"] = "qwen_review"
        stored_constraints["clarification_question"] = ""
        _op_log(operation_id, "Plan requires user feedback before confirmation", metadata={"severity": stored_constraints["qwen_feedback"].get("severity")})
    else:
        stored_constraints["user_feedback_requested"] = False
        stored_constraints["feedback_prompt"] = ""
        stored_constraints["feedback_source"] = ""
        stored_constraints["clarification_question"] = ""

    return slots, stored_constraints, coverage


def generate_draft_plan(
    db: Session,
    schedule_id: str,
    constraints: dict,
    user_id: str | None = None,
    operation_id: str | None = None,
    user_feedback: str = "",
    feedback_history: list[str] | None = None,
) -> dict:
    
    latest_version = db.scalar(
        select(StudyPlan)
        .where(StudyPlan.schedule_id == schedule_id)
        .order_by(StudyPlan.created_at.desc())
    )

    next_version = 1
    if latest_version and hasattr(latest_version, "version"):
        next_version = (latest_version.version or 0) + 1

    # ---------------------------------------------------------
    # START REPLACEMENT BLOCK
    # ---------------------------------------------------------
    effective_user_feedback = str(user_feedback or "").strip()
    effective_feedback_history = _coerce_feedback_history(feedback_history)
    
    if effective_user_feedback:
        # Inject the raw text into additional_constraints so Gemma sees it on pass 1
        existing_add = str(constraints.get("additional_constraints") or "").strip()
        constraints["additional_constraints"] = f"{existing_add} | Initial Request: {effective_user_feedback}".strip(" |")
        
    # Unconditionally wipe the feedback variables so the pipeline doesn't trigger an immediate revision loop
    effective_user_feedback = ""
    effective_feedback_history = []

    sink_token = set_planner_runtime_log_sink(_build_planner_runtime_sink(operation_id))
    # ---------------------------------------------------------
    # END REPLACEMENT BLOCK
    # ---------------------------------------------------------

    try:
        slots, stored_constraints, coverage = _run_planner_pipeline(
            db=db,
            schedule_id=schedule_id,
            constraints=constraints,
            user_id=user_id,
            operation_id=operation_id,
            run_main_passes=True,
            user_feedback=effective_user_feedback,
            feedback_history=effective_feedback_history,
        )
    finally:
        reset_planner_runtime_log_sink(sink_token)

    plan = StudyPlan(
        schedule_id=schedule_id,
        sessions_payload=slots,
        constraints_json={
            **stored_constraints,
            "coverage": coverage,
        },
        status="draft",
        version=next_version,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return _serialize_plan(plan)


def get_current_plan(db: Session, schedule_id: str) -> dict | None:
    plan = db.scalar(
        select(StudyPlan)
        .where(
            StudyPlan.schedule_id == schedule_id,
            StudyPlan.status == "active",
        )
        .order_by(StudyPlan.updated_at.desc())
    )

    if plan is None:
        plan = db.scalar(
            select(StudyPlan)
            .where(StudyPlan.schedule_id == schedule_id)
            .order_by(StudyPlan.updated_at.desc())
        )
    if plan is None:
        return None
    return _serialize_plan(plan)


def list_plans_for_schedule(db: Session, schedule_id: str) -> list[dict]:
    rows = db.scalars(
        select(StudyPlan)
        .where(StudyPlan.schedule_id == schedule_id)
        .order_by(
        StudyPlan.status.desc(),  # active first
        StudyPlan.updated_at.desc()
    ))
    return [_serialize_plan(row) for row in rows]


def _plan_has_synced_calendar_events(plan: StudyPlan) -> bool:
    for row in list(plan.sessions_payload or []):
        if not isinstance(row, dict):
            continue

        calendar_event_id = str(row.get("calendar_event_id") or "").strip()
        calendar_status = str(row.get("calendar_status") or "").strip().lower()
        if calendar_event_id:
            return True
        if calendar_status == "created":
            return True

    return False


def delete_plan(db: Session, schedule_id: str, plan_id: str) -> dict:
    plan = db.scalar(
        select(StudyPlan).where(
            StudyPlan.id == plan_id,
            StudyPlan.schedule_id == schedule_id,
        )
    )
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")
    if _plan_has_synced_calendar_events(plan):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This plan is synced to Google Calendar. Unsync it first before deleting.",
        )

    materialized_session_ids = db.scalars(
        select(StudySession.id).where(
            StudySession.schedule_id == schedule_id,
            StudySession.plan_id == plan_id,
        )
    ).all()

    deleted_session_chat_messages = 0
    if materialized_session_ids:
        chat_result = db.execute(
            delete(SessionChatMessage).where(SessionChatMessage.session_id.in_(list(materialized_session_ids)))
        )
        deleted_session_chat_messages = int(chat_result.rowcount or 0)

    session_result = db.execute(
        delete(StudySession).where(
            StudySession.schedule_id == schedule_id,
            StudySession.plan_id == plan_id,
        )
    )
    deleted_materialized_sessions = int(session_result.rowcount or 0)

    deleted_plan_status = str(plan.status or "")
    db.delete(plan)
    db.commit()

    return {
        "deleted_plan_id": plan_id,
        "deleted_plan_status": deleted_plan_status,
        "schedule_id": schedule_id,
        "deleted_materialized_sessions": deleted_materialized_sessions,
        "deleted_session_chat_messages": deleted_session_chat_messages,
    }


def delete_all_plans(db: Session, schedule_id: str) -> dict:
    plans = db.scalars(
        select(StudyPlan).where(StudyPlan.schedule_id == schedule_id)
    ).all()
    synced_plan_ids = [str(plan.id) for plan in plans if _plan_has_synced_calendar_events(plan)]
    if synced_plan_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more plans are synced to Google Calendar. Unsync them first before deleting.",
        )

    plan_ids = db.scalars(
        select(StudyPlan.id).where(StudyPlan.schedule_id == schedule_id)
    ).all()

    materialized_session_ids = db.scalars(
        select(StudySession.id).where(StudySession.schedule_id == schedule_id)
    ).all()

    deleted_session_chat_messages = 0
    if materialized_session_ids:
        chat_result = db.execute(
            delete(SessionChatMessage).where(SessionChatMessage.session_id.in_(list(materialized_session_ids)))
        )
        deleted_session_chat_messages = int(chat_result.rowcount or 0)

    session_result = db.execute(delete(StudySession).where(StudySession.schedule_id == schedule_id))
    deleted_materialized_sessions = int(session_result.rowcount or 0)

    plan_result = db.execute(delete(StudyPlan).where(StudyPlan.schedule_id == schedule_id))
    deleted_plans = int(plan_result.rowcount or 0)

    db.commit()

    return {
        "schedule_id": schedule_id,
        "deleted_plans": deleted_plans,
        "deleted_plan_ids": [str(plan_id) for plan_id in plan_ids],
        "deleted_materialized_sessions": deleted_materialized_sessions,
        "deleted_session_chat_messages": deleted_session_chat_messages,
    }


def revise_plan_with_feedback(
    db: Session,
    schedule_id: str,
    feedback: str,
    user_id: str | None = None,
    plan_id: str | None = None,
    operation_id: str | None = None,
    feedback_history: list[str] | None = None,
) -> dict:
    if plan_id:
        plan = db.scalar(
            select(StudyPlan).where(
                StudyPlan.id == plan_id,
                StudyPlan.schedule_id == schedule_id,
            )
        )
    else:
        plan = db.scalar(
            select(StudyPlan)
            .where(StudyPlan.schedule_id == schedule_id)
            .order_by(StudyPlan.updated_at.desc())
        )
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No plan found for this schedule")

    current_constraints = dict(plan.constraints_json or {})
    merged_feedback_history = _coerce_feedback_history(feedback_history)
    feedback_text = str(feedback or "").strip()
    if feedback_text and feedback_text not in merged_feedback_history:
        merged_feedback_history.append(feedback_text)

    current_constraints, merged_feedback_history, clarification_updates = _apply_pending_clarification_to_revision(
        constraints=current_constraints,
        feedback_text=feedback_text,
        feedback_history=merged_feedback_history,
    )

    direct_daily_override = _extract_direct_daily_override_from_feedback(
        feedback_text=feedback_text,
        constraints=current_constraints,
    )
    if direct_daily_override is not None:
        current_constraints["daily_max_minutes"] = direct_daily_override
        clarification_updates["daily_max_minutes"] = direct_daily_override

    if clarification_updates:
        _op_log(
            operation_id,
            "Applied clarification response before revision",
            metadata=clarification_updates,
        )

    sink_token = set_planner_runtime_log_sink(_build_planner_runtime_sink(operation_id))
    try:
        slots, stored_constraints, coverage = _run_planner_pipeline(
            db=db,
            schedule_id=schedule_id,
            constraints=current_constraints,
            user_id=user_id,
            operation_id=operation_id,
            run_main_passes=False,
            user_feedback=feedback_text,
            feedback_history=merged_feedback_history,
        )
    finally:
        reset_planner_runtime_log_sink(sink_token)

    model_notes = dict(stored_constraints.get("model_notes") or {})
    feedback_updates = dict(model_notes.get("feedback_constraint_updates") or {})

    plan.sessions_payload = slots
    plan.constraints_json = {
        **stored_constraints,
        "coverage": coverage,
        "feedback": feedback,
        "feedback_constraint_updates": feedback_updates,
    }
    plan.status = "draft"

    db.commit()
    db.refresh(plan)
    return _serialize_plan(plan)


def confirm_plan(db: Session, schedule_id: str, plan_id: str | None = None) -> dict:
    if plan_id:
        plan = db.scalar(
            select(StudyPlan).where(
                StudyPlan.id == plan_id,
                StudyPlan.schedule_id == schedule_id,
            )
        )
    else:
        plan = db.scalar(
            select(StudyPlan)
            .where(StudyPlan.schedule_id == schedule_id)
            .order_by(StudyPlan.updated_at.desc())
        )

    if plan is None:
        plan = db.scalar(
            select(StudyPlan)
            .where(StudyPlan.schedule_id == schedule_id)
            .order_by(StudyPlan.updated_at.desc())
        )

    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No plan available to confirm")

    current_active = db.scalars(
        select(StudyPlan).where(
            StudyPlan.schedule_id == schedule_id,
            StudyPlan.status == "active",
        )
    ).all()
    for active in current_active:
        active.status = "confirmed"

    db.query(StudySession).filter(StudySession.schedule_id == schedule_id).delete(synchronize_session=False)

    materialized = 0
    for idx, slot in enumerate(list(plan.sessions_payload or []), 1):
        start_dt = _parse_iso(slot.get("start_time"))
        end_dt = _parse_iso(slot.get("end_time"))
        slot_items = list(slot.get("items", []))

        focus_chunks = []
        for item in slot_items:
            focus_chunks.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "topic": item.get("topic"),
                    "focus_points": list(item.get("focus_topics") or []),
                }
            )

        study_session = StudySession(
            plan_id=plan.id,
            schedule_id=schedule_id,
            session_number=idx,
            title=str(slot.get("title") or f"Session {idx}"),
            scheduled_date=start_dt.date() if start_dt else None,
            start_time=start_dt.time() if start_dt else None,
            end_time=end_dt.time() if end_dt else None,
            focus_chunks_json=focus_chunks,
            prerequisites_json=list(slot.get("prerequisites") or []),
            status="upcoming",
            generated_briefing="",
            briefing_status="pending",
        )
        db.add(study_session)
        materialized += 1

    plan.status = "active"
    db.commit()

    return {
        "plan_id": plan.id,
        "status": plan.status,
        "materialized_sessions": materialized,
    }


def _resolve_target_plan(db: Session, schedule_id: str, plan_id: str | None = None) -> StudyPlan | None:
    if str(plan_id or "").strip():
        return db.scalar(
            select(StudyPlan).where(
                StudyPlan.id == str(plan_id).strip(),
                StudyPlan.schedule_id == schedule_id,
            )
        )

    target_plan = db.scalar(
        select(StudyPlan)
        .where(
            StudyPlan.schedule_id == schedule_id,
            StudyPlan.status == "active",
        )
        .order_by(StudyPlan.updated_at.desc())
    )
    if target_plan is None:
        target_plan = db.scalar(
            select(StudyPlan)
            .where(
                StudyPlan.schedule_id == schedule_id,
                StudyPlan.status == "draft",
            )
            .order_by(StudyPlan.updated_at.desc())
        )
    if target_plan is None:
        target_plan = db.scalar(
            select(StudyPlan)
            .where(StudyPlan.schedule_id == schedule_id)
            .order_by(StudyPlan.updated_at.desc())
        )
    return target_plan


def _load_calendar_credentials_for_user(db: Session, user_id: str) -> dict:
    row = db.scalar(select(GoogleCalendarCredential).where(GoogleCalendarCredential.user_id == user_id))
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Calendar is not connected for this user. Connect it from auth endpoint first.",
        )

    try:
        payload = decrypt_calendar_credentials(row.encrypted_credentials)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stored Google Calendar credential is invalid. Reconnect Google Calendar.",
        ) from exc

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stored Google Calendar credential is empty. Reconnect Google Calendar.",
        )

    return payload


def _persist_calendar_credentials_for_user(
    db: Session,
    *,
    user_id: str,
    credentials_payload: dict,
    google_account_id: str,
    google_account_email: str,
) -> None:
    payload = dict(credentials_payload or {})
    if not payload:
        return

    row = db.scalar(select(GoogleCalendarCredential).where(GoogleCalendarCredential.user_id == user_id))
    if row is None:
        row = GoogleCalendarCredential(
            user_id=user_id,
            encrypted_credentials=encrypt_calendar_credentials(payload),
            scopes_json=list(payload.get("scopes") or []),
            google_account_id=google_account_id,
            google_account_email=google_account_email,
        )
        db.add(row)
    else:
        row.encrypted_credentials = encrypt_calendar_credentials(payload)
        row.scopes_json = list(payload.get("scopes") or [])
        row.google_account_id = google_account_id
        row.google_account_email = google_account_email

    db.commit()


def _verify_calendar_identity(
    calendar_service: GoogleCalendarService,
    *,
    expected_google_id: str | None = None,
    expected_email: str | None = None,
) -> None:
    normalized_google_id = str(expected_google_id or "").strip()
    normalized_email = str(expected_email or "").strip().lower()
    if not normalized_google_id and not normalized_email:
        return

    identity = dict(calendar_service.get_authenticated_identity() or {})
    actual_google_id = str(identity.get("google_id") or "").strip()
    actual_email = str(identity.get("email") or "").strip().lower()

    if not actual_email:
        primary_calendar_id = str(identity.get("primary_calendar_id") or "").strip().lower()
        if "@" in primary_calendar_id:
            actual_email = primary_calendar_id

    google_id_matches = bool(normalized_google_id and actual_google_id and normalized_google_id == actual_google_id)
    email_matches = bool(normalized_email and actual_email and normalized_email == actual_email)

    if normalized_google_id and normalized_email:
        if google_id_matches or email_matches:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Google Calendar account mismatch for authenticated user",
        )

    if normalized_google_id and not google_id_matches:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Google Calendar account mismatch for authenticated user",
        )

    if normalized_email and not email_matches:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Google Calendar account mismatch for authenticated user",
        )


def sync_plan_sessions_to_calendar(
    db: Session,
    schedule_id: str,
    *,
    user_id: str,
    plan_id: str | None = None,
    calendar_id: str | None = None,
    timezone_name: str | None = None,
    skip_existing: bool = True,
    expected_google_id: str | None = None,
    expected_email: str | None = None,
) -> dict:
    target_plan = _resolve_target_plan(db, schedule_id, plan_id=plan_id)

    if target_plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No plan found for this schedule")

    slots = list(target_plan.sessions_payload or [])
    if not slots:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Plan has no sessions to sync")

    constraints = dict(target_plan.constraints_json or {})
    resolved_calendar_id = str(calendar_id or constraints.get("calendar_id") or "primary").strip() or "primary"
    resolved_timezone_name = _normalize_timezone_name(timezone_name or constraints.get("timezone_name") or "Asia/Kolkata")
    credential_payload = _load_calendar_credentials_for_user(db, user_id)

    try:
        calendar_service = GoogleCalendarService(
            timezone_name=resolved_timezone_name,
            credentials_info=credential_payload,
            allow_local_oauth=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to initialize Google Calendar service: {exc}",
        ) from exc

    _verify_calendar_identity(
        calendar_service,
        expected_google_id=expected_google_id,
        expected_email=expected_email,
    )
    identity = dict(calendar_service.get_authenticated_identity() or {})

    prepared_rows: list[dict[str, Any]] = []
    slot_lookup: dict[str, dict] = {}
    skipped_existing = 0
    invalid_rows = 0
    skipped_results: list[dict[str, Any]] = []

    for idx, row in enumerate(slots, 1):
        if not isinstance(row, dict):
            invalid_rows += 1
            skipped_results.append(
                {
                    "slot_id": f"slot-{idx}",
                    "event_id": None,
                    "status": "failed: invalid_slot_payload",
                }
            )
            continue

        slot_id = str(row.get("slot_id") or "").strip()
        if not slot_id:
            slot_id = f"slot-{idx}"
            row["slot_id"] = slot_id
        slot_lookup[slot_id] = row

        existing_event_id = str(row.get("calendar_event_id") or "").strip()
        existing_status = str(row.get("calendar_status") or "").strip().lower()
        if skip_existing and existing_event_id and existing_status == "created":
            skipped_existing += 1
            skipped_results.append(
                {
                    "slot_id": slot_id,
                    "event_id": existing_event_id,
                    "status": "skipped: already_synced",
                }
            )
            continue

        start_time = str(row.get("start_time") or "").strip()
        end_time = str(row.get("end_time") or "").strip()
        if not start_time or not end_time:
            invalid_rows += 1
            row["calendar_status"] = "failed: missing_start_or_end_time"
            skipped_results.append(
                {
                    "slot_id": slot_id,
                    "event_id": None,
                    "status": "failed: missing_start_or_end_time",
                }
            )
            continue

        items = list(row.get("items") or [])
        title = str(row.get("title") or "").strip() or _build_slot_title(items)
        description = str(row.get("description") or "").strip()
        if not description:
            description = _build_slot_description({**row, "items": items})

        row["title"] = title
        row["description"] = description

        prepared_rows.append(
            {
                "slot_id": slot_id,
                "start_time": start_time,
                "end_time": end_time,
                "title": title,
                "description": description,
            }
        )

    create_results = []
    if prepared_rows:
        create_results = list(calendar_service.create_events(calendar_id=resolved_calendar_id, slots=prepared_rows) or [])

    created_count = 0
    failed_count = 0
    for row in create_results:
        slot_id = str(row.get("slot_id") or "").strip()
        status_text = str(row.get("status") or "unknown")
        event_id = str(row.get("event_id") or "").strip() or None

        slot = slot_lookup.get(slot_id)
        if slot is not None:
            slot["calendar_status"] = status_text
            if event_id:
                slot["calendar_event_id"] = event_id

        if status_text == "created":
            created_count += 1
        else:
            failed_count += 1

    constraints["calendar_id"] = resolved_calendar_id
    constraints["timezone_name"] = resolved_timezone_name
    constraints["calendar_mode"] = "real"

    target_plan.sessions_payload = slots
    target_plan.constraints_json = constraints
    
    # Add these two lines before db.commit()
    flag_modified(target_plan, "sessions_payload")
    flag_modified(target_plan, "constraints_json")
    
    db.commit()
    db.refresh(target_plan)

    _persist_calendar_credentials_for_user(
        db,
        user_id=user_id,
        credentials_payload=calendar_service.get_credentials_payload(),
        google_account_id=str(identity.get("google_id") or "").strip(),
        google_account_email=str(identity.get("email") or "").strip().lower(),
    )

    all_results = skipped_results + create_results
    return {
        "plan_id": target_plan.id,
        "schedule_id": schedule_id,
        "calendar_id": resolved_calendar_id,
        "timezone_name": resolved_timezone_name,
        "attempted": len(prepared_rows),
        "created": created_count,
        "failed": failed_count + invalid_rows,
        "skipped_existing": skipped_existing,
        "results": all_results,
    }


def unsync_plan_sessions_from_calendar(
    db: Session,
    schedule_id: str,
    *,
    user_id: str,
    plan_id: str | None = None,
    calendar_id: str | None = None,
    timezone_name: str | None = None,
    expected_google_id: str | None = None,
    expected_email: str | None = None,
) -> dict:
    target_plan = _resolve_target_plan(db, schedule_id, plan_id=plan_id)
    if target_plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No plan found for this schedule")

    slots = list(target_plan.sessions_payload or [])
    if not slots:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Plan has no sessions to unsync")

    constraints = dict(target_plan.constraints_json or {})
    resolved_calendar_id = str(calendar_id or constraints.get("calendar_id") or "primary").strip() or "primary"
    resolved_timezone_name = _normalize_timezone_name(timezone_name or constraints.get("timezone_name") or "Asia/Kolkata")
    credential_payload = _load_calendar_credentials_for_user(db, user_id)

    try:
        calendar_service = GoogleCalendarService(
            timezone_name=resolved_timezone_name,
            credentials_info=credential_payload,
            allow_local_oauth=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to initialize Google Calendar service: {exc}",
        ) from exc

    _verify_calendar_identity(
        calendar_service,
        expected_google_id=expected_google_id,
        expected_email=expected_email,
    )
    identity = dict(calendar_service.get_authenticated_identity() or {})

    event_slot_rows: dict[str, list[dict]] = {}
    event_slot_ids: dict[str, list[str]] = {}
    unique_event_ids: list[str] = []
    seen_event_ids: set[str] = set()
    skipped_results: list[dict[str, Any]] = []
    invalid_rows = 0
    skipped_not_synced = 0

    for idx, row in enumerate(slots, 1):
        if not isinstance(row, dict):
            invalid_rows += 1
            skipped_results.append(
                {
                    "slot_id": f"slot-{idx}",
                    "event_id": None,
                    "status": "failed: invalid_slot_payload",
                }
            )
            continue

        slot_id = str(row.get("slot_id") or "").strip()
        if not slot_id:
            slot_id = f"slot-{idx}"
            row["slot_id"] = slot_id

        event_id = str(row.get("calendar_event_id") or "").strip()
        if not event_id:
            skipped_not_synced += 1
            row["calendar_status"] = "skipped: no_calendar_event_id"
            skipped_results.append(
                {
                    "slot_id": slot_id,
                    "event_id": None,
                    "status": "skipped: no_calendar_event_id",
                }
            )
            continue

        event_slot_rows.setdefault(event_id, []).append(row)
        event_slot_ids.setdefault(event_id, []).append(slot_id)
        if event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            unique_event_ids.append(event_id)

    delete_results = []
    if unique_event_ids:
        delete_results = list(calendar_service.delete_events(calendar_id=resolved_calendar_id, event_ids=unique_event_ids) or [])

    deleted_count = 0
    failed_count = 0
    normalized_delete_results: list[dict[str, Any]] = []

    for row in delete_results:
        event_id = str(row.get("event_id") or "").strip()
        status_text = str(row.get("status") or "unknown")

        matched_slots = event_slot_rows.get(event_id, [])
        matched_slot_ids = event_slot_ids.get(event_id, [])
        if not matched_slots:
            continue

        is_deleted = status_text == "deleted"
        for slot_index, slot_row in enumerate(matched_slots):
            slot_id = matched_slot_ids[slot_index] if slot_index < len(matched_slot_ids) else ""
            if is_deleted:
                slot_row["calendar_status"] = "deleted"
                slot_row.pop("calendar_event_id", None)
                deleted_count += 1
            else:
                slot_row["calendar_status"] = status_text
                failed_count += 1

            normalized_delete_results.append(
                {
                    "slot_id": slot_id,
                    "event_id": event_id,
                    "status": status_text,
                }
            )

    constraints["calendar_id"] = resolved_calendar_id
    constraints["timezone_name"] = resolved_timezone_name
    constraints["calendar_mode"] = "real"

    target_plan.sessions_payload = slots
    target_plan.constraints_json = constraints
    
    flag_modified(target_plan, "sessions_payload")
    flag_modified(target_plan, "constraints_json")
    
    db.commit()
    db.refresh(target_plan)

    _persist_calendar_credentials_for_user(
        db,
        user_id=user_id,
        credentials_payload=calendar_service.get_credentials_payload(),
        google_account_id=str(identity.get("google_id") or "").strip(),
        google_account_email=str(identity.get("email") or "").strip().lower(),
    )

    all_results = skipped_results + normalized_delete_results
    return {
        "plan_id": target_plan.id,
        "schedule_id": schedule_id,
        "calendar_id": resolved_calendar_id,
        "timezone_name": resolved_timezone_name,
        "attempted": sum(len(rows) for rows in event_slot_rows.values()),
        "deleted": deleted_count,
        "failed": failed_count + invalid_rows,
        "skipped_not_synced": skipped_not_synced,
        "results": all_results,
    }


def list_materialized_sessions(db: Session, schedule_id: str) -> list[dict]:
    rows = db.scalars(
        select(StudySession)
        .where(StudySession.schedule_id == schedule_id)
        .order_by(StudySession.session_number.asc())
    ).all()

    out = []
    for row in rows:
        focus_topics = []
        seen: set[str] = set()
        for item in list(row.focus_chunks_json or []):
            topic = str((item or {}).get("topic") or "").strip()
            if topic and topic not in seen:
                seen.add(topic)
                focus_topics.append(topic)
        out.append(
            {
                "id": row.id,
                "session_number": row.session_number,
                "title": row.title,
                "scheduled_date": row.scheduled_date.isoformat() if row.scheduled_date else None,
                "start_time": row.start_time.isoformat() if row.start_time else None,
                "end_time": row.end_time.isoformat() if row.end_time else None,
                "focus_topics": focus_topics,
                "status": row.status,
                "briefing_status": row.briefing_status,
            }
        )
    return out


def get_materialized_session_detail(db: Session, schedule_id: str, session_id: str) -> dict:
    study_session = db.scalar(
        select(StudySession).where(
            StudySession.id == session_id,
            StudySession.schedule_id == schedule_id,
        )
    )
    if study_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study session not found")

    focus_chunks = list(study_session.focus_chunks_json or [])
    focus_topics = []
    seen: set[str] = set()
    enriched_focus = []

    for item in focus_chunks:
        chunk_id = str(item.get("chunk_id") or "").strip()
        chunk = db.get(Chunk, chunk_id) if chunk_id else None
        topic = str(item.get("topic") or "").strip()
        if topic and topic not in seen:
            seen.add(topic)
            focus_topics.append(topic)
        preview = ""
        if chunk is not None:
            text = str(chunk.content or "").strip().replace("\n", " ")
            preview = text[:220] + ("..." if len(text) > 220 else "")

        enriched_focus.append(
            {
                "chunk_id": chunk_id,
                "topic": item.get("topic"),
                "focus_points": list(item.get("focus_points") or []),
                "content_preview": preview,
            }
        )

    upcoming_rows = db.scalars(
        select(StudySession)
        .where(
            StudySession.schedule_id == schedule_id,
            StudySession.session_number > study_session.session_number,
        )
        .order_by(StudySession.session_number.asc())
        .limit(3)
    ).all()

    upcoming = [
        {
            "id": row.id,
            "session_number": row.session_number,
            "title": row.title,
            "scheduled_date": row.scheduled_date.isoformat() if row.scheduled_date else None,
            "status": row.status,
        }
        for row in upcoming_rows
    ]

    return {
        "id": study_session.id,
        "plan_id": study_session.plan_id,
        "schedule_id": study_session.schedule_id,
        "session_number": study_session.session_number,
        "title": study_session.title,
        "scheduled_date": study_session.scheduled_date.isoformat() if study_session.scheduled_date else None,
        "start_time": study_session.start_time.isoformat() if study_session.start_time else None,
        "end_time": study_session.end_time.isoformat() if study_session.end_time else None,
        "status": study_session.status,
        "focus_topics": focus_topics,
        "focus_chunks": enriched_focus,
        "prerequisites": list(study_session.prerequisites_json or []),
        "upcoming_sessions": upcoming,
        "briefing": study_session.generated_briefing,
        "briefing_status": study_session.briefing_status,
    }
