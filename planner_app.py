import json
from datetime import date, time, timedelta

from db import DocumentDB
from planner_ai import (
    GemmaPlannerAgent,
    _apply_qwen_escalation_if_needed,
    _post_review_refinement_reasons,
    _plan_snapshot,
    apply_chunk_prerequisite_updates,
    apply_constraint_updates,
    collect_additional_plan_feedback,
    collect_user_plan_feedback,
    derive_constraint_updates_from_user_feedback,
    qwen_review_plan,
)
from planner_calendar import GoogleCalendarService
from planner_common import (
    DEFAULT_CONSTRAINTS,
    GEMMA_FEEDBACK_MAX_ROUNDS,
    GEMMA_PLANNER_MAX_TURNS,
    _constraints_diff,
    _coverage_diff,
    _locked_constraint_keys,
    _normalize_bool,
    _normalize_constraints,
    _normalize_int,
    _normalize_optional_int,
    _parse_date_str,
    _parse_time_hhmm,
)
from planner_persistence import persist_draft, print_plan_summary, sync_to_calendar_and_mark_approved
from planner_scheduling import build_schedule_data, load_session_chunks
from planner_toolbox import PlanningToolbox


def choose_session(db):
    sessions = db.list_sessions()
    if not sessions:
        print("No sessions found. Ingest documents first.")
        return None

    print("\nAvailable sessions:\n")
    for idx, row in enumerate(sessions):
        session_id, title, created_at, doc_count = row
        print(f"[{idx}] {session_id} | {title or 'Untitled'} | docs={doc_count} | created={created_at}")

    choice = input("Select session index: ").strip()
    index = _normalize_int(choice or 0, 0, minimum=0, maximum=len(sessions) - 1)
    return sessions[index][0]


def ask_user_inputs():
    today = date.today()

    print("\nDate range")
    start_raw = input(f"Start date YYYY-MM-DD (Enter for today {today.isoformat()}): ").strip()
    start_date = _parse_date_str(start_raw) or today

    end_date = None
    while end_date is None:
        end_raw = input("End date / exam date YYYY-MM-DD (required): ").strip()
        end_date = _parse_date_str(end_raw)
        if end_date is None:
            print("Invalid end date. Use YYYY-MM-DD.")
            continue
        if end_date < start_date:
            print("End date must be >= start date.")
            end_date = None

    print("\nGoogle Calendar")
    calendar_id = input("Calendar ID (Enter for primary): ").strip() or "primary"
    timezone_name = input("Timezone (Enter for IST Asia/Kolkata, you can also type IST): ").strip()
    if not timezone_name:
        timezone_name = "Asia/Kolkata"

    print("\nConstraints")
    constraints = dict(DEFAULT_CONSTRAINTS)
    constraints["include_weekends"] = _normalize_bool(
        input("Include weekends? (yes/no) [yes]: ").strip() or "yes",
        default=True,
    )
    constraints["study_window_start"] = _parse_time_hhmm(
        input(f"Study window start HH:MM [{constraints['study_window_start']}]: ").strip(),
        _parse_time_hhmm(constraints["study_window_start"], time(hour=18, minute=0)),
    ).strftime("%H:%M")
    constraints["study_window_end"] = _parse_time_hhmm(
        input(f"Study window end HH:MM [{constraints['study_window_end']}]: ").strip(),
        _parse_time_hhmm(constraints["study_window_end"], time(hour=22, minute=30)),
    ).strftime("%H:%M")
    constraints["buffer_days"] = _normalize_int(
        input(f"Buffer days before exam [{constraints['buffer_days']}]: ").strip() or constraints["buffer_days"],
        constraints["buffer_days"],
        minimum=0,
        maximum=14,
    )

    constraints["daily_max_minutes"] = _normalize_optional_int(
        input("Daily max study minutes (Enter to leave unset): ").strip(),
        minimum=30,
        maximum=720,
    )
    constraints["min_slot_minutes"] = None
    constraints["max_slot_minutes"] = None

    constraints["additional_constraints"] = input("Any additional constraints: ").strip()

    constraints = _normalize_constraints(constraints)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "calendar_id": calendar_id,
        "timezone_name": timezone_name,
        "constraints": constraints,
    }


def planner_main():
    db = DocumentDB()

    session_id = choose_session(db)
    if not session_id:
        return

    session_profile = db.get_session_profile(session_id) or {}
    print(f"\nSelected session: {session_id} | {session_profile.get('title') or 'Untitled'}")

    chunks = load_session_chunks(db, session_id)
    if not chunks:
        print("No chunks found for this session. Ingest documents first.")
        return

    print(f"Loaded {len(chunks)} chunk metadata records.")

    user_inputs = ask_user_inputs()
    start_date = user_inputs["start_date"]
    end_date = user_inputs["end_date"]
    calendar_id = user_inputs["calendar_id"]
    constraints = _normalize_constraints(dict(user_inputs["constraints"]))
    locked_constraint_keys = _locked_constraint_keys(constraints)
    timezone_name = user_inputs["timezone_name"]

    calendar_service = GoogleCalendarService(timezone_name=timezone_name)
    tzinfo = calendar_service.tzinfo
    print("Calendar authentication and client initialization complete.")
    print(f"Planner timezone: {calendar_service.timezone_name}")

    buffer_days = _normalize_int(constraints.get("buffer_days"), 1, minimum=0, maximum=14)
    schedule_end = end_date - timedelta(days=buffer_days)
    if schedule_end < start_date:
        schedule_end = start_date

    # Planning loop with optional regeneration if Gemma updates constraints.
    gemma_agent = GemmaPlannerAgent()

    gemma_feedback = {}
    slots = []
    coverage = {}
    gemma_feedback_history = []
    qwen_feedback_history = []
    round_change_log = []

    for round_idx in range(2):
        print(f"\n[Planner] Pass {round_idx + 1}/2: fetching calendar events...")
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
        constraints = dict(schedule_data.get("constraints", constraints))
        slots = schedule_data.get("slots", [])
        coverage = schedule_data.get("coverage", {})
        busy_events = schedule_data.get("busy_events", [])
        free_blocks = schedule_data.get("free_blocks", [])

        print(f"[Planner] Pass {round_idx + 1}/2: free blocks found = {schedule_data.get('free_block_count', len(free_blocks))}")
        print(
            f"[Planner] Pass {round_idx + 1}/2: built {len(slots)} slots | "
            f"coverage={coverage.get('coverage_pct', 0)}%"
        )

        session_context = {
            "round": round_idx + 1,
            "session_id": session_id,
            "locked_constraints": sorted(list(locked_constraint_keys)),
            "changes_since_last_round": round_change_log[-1] if round_change_log else {},
            "round_change_log": round_change_log[-5:],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "schedule_end": schedule_end.isoformat(),
            },
            "chunk_count": len(chunks),
            "busy_event_count": len(busy_events),
            "free_block_count": len(free_blocks),
            "constraints": constraints,
            "coverage": {
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
                "partial_chunks": coverage.get("partially_covered_chunks"),
            },
        }

        toolbox = PlanningToolbox(
            chunks=chunks,
            slots=slots,
            coverage=coverage,
            constraints=constraints,
        )

        print(f"[Planner] Pass {round_idx + 1}/2: running Gemma planner agent...")
        gemma_feedback = gemma_agent.run(
            session_context=session_context,
            toolbox=toolbox,
            max_turns=GEMMA_PLANNER_MAX_TURNS,
        )
        gemma_feedback_history.append(gemma_feedback)
        print(f"[Planner] Pass {round_idx + 1}/2: Gemma response received.")

        constraint_updates = {}
        model_updates = gemma_feedback.get("updated_constraints", {})
        if isinstance(model_updates, dict):
            constraint_updates.update(model_updates)
        tool_updates = gemma_feedback.get("_tool_constraint_updates", {})
        if isinstance(tool_updates, dict):
            constraint_updates.update(tool_updates)

        round_constraints_before = dict(constraints)
        round_coverage_before = dict(coverage)

        updated_constraints = apply_constraint_updates(
            current=constraints,
            updates=constraint_updates,
            locked_keys=locked_constraint_keys,
        )

        needs_regen = bool(gemma_feedback.get("needs_regeneration", False))
        changed = updated_constraints != constraints
        constraints = updated_constraints

        main_round_entry = {
            "phase": f"main_pass_{round_idx + 1}",
            "constraint_diff": _constraints_diff(round_constraints_before, constraints),
            "coverage_diff": _coverage_diff(round_coverage_before, coverage),
        }
        if not main_round_entry["constraint_diff"] and not main_round_entry["coverage_diff"]:
            main_round_entry["note"] = "no_material_changes"
        round_change_log.append(main_round_entry)

        if (not needs_regen and not changed) or round_idx == 1:
            break

        buffer_days = _normalize_int(constraints.get("buffer_days"), 1, minimum=0, maximum=14)
        schedule_end = end_date - timedelta(days=buffer_days)
        if schedule_end < start_date:
            schedule_end = start_date

        print("\nGemma requested schedule regeneration with updated constraints:")
        print(json.dumps(constraints, indent=2, ensure_ascii=True))

    review_payload = {
        "phase": "initial_qwen_review",
        "session_id": session_id,
        "gemma_feedback": {
            "final_summary": gemma_feedback.get("final_summary", ""),
            "reason": gemma_feedback.get("reason", ""),
            "needs_regeneration": gemma_feedback.get("needs_regeneration", False),
        },
        "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=constraints),
    }
    print("[Planner] Running Qwen review...")
    qwen_feedback = qwen_review_plan(review_payload)
    qwen_feedback_history.append(qwen_feedback)
    print("[Planner] Qwen review completed.")

    escalation_result = _apply_qwen_escalation_if_needed(
        qwen_feedback=qwen_feedback,
        constraints=constraints,
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
        old_constraints = constraints
        old_coverage = coverage
        constraints = escalation_result.get("constraints", constraints)
        slots = escalation_result.get("slots", slots)
        coverage = escalation_result.get("coverage", coverage)
        qwen_feedback = escalation_result.get("qwen_feedback", qwen_feedback)
        schedule_end = escalation_result.get("schedule_end", schedule_end)
        qwen_feedback_history.append(qwen_feedback)
        round_change_log.append({
            "phase": "qwen_severity_3_reset",
            "constraint_diff": _constraints_diff(old_constraints, constraints),
            "coverage_diff": _coverage_diff(old_coverage, coverage),
        })

    post_review_reasons = _post_review_refinement_reasons(qwen_feedback, coverage)
    if post_review_reasons:
        print("\nQwen flagged issues. Running one targeted Gemma refinement pass...")
        for reason in post_review_reasons:
            print(f"[Planner] Refinement trigger: {reason}")

        review_context = {
            "phase": "post_qwen_review_refinement",
            "session_id": session_id,
            "locked_constraints": sorted(list(locked_constraint_keys)),
            "changes_since_last_round": round_change_log[-1] if round_change_log else {},
            "round_change_log": round_change_log[-5:],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "schedule_end": schedule_end.isoformat(),
            },
            "constraints": constraints,
            "coverage": {
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
                "partial_chunks": coverage.get("partially_covered_chunks"),
            },
            "qwen_feedback": qwen_feedback,
        }

        review_toolbox = PlanningToolbox(
            chunks=chunks,
            slots=slots,
            coverage=coverage,
            constraints=constraints,
        )

        post_review_gemma_feedback = gemma_agent.run(
            session_context=review_context,
            toolbox=review_toolbox,
            max_turns=max(2, GEMMA_PLANNER_MAX_TURNS - 1),
        )
        gemma_feedback_history.append(post_review_gemma_feedback)

        refined_constraint_updates = {}
        model_updates = post_review_gemma_feedback.get("updated_constraints", {})
        if isinstance(model_updates, dict):
            refined_constraint_updates.update(model_updates)
        tool_updates = post_review_gemma_feedback.get("_tool_constraint_updates", {})
        if isinstance(tool_updates, dict):
            refined_constraint_updates.update(tool_updates)

        refined_constraints = apply_constraint_updates(
            current=constraints,
            updates=refined_constraint_updates,
            locked_keys=locked_constraint_keys,
        )

        pre_refinement_constraints = dict(constraints)
        pre_refinement_coverage = dict(coverage)
        should_regenerate = bool(post_review_gemma_feedback.get("needs_regeneration", False))
        constraints_changed = refined_constraints != constraints

        if should_regenerate or constraints_changed:
            constraints = refined_constraints
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
            constraints = dict(schedule_data.get("constraints", constraints))
            slots = schedule_data.get("slots", [])
            coverage = schedule_data.get("coverage", {})

            review_payload = {
                "phase": "post_qwen_refinement_review",
                "session_id": session_id,
                "gemma_feedback": {
                    "final_summary": post_review_gemma_feedback.get("final_summary", ""),
                    "reason": post_review_gemma_feedback.get("reason", ""),
                    "needs_regeneration": post_review_gemma_feedback.get("needs_regeneration", False),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=constraints),
            }
            qwen_feedback = qwen_review_plan(review_payload)
            qwen_feedback_history.append(qwen_feedback)

        post_review_entry = {
            "phase": "post_qwen_refinement",
            "constraint_diff": _constraints_diff(pre_refinement_constraints, constraints),
            "coverage_diff": _coverage_diff(pre_refinement_coverage, coverage),
        }
        if not post_review_entry["constraint_diff"] and not post_review_entry["coverage_diff"]:
            post_review_entry["note"] = "no_material_changes"
        round_change_log.append(post_review_entry)

        gemma_feedback = post_review_gemma_feedback

    user_feedback = ""

    plan_id = persist_draft(
        db=db,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
        constraints=constraints,
        coverage=coverage,
        slots=slots,
        model_notes={
            "gemma": gemma_feedback,
            "qwen": qwen_feedback,
            "gemma_feedback_history": gemma_feedback_history,
            "qwen_feedback_history": qwen_feedback_history,
            "user_feedback": user_feedback,
        },
        calendar_id=calendar_id,
    )

    print_plan_summary(
        plan_id=plan_id,
        slots=slots,
        coverage=coverage,
        gemma_feedback=gemma_feedback,
        qwen_feedback=qwen_feedback,
    )

    feedback_history = []
    user_feedback = collect_user_plan_feedback()
    if user_feedback:
        feedback_history.append(user_feedback)

    for feedback_round in range(1, GEMMA_FEEDBACK_MAX_ROUNDS + 1):
        if not user_feedback:
            break

        print("\n[Planner] Running Gemma revision from user feedback...")
        user_revision_context = {
            "phase": "user_feedback_revision",
            "feedback_round": feedback_round,
            "session_id": session_id,
            "locked_constraints": sorted(list(locked_constraint_keys)),
            "user_feedback": user_feedback,
            "feedback_history": list(feedback_history),
            "changes_since_last_round": round_change_log[-1] if round_change_log else {},
            "round_change_log": round_change_log[-5:],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "schedule_end": schedule_end.isoformat(),
            },
            "constraints": constraints,
            "coverage": {
                "coverage_pct": coverage.get("coverage_pct"),
                "uncovered_chunks": coverage.get("uncovered_chunks"),
                "partial_chunks": coverage.get("partially_covered_chunks"),
            },
            "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=constraints),
        }

        user_revision_toolbox = PlanningToolbox(
            chunks=chunks,
            slots=slots,
            coverage=coverage,
            constraints=constraints,
        )

        user_revision_feedback = gemma_agent.run(
            session_context=user_revision_context,
            toolbox=user_revision_toolbox,
            max_turns=max(2, GEMMA_PLANNER_MAX_TURNS - 1),
        )
        gemma_feedback_history.append(user_revision_feedback)

        revised_constraint_updates = {}
        model_updates = user_revision_feedback.get("updated_constraints", {})
        if isinstance(model_updates, dict):
            revised_constraint_updates.update(model_updates)
        tool_updates = user_revision_feedback.get("_tool_constraint_updates", {})
        if isinstance(tool_updates, dict):
            revised_constraint_updates.update(tool_updates)

        deterministic_updates, deterministic_notes = derive_constraint_updates_from_user_feedback(
            user_feedback=user_feedback,
            feedback_history=feedback_history,
            current_constraints=constraints,
            reference_dates=(
                [
                    str(slot.get("start_time") or "")[:10]
                    for slot in slots
                    if str(slot.get("start_time") or "").strip()
                ]
                + [
                    (start_date + timedelta(days=day_offset)).isoformat()
                    for day_offset in range((end_date - start_date).days + 1)
                ]
            ),
        )
        if deterministic_updates:
            revised_constraint_updates.update(deterministic_updates)

            merged_model_updates = user_revision_feedback.get("updated_constraints", {})
            if not isinstance(merged_model_updates, dict):
                merged_model_updates = {}
            merged_model_updates.update(deterministic_updates)
            user_revision_feedback["updated_constraints"] = merged_model_updates

            user_revision_feedback["needs_regeneration"] = True
            reason_text = str(user_revision_feedback.get("reason") or "").strip()
            if "deterministic_user_feedback_update" not in reason_text:
                reason_text = (reason_text + "; deterministic_user_feedback_update").strip("; ")
            user_revision_feedback["reason"] = reason_text or "deterministic_user_feedback_update"

            if deterministic_notes:
                deterministic_note = " ".join(deterministic_notes).strip()
                summary_text = str(user_revision_feedback.get("final_summary") or "").strip()
                if deterministic_note and deterministic_note.lower() not in summary_text.lower():
                    user_revision_feedback["final_summary"] = (
                        (summary_text + " " + deterministic_note).strip()
                        if summary_text
                        else deterministic_note
                    )

        revised_constraints = apply_constraint_updates(
            current=constraints,
            updates=revised_constraint_updates,
            locked_keys=locked_constraint_keys,
        )
        pre_feedback_constraints = dict(constraints)
        pre_feedback_coverage = dict(coverage)
        chunk_updates = user_revision_feedback.get("updated_chunk_prerequisites", [])
        chunks_changed = apply_chunk_prerequisite_updates(chunks, chunk_updates)
        constraints_changed = revised_constraints != constraints

        if constraints_changed:
            constraints = revised_constraints

        if constraints_changed or chunks_changed or bool(user_revision_feedback.get("needs_regeneration", False)):
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
            constraints = dict(schedule_data.get("constraints", constraints))
            slots = schedule_data.get("slots", [])
            coverage = schedule_data.get("coverage", {})

            review_payload = {
                "phase": f"user_feedback_round_{feedback_round}_qwen_review",
                "session_id": session_id,
                "gemma_feedback": {
                    "final_summary": user_revision_feedback.get("final_summary", ""),
                    "reason": user_revision_feedback.get("reason", ""),
                    "needs_regeneration": user_revision_feedback.get("needs_regeneration", False),
                },
                "plan": _plan_snapshot(slots=slots, coverage=coverage, constraints=constraints),
                "user_feedback": user_feedback,
                "feedback_history": list(feedback_history),
            }
            qwen_feedback = qwen_review_plan(review_payload)
            qwen_feedback_history.append(qwen_feedback)

            escalation_result = _apply_qwen_escalation_if_needed(
                qwen_feedback=qwen_feedback,
                constraints=constraints,
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
                constraints = escalation_result.get("constraints", constraints)
                slots = escalation_result.get("slots", slots)
                coverage = escalation_result.get("coverage", coverage)
                qwen_feedback = escalation_result.get("qwen_feedback", qwen_feedback)
                schedule_end = escalation_result.get("schedule_end", schedule_end)
                qwen_feedback_history.append(qwen_feedback)

        feedback_round_entry = {
            "phase": f"user_feedback_round_{feedback_round}",
            "constraint_diff": _constraints_diff(pre_feedback_constraints, constraints),
            "coverage_diff": _coverage_diff(pre_feedback_coverage, coverage),
        }
        if not feedback_round_entry["constraint_diff"] and not feedback_round_entry["coverage_diff"]:
            feedback_round_entry["note"] = "no_material_changes"
        round_change_log.append(feedback_round_entry)

        gemma_feedback = user_revision_feedback

        db.update_study_plan(
            plan_id=plan_id,
            constraints=constraints,
            coverage=coverage,
            raw_plan={
                "slots": slots,
                "model_notes": {
                    "gemma": gemma_feedback,
                    "qwen": qwen_feedback,
                    "gemma_feedback_history": gemma_feedback_history,
                    "qwen_feedback_history": qwen_feedback_history,
                    "user_feedback": user_feedback,
                    "feedback_history": list(feedback_history),
                },
            },
        )
        db.replace_study_plan_slots(plan_id=plan_id, session_id=session_id, slots=slots)
        db.replace_chunk_schedule_refs(plan_id=plan_id, session_id=session_id, slots=slots)

        print("\nUpdated plan after user feedback:")
        print_plan_summary(
            plan_id=plan_id,
            slots=slots,
            coverage=coverage,
            gemma_feedback=gemma_feedback,
            qwen_feedback=qwen_feedback,
        )

        if feedback_round >= GEMMA_FEEDBACK_MAX_ROUNDS:
            break

        user_feedback = collect_additional_plan_feedback(feedback_round + 1, GEMMA_FEEDBACK_MAX_ROUNDS)
        if user_feedback:
            feedback_history.append(user_feedback)

    print("\nActions:")
    print("1. Approve and write to Google Calendar")
    print("2. Exit and keep draft in DB")

    action = input("Choose action [1-2]: ").strip() or "2"
    if action != "1":
        print("Draft saved. No calendar events were written.")
        return

    sync_report = sync_to_calendar_and_mark_approved(
        db=db,
        calendar_service=calendar_service,
        calendar_id=calendar_id,
        plan_id=plan_id,
    )

    print("\nCalendar sync done.")
    print(f"Events created: {sync_report['created']} / {sync_report['total']}")
