import uuid

from planner_scheduling import _build_slot_description, _build_slot_title


def persist_draft(db, session_id, start_date, end_date, constraints, coverage, slots, model_notes, calendar_id):
    plan_id = str(uuid.uuid4())

    db.create_study_plan(
        plan_id=plan_id,
        session_id=session_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        constraints=constraints,
        coverage=coverage,
        raw_plan={
            "slots": slots,
            "model_notes": model_notes,
        },
        status="draft",
        calendar_mode="real",
        calendar_id=calendar_id,
    )
    db.replace_study_plan_slots(plan_id=plan_id, session_id=session_id, slots=slots)
    db.replace_chunk_schedule_refs(plan_id=plan_id, session_id=session_id, slots=slots)

    return plan_id


def sync_to_calendar_and_mark_approved(db, calendar_service, calendar_id, plan_id):
    slots = db.list_study_plan_slots(plan_id)

    write_payload = []
    for slot in slots:
        summary = _build_slot_title(slot.get("items", []))
        description = _build_slot_description(slot)
        write_payload.append({
            "slot_id": slot.get("slot_id"),
            "start_time": slot.get("start_time"),
            "end_time": slot.get("end_time"),
            "title": summary,
            "description": description,
        })

    results = calendar_service.create_events(calendar_id=calendar_id, slots=write_payload)

    success = 0
    for row in results:
        slot_id = row.get("slot_id")
        status = row.get("status", "unknown")
        event_id = row.get("event_id")

        db.update_study_plan_slot_calendar(
            slot_id=slot_id,
            calendar_event_id=event_id,
            calendar_status=status,
        )
        db.update_chunk_schedule_calendar_by_slot(
            slot_id=slot_id,
            calendar_event_id=event_id,
        )

        if status == "created":
            success += 1

    final_status = "approved" if success == len(results) else "approved_with_calendar_errors"
    db.update_study_plan(plan_id=plan_id, status=final_status, approved=True)

    return {
        "total": len(results),
        "created": success,
        "results": results,
    }


def print_plan_summary(plan_id, slots, coverage, gemma_feedback, qwen_feedback):
    print("\n" + "=" * 95)
    print(f"Draft plan saved: {plan_id}")
    print("=" * 95)
    print(
        "Coverage => "
        f"full={coverage['covered_chunks']}/{coverage['total_chunks']} | "
        f"partial={coverage['partially_covered_chunks']} | "
        f"uncovered={coverage['uncovered_chunks']} | "
        f"planned={coverage['total_planned_minutes']} min | "
        f"required={coverage['total_required_minutes']} min | "
        f"pct={coverage['coverage_pct']}%"
    )

    print("\nGemma planner summary:")
    print(f"- {gemma_feedback.get('final_summary', 'N/A')}")
    print(f"- regenerate: {gemma_feedback.get('needs_regeneration', False)}")
    print(f"- reason: {gemma_feedback.get('reason', '')}")

    print("\nQwen review:")
    print(f"- Severity: {qwen_feedback.get('severity', 'N/A')}")
    print(f"- Summary: {qwen_feedback.get('summary', 'N/A')}")
    for row in qwen_feedback.get("strengths", []):
        print(f"- Strength: {row}")
    for row in qwen_feedback.get("risks", []):
        print(f"- Risk: {row}")
    for row in qwen_feedback.get("suggested_adjustments", []):
        print(f"- Suggestion: {row}")

    print("\nPlanned slots (first 10):")
    for idx, slot in enumerate(slots[:10], 1):
        print(
            f"[{idx}] {slot['start_time']} -> {slot['end_time']} | "
            f"{slot['duration_minutes']} min | {slot['difficulty']} | {slot['title']}"
        )
        prereq = ", ".join(slot.get("prerequisites", [])) or "None"
        print(f"     prerequisites: {prereq}")
        for item in slot.get("items", []):
            focus_topics = ", ".join(item.get("focus_topics", [])) or "General review"
            item_prereqs = ", ".join(item.get("prerequisites", [])) or "None"
            print(
                f"     - {item.get('topic', 'Unknown Topic')} | focus: {focus_topics} | prereqs: {item_prereqs}"
            )

    if len(slots) > 10:
        print(f"... {len(slots) - 10} more slots")

    # Day-level trace log to show which chunks are referenced each day.
    daily = {}
    for slot in slots:
        day = str(slot.get("start_time", ""))[:10] or "unknown-day"
        bucket = daily.setdefault(day, {})
        for item in slot.get("items", []):
            chunk_id = str(item.get("chunk_id", "")).strip() or "unknown-chunk"
            topic = str(item.get("topic", "")).strip() or "Unknown Topic"
            bucket[chunk_id] = topic

    if daily:
        print("\nDaily chunk references:")
        for day in sorted(daily.keys()):
            refs = daily[day]
            pairs = [f"{chunk_id}:{topic}" for chunk_id, topic in refs.items()]
            print(f"- {day} | chunks={len(refs)}")
            print("  " + " | ".join(pairs[:8]))
            if len(pairs) > 8:
                print(f"  ... {len(pairs) - 8} more")
