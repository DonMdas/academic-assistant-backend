import uuid
from datetime import date, datetime, time, timedelta

from planner_calendar import compute_free_blocks
from planner_common import (
    CHUNK_ORDER_MODES,
    DIFFICULTY_ORDER,
    TIME_OF_DAY_OPTIONS,
    ChunkRecord,
    _focus_topics_for_piece,
    _normalize_chunk_hints,
    _normalize_constraints,
    _normalize_difficulty,
    _normalize_int,
    _normalize_optional_int,
    _normalize_slot_overrides,
    _normalize_text_list,
    _parse_date_str,
    _safe_json_loads,
    _time_of_day_bucket,
)


def load_session_chunks(db, session_id):
    rows = db.get_session_chunks(session_id)
    chunks = []

    for chunk_id, doc_id, content, metadata_json in rows:
        metadata = _safe_json_loads(metadata_json, {})

        chunks.append(
            ChunkRecord(
                chunk_id=str(chunk_id),
                doc_id=str(doc_id),
                topic=str(metadata.get("topic") or "Unknown Topic").strip(),
                subtopics=_normalize_text_list(metadata.get("subtopics")),
                summary=str(metadata.get("summary") or "").strip(),
                difficulty=_normalize_difficulty(metadata.get("complexity")),
                prerequisites=_normalize_text_list(metadata.get("prerequisites")),
                scheduling_hints=_normalize_chunk_hints(metadata.get("scheduling_hints")),
                estimated_time=_normalize_int(metadata.get("estimated_time"), 30, minimum=10, maximum=240),
                content=str(content or ""),
            )
        )

    return chunks


def _sort_chunks_for_schedule(chunks, constraints=None):
    topic_map = {chunk.topic.lower().strip(): chunk.chunk_id for chunk in chunks if chunk.topic.strip()}
    mode = str(dict(constraints or {}).get("chunk_order_mode") or "prerequisite").strip().lower()
    if mode not in CHUNK_ORDER_MODES:
        mode = "prerequisite"

    def prereq_count(chunk):
        count = 0
        for item in chunk.prerequisites:
            if item.lower().strip() in topic_map:
                count += 1
        return count

    def _deadline_value(chunk):
        hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
        parsed = _parse_date_str(hints.get("must_schedule_before"))
        return parsed or date.max

    def _priority_value(chunk):
        hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
        return int(hints.get("priority", 3))

    def sort_key(chunk):
        difficulty = DIFFICULTY_ORDER.get(chunk.difficulty, 2)
        prereqs = prereq_count(chunk)
        priority = _priority_value(chunk)
        deadline = _deadline_value(chunk)

        if mode == "difficulty_asc":
            return (difficulty, prereqs, -priority, deadline, chunk.estimated_time, chunk.topic.lower())
        if mode == "difficulty_desc":
            return (-difficulty, prereqs, -priority, deadline, chunk.estimated_time, chunk.topic.lower())
        if mode == "priority":
            return (-priority, prereqs, difficulty, deadline, chunk.estimated_time, chunk.topic.lower())
        if mode == "deadline":
            return (deadline, -priority, prereqs, difficulty, chunk.estimated_time, chunk.topic.lower())

        # Default prerequisite mode.
        return (prereqs, difficulty, -priority, deadline, chunk.estimated_time, chunk.topic.lower())

    filtered = []
    for chunk in chunks:
        hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
        if hints.get("skip"):
            continue
        filtered.append(chunk)

    return sorted(
        filtered,
        key=sort_key,
    )


def _slot_target_minutes(difficulty):
    if difficulty == "beginner":
        return 45
    if difficulty == "advanced":
        return 90
    return 60


def _primary_slot_difficulty(items):
    if not items:
        return "intermediate"
    hardest = max(items, key=lambda item: DIFFICULTY_ORDER.get(item.get("difficulty", "intermediate"), 2))
    return hardest.get("difficulty", "intermediate")


def _build_slot_title(items):
    if len(items) == 1:
        item = items[0]
        topic = str(item.get("topic") or "").strip()
        focus_topics = _normalize_text_list(item.get("focus_topics"))
        if topic and focus_topics:
            return f"Study: {topic} - {focus_topics[0]}"

    topics = []
    seen = set()

    for item in items:
        topic = str(item.get("topic") or "").strip()
        if not topic:
            continue
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        topics.append(topic)

    if not topics:
        return "Study Session"

    if len(topics) == 1:
        return f"Study: {topics[0]}"

    return f"Study: {topics[0]} + {len(topics) - 1} more"


def _build_slot_description(slot):
    lines = []
    lines.append(f"Difficulty focus: {slot.get('difficulty', 'intermediate')}")

    for idx, item in enumerate(slot.get("items", []), 1):
        prereq = ", ".join(item.get("prerequisites", [])) or "None"
        chunk_id = str(item.get("chunk_id") or "").strip() or "unknown"
        focus_topics = _normalize_text_list(item.get("focus_topics"))
        focus_label = "; ".join(focus_topics) if focus_topics else "General review"
        lines.append(
            f"{idx}. {item.get('topic', 'Unknown Topic')} "
            f"({item.get('allocated_minutes', 0)} min, {item.get('difficulty', 'intermediate')})"
        )
        lines.append(f"   chunk_ref: {chunk_id}")
        lines.append(f"   focus_topics: {focus_label}")
        lines.append(f"   prerequisites: {prereq}")

    return "\n".join(lines)


def build_plan_slots(chunks, free_blocks, constraints):
    ordered_chunks = _sort_chunks_for_schedule(chunks, constraints=constraints)

    min_slot_raw = _normalize_optional_int(constraints.get("min_slot_minutes"), minimum=15, maximum=180)
    max_slot_raw = _normalize_optional_int(constraints.get("max_slot_minutes"), minimum=15, maximum=300)

    # Unset min/max means no hard caps; the scheduler still uses difficulty targets.
    min_slot = min_slot_raw if min_slot_raw is not None else 1
    max_slot = max_slot_raw

    slot_overrides = _normalize_slot_overrides(constraints.get("slot_overrides", []))
    overrides_by_chunk = {}
    for row in slot_overrides:
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        overrides_by_chunk.setdefault(chunk_id, []).append(row)

    rank_map = {chunk.chunk_id: index for index, chunk in enumerate(ordered_chunks)}

    def _matches_override_for_block(rule, block_start):
        if not isinstance(rule, dict):
            return False
        block_date = block_start.date().isoformat()
        block_period = _time_of_day_bucket(block_start)

        date_match = True
        expected_date = rule.get("preferred_date")
        if expected_date:
            date_match = str(expected_date) == block_date

        period_match = True
        expected_period = str(rule.get("preferred_time_of_day") or "any").strip().lower()
        if expected_period not in TIME_OF_DAY_OPTIONS:
            expected_period = "any"
        if expected_period != "any":
            period_match = expected_period == block_period

        return bool(date_match and period_match)

    def _chunk_rules(chunk, include_overrides):
        hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
        if not include_overrides:
            return hints, []

        chunk_id = chunk.chunk_id
        rules = list(overrides_by_chunk.get(chunk_id, []))
        if hints.get("preferred_date") or hints.get("preferred_time_of_day") not in {None, "", "any"}:
            rules.append({
                "chunk_id": chunk_id,
                "preferred_date": hints.get("preferred_date"),
                "preferred_time_of_day": hints.get("preferred_time_of_day", "any"),
                "priority": "soft",
            })
        return hints, rules

    def _select_chunk_for_block(block_start, remaining, include_overrides):
        candidates = [chunk for chunk in ordered_chunks if remaining.get(chunk.chunk_id, 0) > 0]
        if not candidates:
            return None

        hard_matches = []
        scored = []
        block_date = block_start.date()
        block_period = _time_of_day_bucket(block_start)

        for chunk in candidates:
            chunk_id = chunk.chunk_id
            hints, rules = _chunk_rules(chunk, include_overrides=include_overrides)

            if include_overrides:
                chunk_hard_rules = [rule for rule in rules if str(rule.get("priority") or "soft").lower() == "hard"]
                if chunk_hard_rules:
                    if any(_matches_override_for_block(rule, block_start) for rule in chunk_hard_rules):
                        hard_matches.append(chunk)
                    continue

            score = 0.0
            if include_overrides and any(_matches_override_for_block(rule, block_start) for rule in rules):
                score += 500.0

            deadline = _parse_date_str(hints.get("must_schedule_before"))
            if deadline is not None:
                days_left = (deadline - block_date).days
                if days_left < 0:
                    score += 800.0
                else:
                    score += max(0.0, 120.0 - float(days_left * 12))

            if include_overrides and hints.get("preferred_time_of_day") == block_period:
                score += 25.0

            score += float(hints.get("priority", 3)) * 10.0
            score += max(0.0, 200.0 - float(rank_map.get(chunk_id, 9999)))

            scored.append((score, chunk))

        if hard_matches:
            hard_matches.sort(key=lambda item: rank_map.get(item.chunk_id, 9999))
            return hard_matches[0]

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _build_slots_pass(include_overrides):
        remaining = {chunk.chunk_id: int(chunk.estimated_time) for chunk in ordered_chunks}
        consumed = {chunk.chunk_id: 0 for chunk in ordered_chunks}
        slots = []

        def _has_remaining_work():
            return any(remaining.get(chunk.chunk_id, 0) > 0 for chunk in ordered_chunks)

        for block_start, block_end in free_blocks:
            if not _has_remaining_work():
                break

            block_minutes = int((block_end - block_start).total_seconds() // 60)
            if block_minutes < min_slot:
                continue

            used = 0
            cursor = block_start
            while used + min_slot <= block_minutes:
                if not _has_remaining_work():
                    break

                remaining_block = block_minutes - used
                slot_limit = remaining_block if max_slot is None else min(max_slot, remaining_block)
                if slot_limit < min_slot:
                    break

                slot_start = cursor + timedelta(minutes=used)
                slot_used = 0
                items = []

                while slot_used < slot_limit:
                    absolute_used = used + slot_used
                    chunk = _select_chunk_for_block(
                        cursor + timedelta(minutes=absolute_used),
                        remaining=remaining,
                        include_overrides=include_overrides,
                    )
                    if chunk is None:
                        break

                    need = remaining[chunk.chunk_id]
                    if need <= 0:
                        continue

                    block_left = block_minutes - absolute_used
                    slot_left = slot_limit - slot_used
                    if block_left <= 0 or slot_left <= 0:
                        break

                    piece = min(
                        need,
                        block_left,
                        slot_left,
                        _slot_target_minutes(chunk.difficulty),
                    )
                    if piece <= 0:
                        break

                    item_start = cursor + timedelta(minutes=absolute_used)
                    item_end = item_start + timedelta(minutes=int(piece))
                    consumed_before = int(consumed.get(chunk.chunk_id, 0))
                    focus_topics = _focus_topics_for_piece(
                        chunk=chunk,
                        consumed_before=consumed_before,
                        piece_minutes=int(piece),
                    )
                    progress_start_pct = round((consumed_before / max(1, int(chunk.estimated_time))) * 100.0, 2)
                    progress_end_pct = round(((consumed_before + int(piece)) / max(1, int(chunk.estimated_time))) * 100.0, 2)

                    items.append({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "topic": chunk.topic,
                        "focus_topics": focus_topics,
                        "summary": chunk.summary,
                        "difficulty": chunk.difficulty,
                        "prerequisites": list(chunk.prerequisites),
                        "allocated_minutes": int(piece),
                        "required_minutes": int(chunk.estimated_time),
                        "chunk_progress_pct_start": progress_start_pct,
                        "chunk_progress_pct_end": progress_end_pct,
                        "scheduled_date": item_start.date().isoformat(),
                        "scheduled_start_time": item_start.isoformat(),
                        "scheduled_end_time": item_end.isoformat(),
                    })

                    slot_used += int(piece)
                    remaining[chunk.chunk_id] -= int(piece)
                    consumed[chunk.chunk_id] += int(piece)

                if not items:
                    break

                slot_end = slot_start + timedelta(minutes=slot_used)
                difficulty = _primary_slot_difficulty(items)
                prerequisites = sorted({p for item in items for p in item.get("prerequisites", []) if str(p).strip()})
                chunk_ids = [item["chunk_id"] for item in items]

                slot = {
                    "slot_id": str(uuid.uuid4()),
                    "start_time": slot_start.isoformat(),
                    "end_time": slot_end.isoformat(),
                    "duration_minutes": int(slot_used),
                    "difficulty": difficulty,
                    "items": items,
                    "prerequisites": prerequisites,
                    "coverage_chunk_ids": chunk_ids,
                    "calendar_status": "pending",
                }
                slot["title"] = _build_slot_title(items)
                slot["description"] = _build_slot_description(slot)
                slots.append(slot)

                used += int(slot_used)
                if slot_used <= 0:
                    break

        return slots

    def _needs_override_second_pass():
        if slot_overrides:
            return True
        for chunk in ordered_chunks:
            hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
            if hints.get("preferred_date"):
                return True
            if hints.get("preferred_time_of_day") not in {None, "", "any"}:
                return True
        return False

    baseline_slots = _build_slots_pass(include_overrides=False)
    if not _needs_override_second_pass():
        return baseline_slots

    # Explicit second pass: rebuild assignment while honoring slot overrides and time preferences.
    return _build_slots_pass(include_overrides=True)


def build_coverage(chunks, slots):
    required = {chunk.chunk_id: int(chunk.estimated_time) for chunk in chunks}
    assigned = {}

    for slot in slots:
        for item in slot.get("items", []):
            chunk_id = item.get("chunk_id")
            mins = _normalize_int(item.get("allocated_minutes"), 0, minimum=0)
            assigned[chunk_id] = assigned.get(chunk_id, 0) + mins

    covered = []
    partial = []
    uncovered = []

    for chunk in chunks:
        req = required.get(chunk.chunk_id, 0)
        got = assigned.get(chunk.chunk_id, 0)

        row = {
            "chunk_id": chunk.chunk_id,
            "topic": chunk.topic,
            "required_minutes": req,
            "planned_minutes": got,
        }

        if got <= 0:
            uncovered.append(row)
        elif got < req:
            row["missing_minutes"] = req - got
            partial.append(row)
        else:
            covered.append(row)

    total_required = sum(required.values())
    total_planned = sum(_normalize_int(slot.get("duration_minutes"), 0, minimum=0) for slot in slots)

    pct = 0.0
    if total_required > 0:
        pct = min(100.0, (total_planned / total_required) * 100.0)

    return {
        "total_chunks": len(chunks),
        "covered_chunks": len(covered),
        "partially_covered_chunks": len(partial),
        "uncovered_chunks": len(uncovered),
        "total_required_minutes": total_required,
        "total_planned_minutes": total_planned,
        "coverage_pct": round(pct, 2),
        "covered": covered,
        "partial": partial,
        "uncovered": uncovered,
    }


def build_schedule_data(
    chunks,
    constraints,
    start_date,
    end_date,
    schedule_end,
    tzinfo,
    calendar_service,
    calendar_id,
):
    normalized_constraints = _normalize_constraints(constraints)

    range_start = datetime.combine(start_date, time.min, tzinfo=tzinfo)
    range_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=tzinfo)

    busy_events = calendar_service.list_events(
        calendar_id=calendar_id,
        start_dt=range_start,
        end_dt=range_end,
    )

    free_blocks = compute_free_blocks(
        start_date=start_date,
        end_date=schedule_end,
        constraints=normalized_constraints,
        tzinfo=tzinfo,
        busy_events=busy_events,
    )

    slots = build_plan_slots(
        chunks=chunks,
        free_blocks=free_blocks,
        constraints=normalized_constraints,
    )
    coverage = build_coverage(chunks=chunks, slots=slots)

    free_block_rows = []
    for block_start, block_end in free_blocks:
        free_block_rows.append({
            "start_time": block_start.isoformat(),
            "end_time": block_end.isoformat(),
            "duration_minutes": int((block_end - block_start).total_seconds() // 60),
        })

    return {
        "constraints": normalized_constraints,
        "busy_events": busy_events,
        "busy_event_count": len(busy_events),
        "free_blocks": free_blocks,
        "free_block_count": len(free_blocks),
        "free_block_rows": free_block_rows,
        "slots": slots,
        "coverage": coverage,
    }
