import re

from planner_common import (
    _keyword_overlap_score,
    _normalize_chunk_hints,
    _normalize_int,
    _normalize_slot_overrides,
    _normalize_specific_day_windows,
    _normalize_text_list,
    _parse_date_str,
)


class PlanningToolbox:
    def __init__(self, chunks, slots, coverage, constraints):
        self.chunks = chunks
        self.slots = slots
        self.coverage = coverage
        self.constraints = dict(constraints)
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.pending_constraint_updates = {}

    def get_plan_context(self):
        by_diff = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for chunk in self.chunks:
            by_diff[chunk.difficulty] = by_diff.get(chunk.difficulty, 0) + 1

        return {
            "chunk_count": len(self.chunks),
            "difficulty_mix": by_diff,
            "slot_count": len(self.slots),
            "coverage": {
                "coverage_pct": self.coverage.get("coverage_pct"),
                "covered_chunks": self.coverage.get("covered_chunks"),
                "partial": self.coverage.get("partially_covered_chunks"),
                "uncovered": self.coverage.get("uncovered_chunks"),
                "required_minutes": self.coverage.get("total_required_minutes"),
                "planned_minutes": self.coverage.get("total_planned_minutes"),
            },
            "constraints": self.constraints,
            "slot_overrides": _normalize_slot_overrides(self.constraints.get("slot_overrides", [])),
        }

    def get_uncovered_chunks(self, limit=10):
        limit = _normalize_int(limit, 10, minimum=1, maximum=50)
        return self.coverage.get("uncovered", [])[:limit]

    def search_chunks(self, query, top_k=6):
        top_k = _normalize_int(top_k, 6, minimum=1, maximum=25)
        query_text = str(query or "").strip()
        if not query_text:
            out = []
            for chunk in self.chunks[:top_k]:
                hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
                out.append({
                    "chunk_id": chunk.chunk_id,
                    "topic": chunk.topic,
                    "difficulty": chunk.difficulty,
                    "estimated_time": chunk.estimated_time,
                    "priority": hints.get("priority"),
                    "skip": hints.get("skip"),
                    "must_schedule_before": hints.get("must_schedule_before"),
                    "score": 0.0,
                    "match_type": "sample",
                })

            if out:
                out[0]["_note"] = "Empty query received; returning sample chunks. Provide a topic keyword for relevant matches."

            return out

        scored = []
        for chunk in self.chunks:
            prereq_text = "; ".join(chunk.prerequisites)
            searchable = f"{chunk.topic}\n{chunk.summary}\n{prereq_text}\n{chunk.difficulty}"
            score = _keyword_overlap_score(query_text, searchable)
            if score <= 0:
                continue
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        out = []
        for score, chunk in scored[:top_k]:
            hints = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
            out.append({
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "difficulty": chunk.difficulty,
                "estimated_time": chunk.estimated_time,
                "priority": hints.get("priority"),
                "skip": hints.get("skip"),
                "must_schedule_before": hints.get("must_schedule_before"),
                "score": round(float(score), 4),
            })
        return out

    def get_chunk_details(self, chunk_ids):
        if not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids]

        out = []
        for chunk_id in chunk_ids:
            key = str(chunk_id)
            chunk = self.chunk_by_id.get(key)
            if not chunk:
                continue

            out.append({
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "summary": chunk.summary,
                "subtopics": list(chunk.subtopics),
                "difficulty": chunk.difficulty,
                "estimated_time": chunk.estimated_time,
                "prerequisites": list(chunk.prerequisites),
                "scheduling_hints": _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {})),
                "content_preview": chunk.content[:420],
            })

        return out

    def get_slot_details(self, limit=8):
        limit = _normalize_int(limit, 8, minimum=1, maximum=30)
        rows = []
        for slot in self.slots[:limit]:
            rows.append({
                "slot_id": slot.get("slot_id"),
                "start_time": slot.get("start_time"),
                "end_time": slot.get("end_time"),
                "duration_minutes": slot.get("duration_minutes"),
                "difficulty": slot.get("difficulty"),
                "topics": [item.get("topic") for item in slot.get("items", [])],
                "focus_topics": [item.get("focus_topics", []) for item in slot.get("items", [])],
                "prerequisites": slot.get("prerequisites", []),
            })
        return rows

    def get_specific_day_windows(self):
        return _normalize_specific_day_windows(self.constraints.get("specific_day_windows", []))

    def set_specific_day_window(self, date_value, start_time=None, end_time=None, priority="hard", period=None):
        parsed_date = _parse_date_str(date_value)
        if parsed_date is None:
            day_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", str(date_value or "").lower())
            if day_match:
                target_day = _normalize_int(day_match.group(1), 0, minimum=1, maximum=31)
                candidate_dates = []
                for slot in self.slots:
                    slot_start = str(slot.get("start_time") or "").strip()
                    if not slot_start:
                        continue
                    parsed_slot_date = _parse_date_str(slot_start[:10])
                    if parsed_slot_date is None:
                        continue
                    if parsed_slot_date.day == target_day:
                        candidate_dates.append(parsed_slot_date)

                if candidate_dates:
                    parsed_date = sorted(candidate_dates)[0]

        if parsed_date is None:
            return {"error": "invalid date. Use YYYY-MM-DD."}

        period_text = str(period or "").strip().lower()
        defaults = {
            "morning": ("08:00", "12:00"),
            "afternoon": ("13:00", "17:00"),
            "evening": ("18:00", "22:30"),
            "night": ("20:00", "23:00"),
        }

        default_start, default_end = defaults.get(period_text, ("08:00", "12:00"))
        start_value = str(start_time or default_start)
        end_value = str(end_time or default_end)

        row = {
            "date": parsed_date.isoformat(),
            "start": start_value,
            "end": end_value,
            "priority": str(priority or "hard").strip().lower() or "hard",
        }
        normalized_new = _normalize_specific_day_windows([row])
        if not normalized_new:
            return {"error": "invalid start/end time"}
        normalized_new = normalized_new[0]

        existing = self.get_specific_day_windows()
        if normalized_new.get("priority") == "hard":
            existing = [w for w in existing if not (w.get("date") == normalized_new.get("date") and w.get("priority") == "hard")]

        existing.append(normalized_new)
        merged = _normalize_specific_day_windows(existing)
        self.constraints["specific_day_windows"] = merged
        self.pending_constraint_updates["specific_day_windows"] = merged
        return {
            "status": "ok",
            "specific_day_windows": merged,
        }

    def clear_specific_day_window(self, date_value=None):
        windows = self.get_specific_day_windows()
        if not windows:
            return {"status": "ok", "specific_day_windows": []}

        if date_value is None:
            self.constraints["specific_day_windows"] = []
            self.pending_constraint_updates["specific_day_windows"] = []
            return {"status": "ok", "specific_day_windows": []}

        parsed_date = _parse_date_str(date_value)
        if parsed_date is None:
            return {"error": "invalid date. Use YYYY-MM-DD."}

        target = parsed_date.isoformat()
        kept = [w for w in windows if w.get("date") != target]
        self.constraints["specific_day_windows"] = kept
        self.pending_constraint_updates["specific_day_windows"] = kept
        return {
            "status": "ok",
            "specific_day_windows": kept,
        }

    def consume_pending_constraint_updates(self):
        pending = dict(self.pending_constraint_updates)
        self.pending_constraint_updates = {}
        return pending

    def set_chunk_hints(self, updates):
        rows = []
        if isinstance(updates, dict):
            rows = [updates]
        elif isinstance(updates, list):
            rows = list(updates)

        changed = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                continue

            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue

            current = _normalize_chunk_hints(getattr(chunk, "scheduling_hints", {}))
            patch = {
                "priority": row.get("priority", current.get("priority")),
                "skip": row.get("skip", current.get("skip")),
                "must_schedule_before": row.get("must_schedule_before", current.get("must_schedule_before")),
                "preferred_date": row.get("preferred_date", current.get("preferred_date")),
                "preferred_time_of_day": row.get("preferred_time_of_day", current.get("preferred_time_of_day")),
            }
            merged = _normalize_chunk_hints(patch)
            if merged != current:
                chunk.scheduling_hints = merged
                changed.append({
                    "chunk_id": chunk_id,
                    "before": current,
                    "after": merged,
                })

        return {
            "status": "ok",
            "updated": changed,
            "updated_count": len(changed),
        }

    def propose_slot_overrides(self, overrides):
        merged = _normalize_slot_overrides(overrides)
        if not merged:
            return {"error": "no valid overrides"}

        current = _normalize_slot_overrides(self.constraints.get("slot_overrides", []))
        current.extend(merged)
        deduped = _normalize_slot_overrides(current)

        self.constraints["slot_overrides"] = deduped
        self.pending_constraint_updates["slot_overrides"] = deduped
        return {
            "status": "ok",
            "slot_overrides": deduped,
        }

    def execute_tool(self, tool_name, arguments):
        args = arguments if isinstance(arguments, dict) else {}

        if tool_name == "get_plan_context":
            return self.get_plan_context()

        if tool_name == "get_uncovered_chunks":
            return self.get_uncovered_chunks(limit=args.get("limit", 10))

        if tool_name == "search_chunks":
            return self.search_chunks(query=args.get("query", ""), top_k=args.get("top_k", 6))

        if tool_name == "get_chunk_details":
            return self.get_chunk_details(chunk_ids=args.get("chunk_ids", []))

        if tool_name == "get_slot_details":
            return self.get_slot_details(limit=args.get("limit", 8))

        if tool_name == "get_specific_day_windows":
            return self.get_specific_day_windows()

        if tool_name == "set_specific_day_window":
            return self.set_specific_day_window(
                date_value=args.get("date"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
                priority=args.get("priority", "hard"),
                period=args.get("period"),
            )

        if tool_name == "clear_specific_day_window":
            return self.clear_specific_day_window(date_value=args.get("date"))

        if tool_name == "set_chunk_hints":
            return self.set_chunk_hints(updates=args.get("updates", args))

        if tool_name == "propose_slot_overrides":
            return self.propose_slot_overrides(overrides=args.get("overrides", args))

        return {"error": f"Unknown tool: {tool_name}"}
