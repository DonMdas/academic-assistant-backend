import ast
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_text(name, default):
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw).strip()


GEMMA_PLANNER_TIMEOUT_SECONDS = max(15, _env_int("GEMMA_PLANNER_TIMEOUT_SECONDS", 45))
GEMMA_PLANNER_MAX_OUTPUT_TOKENS = max(256, _env_int("GEMMA_PLANNER_MAX_OUTPUT_TOKENS", 700))
GEMMA_PLANNER_MAX_TURNS = max(1, _env_int("GEMMA_PLANNER_MAX_TURNS", 4))
GEMMA_PLANNER_MAX_RUNTIME_SECONDS = max(20, _env_int("GEMMA_PLANNER_MAX_RUNTIME_SECONDS", 90))
GEMMA_HISTORY_MAX_ITEMS = max(3, _env_int("GEMMA_HISTORY_MAX_ITEMS", 8))
GEMMA_HISTORY_MAX_CHARS = max(120, _env_int("GEMMA_HISTORY_MAX_CHARS", 500))
GEMMA_HISTORY_MAX_DEPTH = max(3, _env_int("GEMMA_HISTORY_MAX_DEPTH", 4))
GEMMA_FEEDBACK_MAX_ROUNDS = max(1, _env_int("GEMMA_FEEDBACK_MAX_ROUNDS", 3))

CHUNK_ORDER_MODES = {"prerequisite", "difficulty_asc", "difficulty_desc", "priority", "deadline"}
TIME_OF_DAY_OPTIONS = {"morning", "afternoon", "evening", "night", "any"}

WEEKDAY_NAME_TO_INDEX = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

DIFFICULTY_ORDER = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
}

DEFAULT_CONSTRAINTS = {
    "daily_max_minutes": None,
    "min_slot_minutes": None,
    "max_slot_minutes": None,
    "include_weekends": True,
    "blocked_weekdays": [],
    "blocked_dates": [],
    "study_window_start": "18:00",
    "study_window_end": "22:30",
    "specific_day_windows": [],
    "slot_overrides": [],
    "chunk_order_mode": "prerequisite",
    "buffer_days": 1,
    "additional_constraints": "",
}

SCHEDULING_CONSTRAINT_KEYS = {
    "daily_max_minutes",
    "min_slot_minutes",
    "max_slot_minutes",
    "start_date",
    "end_date",
    "include_weekends",
    "blocked_weekdays",
    "blocked_dates",
    "study_window_start",
    "study_window_end",
    "specific_day_windows",
    "slot_overrides",
    "chunk_order_mode",
    "buffer_days",
    "additional_constraints",
}


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    topic: str
    subtopics: list[str]
    summary: str
    difficulty: str
    prerequisites: list[str]
    scheduling_hints: dict
    estimated_time: int 
    content: str


@dataclass
class CalendarEvent:
    start: datetime
    end: datetime
    summary: str
    event_id: str | None = None


def _safe_json_loads(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _parse_dict_candidate(text):
    value = str(text or "").strip()
    if not value:
        return {}

    if value.startswith("```"):
        value = re.sub(r"^```(?:json|JSON)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
        value = value.strip()

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _iter_braced_dict_candidates(text):
    value = str(text or "")
    if not value:
        return

    in_string = False
    quote_char = ""
    escaped = False
    depth = 0
    start_index = None

    for index, char in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                yield value[start_index : index + 1]
                start_index = None


def _extract_first_json_object(text):
    if isinstance(text, dict):
        return text

    value = str(text or "").strip()
    if not value:
        return {}

    parsed = _parse_dict_candidate(value)
    if parsed:
        return parsed

    fenced_blocks = re.findall(r"```(?:json|JSON)?\s*([\s\S]*?)```", value)
    for block in fenced_blocks:
        parsed = _parse_dict_candidate(block)
        if parsed:
            return parsed

    for candidate in _iter_braced_dict_candidates(value):
        parsed = _parse_dict_candidate(candidate)
        if parsed:
            return parsed

    return {}


def _prepend_no_think_message(messages):
    cleaned = list(messages or [])
    has_no_think = any(
        isinstance(message, dict) and "/no_think" in str(message.get("content", ""))
        for message in cleaned
    )
    if has_no_think:
        return cleaned

    return [{"role": "system", "content": "/no_think"}] + cleaned


def _compact_for_llm(value, depth=0):
    if depth >= GEMMA_HISTORY_MAX_DEPTH and isinstance(value, (dict, list)):
        return "<truncated-depth>"

    if isinstance(value, dict):
        out = {}
        items = list(value.items())
        for index, (key, item_value) in enumerate(items):
            if index >= GEMMA_HISTORY_MAX_ITEMS:
                out["_truncated_keys"] = len(items) - GEMMA_HISTORY_MAX_ITEMS
                break
            safe_key = str(key)[:80]
            out[safe_key] = _compact_for_llm(item_value, depth=depth + 1)
        return out

    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            if index >= GEMMA_HISTORY_MAX_ITEMS:
                out.append(f"<truncated_items:{len(value) - GEMMA_HISTORY_MAX_ITEMS}>")
                break
            out.append(_compact_for_llm(item, depth=depth + 1))
        return out

    if isinstance(value, str):
        compact = re.sub(r"\s+", " ", value).strip()
        if len(compact) > GEMMA_HISTORY_MAX_CHARS:
            return compact[:GEMMA_HISTORY_MAX_CHARS].rstrip() + "..."
        return compact

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    text = str(value)
    if len(text) > GEMMA_HISTORY_MAX_CHARS:
        return text[:GEMMA_HISTORY_MAX_CHARS].rstrip() + "..."
    return text


def _parse_date_str(value):
    text = str(value or "").strip()
    if not text:
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except Exception:
            return None

    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except Exception:
            return None

    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    if match:
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except Exception:
            return None

    return None


def _parse_time_hhmm(value, fallback):
    text = str(value or "").strip()
    if not text:
        return fallback

    try:
        return datetime.strptime(text, "%H:%M").time()
    except Exception:
        return fallback


def _normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _normalize_int(value, default, minimum=None, maximum=None):
    try:
        result = int(value)
    except Exception:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)

    return result


def _normalize_optional_int(value, minimum=None, maximum=None):
    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        result = int(value)
    except Exception:
        return None

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)

    return result


def _normalize_specific_day_windows(value):
    entries = []
    if isinstance(value, dict):
        entries = [value]
    elif isinstance(value, list):
        entries = list(value)

    normalized = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        date_candidate = entry.get("date") or entry.get("day") or entry.get("scheduled_date")
        parsed_date = _parse_date_str(date_candidate)
        if parsed_date is None:
            continue

        start_default = time(hour=8, minute=0)
        end_default = time(hour=12, minute=0)
        start_value = entry.get("start") or entry.get("start_time") or entry.get("window_start")
        end_value = entry.get("end") or entry.get("end_time") or entry.get("window_end")
        start_clock = _parse_time_hhmm(start_value, start_default)
        end_clock = _parse_time_hhmm(end_value, end_default)

        if datetime.combine(parsed_date, end_clock) <= datetime.combine(parsed_date, start_clock):
            end_clock = (datetime.combine(parsed_date, start_clock) + timedelta(hours=2)).time()

        priority_text = str(entry.get("priority") or entry.get("mode") or "hard").strip().lower()
        priority = "hard" if priority_text in {"hard", "strict", "replace", "must"} else "soft"

        normalized_entry = {
            "date": parsed_date.isoformat(),
            "start": start_clock.strftime("%H:%M"),
            "end": end_clock.strftime("%H:%M"),
            "priority": priority,
        }

        key = (
            normalized_entry["date"],
            normalized_entry["start"],
            normalized_entry["end"],
            normalized_entry["priority"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_entry)

    return normalized


def _normalize_slot_overrides(value):
    entries = []
    if isinstance(value, dict):
        entries = [value]
    elif isinstance(value, list):
        entries = list(value)

    normalized = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        chunk_id = str(entry.get("chunk_id") or "").strip()
        if not chunk_id:
            continue

        preferred_date = None
        if entry.get("preferred_date") is not None:
            parsed_date = _parse_date_str(entry.get("preferred_date"))
            if parsed_date is not None:
                preferred_date = parsed_date.isoformat()

        tod = str(entry.get("preferred_time_of_day") or "any").strip().lower()
        if tod not in TIME_OF_DAY_OPTIONS:
            tod = "any"

        mode = str(entry.get("priority") or entry.get("mode") or "soft").strip().lower()
        mode = "hard" if mode in {"hard", "strict", "must", "replace"} else "soft"

        item = {
            "chunk_id": chunk_id,
            "preferred_date": preferred_date,
            "preferred_time_of_day": tod,
            "priority": mode,
        }

        key = (item["chunk_id"], item["preferred_date"], item["preferred_time_of_day"], item["priority"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)

    return normalized


def _normalize_blocked_weekdays(value):
    rows = []
    if isinstance(value, (list, tuple, set)):
        rows = list(value)
    elif value is not None:
        rows = [value]

    normalized = []
    seen = set()
    for row in rows:
        index = None
        if isinstance(row, int):
            index = row
        else:
            text = str(row or "").strip().lower()
            if text.isdigit():
                try:
                    index = int(text)
                except Exception:
                    index = None
            else:
                index = WEEKDAY_NAME_TO_INDEX.get(text)

        if index is None:
            continue

        if index < 0 or index > 6:
            continue

        if index in seen:
            continue
        seen.add(index)
        normalized.append(index)

    return sorted(normalized)


def _normalize_blocked_dates(value):
    rows = []
    if isinstance(value, (list, tuple, set)):
        rows = list(value)
    elif value is not None:
        rows = [value]

    normalized = []
    seen = set()
    for row in rows:
        parsed = None
        if isinstance(row, date):
            parsed = row
        else:
            parsed = _parse_date_str(row)

        if parsed is None:
            continue

        iso_value = parsed.isoformat()
        if iso_value in seen:
            continue
        seen.add(iso_value)
        normalized.append(iso_value)

    return sorted(normalized)


def _normalize_chunk_hints(value):
    source = value if isinstance(value, dict) else {}
    priority = _normalize_int(source.get("priority"), 3, minimum=1, maximum=5)
    skip = _normalize_bool(source.get("skip"), default=False)

    deadline = None
    if source.get("must_schedule_before") is not None:
        parsed = _parse_date_str(source.get("must_schedule_before"))
        if parsed is not None:
            deadline = parsed.isoformat()

    preferred_date = None
    if source.get("preferred_date") is not None:
        parsed = _parse_date_str(source.get("preferred_date"))
        if parsed is not None:
            preferred_date = parsed.isoformat()

    tod = str(source.get("preferred_time_of_day") or "any").strip().lower()
    if tod not in TIME_OF_DAY_OPTIONS:
        tod = "any"

    return {
        "priority": priority,
        "skip": skip,
        "must_schedule_before": deadline,
        "preferred_date": preferred_date,
        "preferred_time_of_day": tod,
    }


def _time_of_day_bucket(value):
    hour = value.hour if isinstance(value, datetime) else int(getattr(value, "hour", 0))
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _constraints_diff(before, after):
    prev = dict(before or {})
    curr = dict(after or {})
    keys = sorted(set(prev.keys()).union(curr.keys()))
    changes = []
    for key in keys:
        if prev.get(key) == curr.get(key):
            continue
        changes.append({
            "key": key,
            "before": prev.get(key),
            "after": curr.get(key),
        })
    return changes[:12]


def _coverage_diff(before, after):
    prev = dict(before or {})
    curr = dict(after or {})
    fields = [
        "coverage_pct",
        "covered_chunks",
        "partially_covered_chunks",
        "uncovered_chunks",
        "total_planned_minutes",
        "total_required_minutes",
    ]
    diff = {}
    for key in fields:
        if prev.get(key) != curr.get(key):
            diff[key] = {"before": prev.get(key), "after": curr.get(key)}
    return diff


def _infer_daily_max_minutes_from_text(text):
    value = str(text or "").strip().lower()
    if not value:
        return None

    patterns = [
        r"(?:max(?:imum)?\s*)?(\d{1,4})\s*(?:min|mins|minute|minutes)\s*(?:per\s*day|/\s*day|daily)",
        r"daily\s*(?:max(?:imum)?\s*)?(?:of\s*)?(\d{1,4})\s*(?:min|mins|minute|minutes)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None

    return None


def _normalize_constraints(constraints):
    merged = dict(constraints or {})

    if merged.get("daily_max_minutes") is None:
        inferred_daily_cap = _infer_daily_max_minutes_from_text(merged.get("additional_constraints", ""))
        if inferred_daily_cap is not None:
            merged["daily_max_minutes"] = inferred_daily_cap

    merged["daily_max_minutes"] = _normalize_optional_int(merged.get("daily_max_minutes"), minimum=30, maximum=720)
    merged["min_slot_minutes"] = _normalize_optional_int(merged.get("min_slot_minutes"), minimum=15, maximum=180)
    merged["max_slot_minutes"] = _normalize_optional_int(merged.get("max_slot_minutes"), minimum=15, maximum=300)
    merged["blocked_weekdays"] = _normalize_blocked_weekdays(merged.get("blocked_weekdays", []))
    merged["blocked_dates"] = _normalize_blocked_dates(merged.get("blocked_dates", []))
    merged["specific_day_windows"] = _normalize_specific_day_windows(merged.get("specific_day_windows", []))
    merged["slot_overrides"] = _normalize_slot_overrides(merged.get("slot_overrides", []))
    mode = str(merged.get("chunk_order_mode") or "prerequisite").strip().lower()
    merged["chunk_order_mode"] = mode if mode in CHUNK_ORDER_MODES else "prerequisite"
    merged["buffer_days"] = _normalize_int(merged.get("buffer_days"), 1, minimum=0, maximum=14)

    if (
        merged["min_slot_minutes"] is not None
        and merged["max_slot_minutes"] is not None
        and merged["max_slot_minutes"] < merged["min_slot_minutes"]
    ):
        merged["max_slot_minutes"] = merged["min_slot_minutes"]

    return merged


def _locked_constraint_keys(constraints):
    locked = set()
    source = dict(constraints or {})
    for key in ("daily_max_minutes", "min_slot_minutes", "max_slot_minutes"):
        if source.get(key) is not None:
            locked.add(key)

    windows = source.get("specific_day_windows")
    if isinstance(windows, list) and windows:
        locked.add("specific_day_windows")

    if str(source.get("chunk_order_mode") or "").strip():
        locked.add("chunk_order_mode")
    return locked


def _normalize_difficulty(value):
    text = str(value or "").strip().lower()
    if text not in DIFFICULTY_ORDER:
        return "intermediate"
    return text


def _normalize_text_list(value):
    if not isinstance(value, list):
        return []

    out = []
    seen = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _derive_topics_from_summary(summary_text):
    text = str(summary_text or "").strip()
    if not text:
        return []

    lowered = text.lower()
    marker = "topics covered:"
    if marker not in lowered:
        return []

    start = lowered.find(marker)
    topics_part = text[start + len(marker):].strip()
    if not topics_part:
        return []

    topics_part = re.split(r"\.\s", topics_part, maxsplit=1)[0]
    raw_parts = [p.strip() for p in topics_part.split(";")]
    return _normalize_text_list(raw_parts)


def _chunk_topic_track(chunk):
    base = _normalize_text_list(getattr(chunk, "subtopics", []))
    if not base:
        base = _derive_topics_from_summary(getattr(chunk, "summary", ""))
    if not base:
        base = [str(getattr(chunk, "topic", "") or "Unknown Topic").strip()]
    return _normalize_text_list(base)


def _focus_topics_for_piece(chunk, consumed_before, piece_minutes):
    required = max(1, _normalize_int(getattr(chunk, "estimated_time", 0), 30, minimum=1, maximum=600))
    consumed = max(0, _normalize_int(consumed_before, 0, minimum=0))
    piece = max(1, _normalize_int(piece_minutes, 1, minimum=1, maximum=required))

    topics = _chunk_topic_track(chunk)
    if not topics:
        topics = [str(getattr(chunk, "topic", "") or "Unknown Topic").strip()]

    n = len(topics)
    start_ratio = min(1.0, float(consumed / required))
    end_ratio = min(1.0, float((consumed + piece) / required))

    start_idx = min(n - 1, int(start_ratio * n))
    end_idx = max(start_idx, min(n - 1, int(max(0.0, end_ratio * n - 1e-9))))

    focus = topics[start_idx:end_idx + 1]
    if not focus:
        focus = [topics[min(n - 1, start_idx)]]

    return _normalize_text_list(focus)


def _tokenize(text):
    return re.findall(r"[a-z0-9]{2,}", str(text or "").lower())


def _keyword_overlap_score(query_text, candidate_text):
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return 0.0

    candidate_tokens = set(_tokenize(candidate_text))
    if not candidate_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(candidate_tokens))
    return float(overlap / max(1, len(query_tokens)))


def _normalize_timezone_name(name):
    text = str(name or "").strip()
    if not text:
        return "Asia/Kolkata"

    upper = text.upper()
    if upper in {"UTC", "ETC/UTC", "Z"}:
        return "UTC"

    aliases = {
        "IST": "Asia/Kolkata",
        "ASIA/CALCUTTA": "Asia/Kolkata",
        "INDIA/STANDARD": "Asia/Kolkata",
    }
    return aliases.get(upper, text)


def _resolve_timezone(name):
    text = _normalize_timezone_name(name)
    if not text:
        return timezone.utc

    if text.upper() in {"UTC", "ETC/UTC", "Z"}:
        return timezone.utc

    try:
        return ZoneInfo(text)
    except Exception:
        print(f"Invalid timezone '{text}'. Falling back to UTC.")
        return timezone.utc
