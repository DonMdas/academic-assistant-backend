import json
import os
import re
import time as time_module
from contextvars import ContextVar, Token
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable

from dotenv import load_dotenv

from config import (
    OLLAMA_BASE_URL,
    GEMMA_BACKEND,
    GEMMA_GEMINI_MODEL,
    GEMMA_OLLAMA_MODEL,
    DISABLE_OLLAMA_THINKING,
)
from ollama_parser import safe_ollama_call
from planner_common import (
    CHUNK_ORDER_MODES,
    DEFAULT_CONSTRAINTS,
    GEMMA_PLANNER_MAX_OUTPUT_TOKENS,
    GEMMA_PLANNER_MAX_RUNTIME_SECONDS,
    GEMMA_PLANNER_MAX_TURNS,
    GEMMA_PLANNER_TIMEOUT_SECONDS,
    SCHEDULING_CONSTRAINT_KEYS,
    WEEKDAY_NAME_TO_INDEX,
    _compact_for_llm,
    _env_text,
    _extract_first_json_object,
    _normalize_bool,
    _normalize_blocked_dates,
    _normalize_blocked_weekdays,
    _normalize_chunk_hints,
    _normalize_constraints,
    _normalize_int,
    _normalize_optional_int,
    _normalize_slot_overrides,
    _normalize_specific_day_windows,
    _normalize_text_list,
    _parse_date_str,
    _parse_time_hhmm,
    _prepend_no_think_message,
)
from planner_scheduling import build_schedule_data

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

try:
    import ollama
except Exception:
    ollama = None


CALENDAR_INTERFACE_MODEL_LOG_FILE = (
    str(os.getenv("CALENDAR_INTERFACE_MODEL_LOG_FILE", "calendar_planner_model_log.jsonl") or "").strip()
    or "calendar_planner_model_log.jsonl"
)

GEMMA_RAW_OUTPUT_LOG_FILE = (
    str(os.getenv("GEMMA_RAW_OUTPUT_LOG_FILE", "gemma_raw_output_log.jsonl") or "").strip()
    or "gemma_raw_output_log.jsonl"
)

_PLANNER_RUNTIME_LOG_SINK: ContextVar[Callable[[str], None] | None] = ContextVar(
    "planner_runtime_log_sink",
    default=None,
)


def set_planner_runtime_log_sink(sink: Callable[[str], None] | None) -> Token:
    return _PLANNER_RUNTIME_LOG_SINK.set(sink)


def reset_planner_runtime_log_sink(token: Token) -> None:
    _PLANNER_RUNTIME_LOG_SINK.reset(token)


def _planner_log(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return

    print(text)

    sink = _PLANNER_RUNTIME_LOG_SINK.get()
    if sink is None:
        return
    try:
        sink(text)
    except Exception:
        # Runtime log sinks must never interrupt planner execution.
        return


def _append_model_log(record):
    try:
        payload = dict(record or {})
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with open(CALENDAR_INTERFACE_MODEL_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str))
            handle.write("\n")
    except Exception:
        # Logging must never interrupt planner execution.
        return


def _append_gemma_raw_output_log(record):
    try:
        payload = dict(record or {})
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with open(GEMMA_RAW_OUTPUT_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str))
            handle.write("\n")
    except Exception:
        # Logging must never interrupt planner execution.
        return


def qwen_extract_json(raw_text: str):
    prompt = f"""
You are a JSON extraction engine.

Task:
- Extract ONLY the valid JSON object from the text below.
- If multiple JSON objects exist, return the most complete one.
- If JSON is invalid, FIX it.
- Remove any explanation, thinking, or extra text.

STRICT RULES:
- Output ONLY valid JSON
- No markdown
- No explanation
- No extra text
Return JSON EXACTLY matching the schema keys present in the input. Do not invent fields.
the json in the text usually will start with ```json and ends with ```. convert what is enclosed in it.

Text:
{raw_text}
"""

    response = safe_ollama_call(prompt)

    if not isinstance(response, dict):
        return None

    return response


class _GemmaGeminiClient:
    def __init__(self, model_name):
        if genai is None:
            raise RuntimeError("google-genai is required for GEMMA_BACKEND=gemini. Install with: pip install google-genai")

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in your environment or .env file.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = (
            "You are the primary study-plan orchestrator. "
            "You can call tools, request clarification, and then finalize constraints "
            "for scheduling. Return JSON only when asked."
        )
        

    def generate_json(self, prompt, timeout_seconds, max_output_tokens):
        full_prompt = self.system_instruction + "\n\n" + prompt
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            ),
        )

        raw_text = str(getattr(response, "text", "") or "")
        qwen_parsed = None

        # 🔹 Step 1: try direct extraction
        
        parsed = _extract_first_json_object(raw_text)

        # 🔹 Step 2: fallback to Qwen if needed
        if (not isinstance(parsed, dict)) or parsed == {}:
            _planner_log("[Gemma] JSON extraction failed → invoking Qwen")

            qwen_parsed = qwen_extract_json(raw_text)

            if isinstance(qwen_parsed, dict):
                parsed = qwen_parsed
            else:
                parsed = _extract_first_json_object(str(qwen_parsed or ""))

        # 🔹 Step 3: final safety
        if not isinstance(parsed, dict):
            _planner_log("[Gemma] Qwen failed → returning empty dict")
            parsed = {}

        _append_gemma_raw_output_log({"result":f"[Planner] Gemini raw_text length={len(raw_text)} qwen_res={qwen_parsed},json={parsed}"})
        return {
            "raw_text": raw_text,
            "parsed": parsed,
        }


class _GemmaOllamaClient:
    def __init__(self, model_name):
        if ollama is None:
            raise RuntimeError("ollama package is required for GEMMA_BACKEND=ollama.")

        self.model_name = str(model_name or GEMMA_OLLAMA_MODEL).strip() or GEMMA_OLLAMA_MODEL
        self.client = ollama.Client(host=OLLAMA_BASE_URL) if hasattr(ollama, "Client") else None

    def _chat_once(self, messages):
        chat_kwargs = {
            "model": self.model_name,
            "messages": list(messages or []),
            "format": "json",
        }

        if DISABLE_OLLAMA_THINKING:
            chat_kwargs["think"] = False

        try:
            if self.client is not None:
                response = self.client.chat(**chat_kwargs)
            else:
                response = ollama.chat(**chat_kwargs)
            return response
        except TypeError:
            chat_kwargs.pop("think", None)
            if DISABLE_OLLAMA_THINKING:
                chat_kwargs["messages"] = _prepend_no_think_message(chat_kwargs.get("messages", []))
            if self.client is not None:
                return self.client.chat(**chat_kwargs)
            return ollama.chat(**chat_kwargs)

    def generate_json(self, prompt, timeout_seconds, max_output_tokens):
        # timeout_seconds and max_output_tokens are currently not used directly by Ollama SDK.
        # They remain in the signature to keep backend interface uniform.
        _ = timeout_seconds
        _ = max_output_tokens
        response = self._chat_once(messages=[{"role": "user", "content": prompt}])

        content = ""
        if isinstance(response, dict):
            content = str(response.get("message", {}).get("content", "") or "")
        else:
            message = getattr(response, "message", None)
            if isinstance(message, dict):
                content = str(message.get("content", "") or "")
            else:
                content = str(getattr(message, "content", "") or "")

        return {
            "raw_text": content,
            "parsed": _extract_first_json_object(content),
        }


class GemmaPlannerAgent:
    def __init__(self):
        backend = _env_text("GEMMA_BACKEND", GEMMA_BACKEND or "gemini").lower() or "gemini"
        self.backend = backend
        self._run_counter = 0

        if backend == "ollama":
            model_name = _env_text("GEMMA_OLLAMA_MODEL", GEMMA_OLLAMA_MODEL)
            self.client = _GemmaOllamaClient(model_name=model_name)
            _planner_log(f"[Planner] Gemma backend: ollama ({model_name})")
        else:
            model_name = _env_text("GEMMA_GEMINI_MODEL", GEMMA_GEMINI_MODEL)
            self.client = _GemmaGeminiClient(model_name=model_name)
            _planner_log(f"[Planner] Gemma backend: gemini ({model_name})")

    def _build_prompt(self, session_context, history, turn_index, max_turns):
        turn_number = int(turn_index) + 1
        turns_remaining = max(1, int(max_turns) - int(turn_index))

        schema = {
            "action": "tool_call | ask_user | final",
            "tool_name": "if action=tool_call",
            "arguments": "if action=tool_call",
            "question": "if action=ask_user",
            "final_summary": "if action=final",
            "updated_constraints": {
                "daily_max_minutes": "optional int",
                "min_slot_minutes": "optional int",
                "max_slot_minutes": "optional int",
                "include_weekends": "optional bool",
                "blocked_weekdays": "optional list[int weekday 0=Mon..6=Sun]",
                "blocked_dates": "optional list[YYYY-MM-DD]",
                "study_window_start": "optional HH:MM",
                "study_window_end": "optional HH:MM",
                "specific_day_windows": "optional list[{date,start,end,priority}]",
                "slot_overrides": "optional list[{chunk_id,preferred_date,preferred_time_of_day,priority}]",
                "chunk_order_mode": "optional prerequisite|difficulty_asc|difficulty_desc|priority|deadline",
                "buffer_days": "optional int",
                "additional_constraints": "optional string"
            },
            "updated_chunk_prerequisites": [
                {
                    "chunk_id": "chunk id",
                    "add": ["prereq 1", "prereq 2"],
                    "remove": ["prereq to remove"],
                    "reason": "short reason"
                }
            ],
            "needs_regeneration": "bool",
            "reason": "short reason"
        }

        tools = [
            {
                "name": "get_plan_context",
                "arguments": {},
                "description": "Returns coverage metrics, slot count, and active constraints.",
            },
            {
                "name": "get_uncovered_chunks",
                "arguments": {"limit": 10},
                "description": "Returns uncovered chunk list.",
            },
            {
                "name": "search_chunks",
                "arguments": {"query": "text", "top_k": 6},
                "description": "Searches only relevant chunks by query.",
            },
            {
                "name": "get_chunk_details",
                "arguments": {"chunk_ids": ["id1", "id2"]},
                "description": "Returns detailed metadata and preview for selected chunks.",
            },
            {
                "name": "get_slot_details",
                "arguments": {"limit": 8},
                "description": "Returns summary of existing slots.",
            },
            {
                "name": "get_specific_day_windows",
                "arguments": {},
                "description": "Returns current specific-day time windows.",
            },
            {
                "name": "set_specific_day_window",
                "arguments": {
                    "date": "YYYY-MM-DD",
                    "start_time": "HH:MM optional",
                    "end_time": "HH:MM optional",
                    "period": "morning|afternoon|evening optional",
                    "priority": "hard|soft optional"
                },
                "description": "Adds or updates a date-specific scheduling window.",
            },
            {
                "name": "clear_specific_day_window",
                "arguments": {"date": "YYYY-MM-DD optional"},
                "description": "Clears one date override or all if date is omitted.",
            },
            {
                "name": "set_chunk_hints",
                "arguments": {
                    "updates": [
                        {
                            "chunk_id": "id",
                            "priority": "1-5 optional",
                            "skip": "bool optional",
                            "must_schedule_before": "YYYY-MM-DD optional",
                            "preferred_date": "YYYY-MM-DD optional",
                            "preferred_time_of_day": "morning|afternoon|evening|night|any optional"
                        }
                    ]
                },
                "description": "Sets chunk-level scheduling hints (priority, skip, deadline, preferred timing).",
            },
            {
                "name": "propose_slot_overrides",
                "arguments": {
                    "overrides": [
                        {
                            "chunk_id": "id",
                            "preferred_date": "YYYY-MM-DD optional",
                            "preferred_time_of_day": "morning|afternoon|evening|night|any optional",
                            "priority": "hard|soft optional"
                        }
                    ]
                },
                "description": "Adds chunk-level slot placement preferences for the scheduler second pass.",
            },
        ]

        prompt = f"""

You are the primary planner model (Gemma).
You can inspect only required chunks using tools.
If context is insufficient, ask a direct clarification question.

Turn context:
{json.dumps({"current_turn": turn_number, "max_turns": int(max_turns), "turns_remaining_including_this_turn": turns_remaining}, ensure_ascii=True)}

Session context:
{json.dumps(session_context, ensure_ascii=True)}

Tool call history:
{json.dumps(history, ensure_ascii=True)}

Available tools:
{json.dumps(tools, ensure_ascii=True)}

Return JSON only with this schema:
{json.dumps(schema, ensure_ascii=True)}

Rules:
- Default to metadata-level tools first.
- Call get_chunk_details only if metadata is insufficient.
- Never call get_chunk_details for more than 3 chunk_ids in one turn.
- Never call search_chunks with an empty query.
- Call tools when you need more evidence.
- Ask clarification only if needed.
- Final response may update constraints to regenerate schedule.
- You may also update chunk prerequisites when a topic depends on missing material.
- Prefer adding prerequisites over removing them unless removal is clearly justified.
- If session_context includes locked_constraints, do not propose updates for those keys.
- For specific requests like "on 25th morning slot", call set_specific_day_window.
- If period is mentioned but exact time is missing, default to: morning=08:00-12:00, afternoon=13:00-17:00, evening=18:00-22:30.
- For "skip this topic" or "prioritize this chunk", use set_chunk_hints.
- For "put hardest topic on Saturday morning", use propose_slot_overrides.
- For feedback like "no sessions on Friday", set updated_constraints.blocked_weekdays using weekday indices (Mon=0..Sun=6).
- For feedback like "no sessions on 2026-04-23" or "no sessions on 23rd", set updated_constraints.blocked_dates.
- If turns_remaining_including_this_turn <= 1, return action="final". Do not call tools.
- If you call set_specific_day_window, clear_specific_day_window, set_chunk_hints, or propose_slot_overrides, your next response must be action="final" with needs_regeneration=true.
- Use scheduling_hints values (priority, skip, must_schedule_before, preferred_date, preferred_time_of_day) when available.
- For feedback restricting session lengths (e.g. "keep sessions short"):
  Set updated_constraints for "max_slot_minutes" and "daily_max_minutes" accordingly and set needs_regeneration=true.
- If shorter sessions may increase total days beyond end_date, that is acceptable — do not block the update.

in the end for the final json result start with ```json and end with ```
"""
        return prompt

    def _call_json(self, prompt, run_id=None, turn_index=None, session_context=None):
        started = time_module.perf_counter()
        raw_text = ""
        try:
            result = self.client.generate_json(
                prompt=prompt,
                timeout_seconds=GEMMA_PLANNER_TIMEOUT_SECONDS,
                max_output_tokens=GEMMA_PLANNER_MAX_OUTPUT_TOKENS,
            )
            if isinstance(result, dict) and "parsed" in result:
                parsed = result.get("parsed")
                raw_text = str(result.get("raw_text", "") or "")
            else:
                parsed = result

        except Exception as exc:
            _planner_log(f"[Planner] Gemma call failed: {exc}")
            context = dict(session_context or {})
            phase = context.get("phase") or f"main_round_{context.get('round', 'unknown')}"
            _append_model_log({
                "model": "gemma",
                "event": "call_error",
                "backend": self.backend,
                "run_id": run_id,
                "turn": turn_index,
                "session_id": context.get("session_id"),
                "phase": phase,
                "error": str(exc),
            })
            _append_gemma_raw_output_log({
                "model": "gemma",
                "event": "call_error",
                "backend": self.backend,
                "run_id": run_id,
                "turn": turn_index,
                "session_id": context.get("session_id"),
                "phase": phase,
                "prompt": prompt,
                "raw_output": raw_text,
                "error": str(exc),
            })
            return {}

        elapsed = time_module.perf_counter() - started
        _planner_log(f"[Planner] Gemma call completed in {elapsed:.1f}s")
        context = dict(session_context or {})
        phase = context.get("phase") or f"main_round_{context.get('round', 'unknown')}"
        if not isinstance(parsed, dict):
            parsed = _extract_first_json_object(str(parsed or ""))

        _append_gemma_raw_output_log({
            "model": "gemma",
            "event": "turn_raw_output",
            "backend": self.backend,
            "run_id": run_id,
            "turn": turn_index,
            "elapsed_seconds": round(float(elapsed), 3),
            "session_id": context.get("session_id"),
            "phase": phase,
            "prompt": prompt,
            "raw_output": raw_text,
            "parsed_response": parsed,
            "action": str(parsed.get("action", "") or "").strip().lower(),
            "updated_constraints": dict(parsed.get("updated_constraints") or {}),
        })

        _append_model_log({
            "model": "gemma",
            "event": "turn_response",
            "backend": self.backend,
            "run_id": run_id,
            "turn": turn_index,
            "elapsed_seconds": round(float(elapsed), 3),
            "session_id": context.get("session_id"),
            "phase": phase,
            "response": parsed,
        })
        return parsed

    @staticmethod
    def _history_has_mutating_tool_call(history):
        mutating_tools = {
            "set_specific_day_window",
            "clear_specific_day_window",
            "set_chunk_hints",
            "propose_slot_overrides",
        }
        for row in history:
            if not isinstance(row, dict):
                continue
            action = row.get("assistant_action", {})
            if not isinstance(action, dict):
                continue
            if str(action.get("action", "")).strip().lower() != "tool_call":
                continue
            tool_name = str(action.get("tool_name", "")).strip()
            if tool_name in mutating_tools:
                return True
        return False

    @staticmethod
    def _preserve_hint_values_for_history(tool_name, raw_result, compact_result):
        # Ensure chunk scheduling hints remain readable to Gemma even when
        # generic compaction settings change.
        if tool_name != "get_chunk_details":
            return compact_result

        if not isinstance(raw_result, list) or not isinstance(compact_result, list):
            return compact_result

        for index, raw_row in enumerate(raw_result):
            if index >= len(compact_result):
                break

            compact_row = compact_result[index]
            if not isinstance(raw_row, dict) or not isinstance(compact_row, dict):
                continue

            compact_row["scheduling_hints"] = _normalize_chunk_hints(raw_row.get("scheduling_hints", {}))

        return compact_result

    @staticmethod
    def _coverage_gap_status(session_context):
        coverage = dict(dict(session_context or {}).get("coverage") or {})
        uncovered = _normalize_int(coverage.get("uncovered_chunks"), 0, minimum=0)
        partial = _normalize_int(coverage.get("partial_chunks"), 0, minimum=0)
        return bool(uncovered > 0 or partial > 0), uncovered, partial

    @staticmethod
    def _auto_gap_recovery_updates(session_context):
        context = dict(session_context or {})
        constraints = dict(context.get("constraints") or {})
        locked = {str(item) for item in list(context.get("locked_constraints") or [])}

        has_gap, uncovered, partial = GemmaPlannerAgent._coverage_gap_status(context)
        if not has_gap:
            return {}, []

        notes = [f"Coverage gap detected (uncovered={uncovered}, partial={partial})."]

        current_buffer = _normalize_int(constraints.get("buffer_days"), 1, minimum=0, maximum=14)
        if "buffer_days" not in locked and current_buffer > 0:
            updated = max(0, current_buffer - 1)
            notes.append(
                f"Auto-adjustment: reduce buffer_days from {current_buffer} to {updated} to free extra scheduling time."
            )
            return {"buffer_days": updated}, notes

        current_daily = _normalize_optional_int(constraints.get("daily_max_minutes"), minimum=30, maximum=720)
        if "daily_max_minutes" not in locked:
            base_daily = 30 if current_daily is None else int(current_daily)
            bumped = min(720, base_daily + 10)
            if bumped > base_daily:
                notes.append(
                    f"Auto-adjustment: increase daily_max_minutes from {base_daily} to {bumped} to close remaining minutes."
                )
                return {"daily_max_minutes": bumped}, notes

        notes.append(
            "No safe automatic constraint update available because candidate fields are locked or already at limits."
        )
        return {}, notes

    def _build_auto_final_action(self, history, toolbox, reason, session_context=None):
        tool_updates = (
            toolbox.consume_pending_constraint_updates()
            if hasattr(toolbox, "consume_pending_constraint_updates")
            else {}
        )
        mutating_tool_used = self._history_has_mutating_tool_call(history)
        needs_regeneration = bool(tool_updates) or mutating_tool_used
        updated_constraints = {}

        has_gap, uncovered, partial = self._coverage_gap_status(session_context)
        gap_notes = []
        if has_gap and not needs_regeneration:
            updated_constraints, gap_notes = self._auto_gap_recovery_updates(session_context)
            if updated_constraints:
                needs_regeneration = True

        if needs_regeneration:
            if updated_constraints:
                summary = "Applied automatic coverage-gap adjustments. Regenerate schedule to reflect changes."
                if gap_notes:
                    summary = (summary + " " + " ".join(gap_notes)).strip()
            else:
                summary = "Applied tool-driven planning updates. Regenerate schedule to reflect changes."
        else:
            if has_gap:
                summary = (
                    f"Coverage gap remains (uncovered={uncovered}, partial={partial}). "
                    "No automatic change was applied; keep current schedule."
                )
                if gap_notes:
                    summary = (summary + " " + " ".join(gap_notes)).strip()
            else:
                summary = "No additional model update. Keep current schedule."

        reason_text = str(reason or "auto_finalized").strip() or "auto_finalized"
        if has_gap and not needs_regeneration:
            reason_text = reason_text + "; unresolved_coverage_gap_no_action"
        if updated_constraints:
            reason_text = reason_text + "; unresolved_coverage_gap_auto_adjustment"

        return {
            "action": "final",
            "final_summary": summary,
            "updated_constraints": updated_constraints,
            "needs_regeneration": needs_regeneration,
            "reason": reason_text,
            "_history": history,
            "_tool_constraint_updates": tool_updates,
        }

    def run(self, session_context, toolbox, max_turns=None, allow_user_input=True):
        if max_turns is None:
            max_turns = GEMMA_PLANNER_MAX_TURNS
        max_turns = max(1, int(max_turns))
        self._run_counter += 1
        run_id = self._run_counter

        history = []
        started = time_module.perf_counter()
        stop_reason = "agent_max_turns_exhausted"

        for turn_index in range(max_turns):
            elapsed_runtime = time_module.perf_counter() - started
            if elapsed_runtime > GEMMA_PLANNER_MAX_RUNTIME_SECONDS:
                _planner_log(
                    "[Planner] Gemma runtime budget reached "
                    f"({elapsed_runtime:.1f}s > {GEMMA_PLANNER_MAX_RUNTIME_SECONDS}s)."
                )
                stop_reason = "runtime_budget_exceeded"
                break

            _planner_log(f"[Planner] Gemma turn {turn_index + 1}/{max_turns}")
            prompt = self._build_prompt(
                session_context=session_context,
                history=history,
                turn_index=turn_index,
                max_turns=max_turns,
            )
            action = self._call_json(
                prompt,
                run_id=run_id,
                turn_index=turn_index + 1,
                session_context=session_context,
            )
            if not isinstance(action, dict) or not action:
                stop_reason = "parse_empty_response"
                break

            kind = str(action.get("action", "")).strip().lower()

            if kind == "tool_call":
                tool_name = str(action.get("tool_name", "")).strip()
                arguments = action.get("arguments", {})
                result = toolbox.execute_tool(tool_name, arguments)
                compact_result = _compact_for_llm(result)
                compact_result = self._preserve_hint_values_for_history(tool_name, result, compact_result)
                history.append({
                    "assistant_action": {
                        "action": "tool_call",
                        "tool_name": tool_name,
                        "arguments": arguments,
                    },
                    "tool_result": compact_result,
                })

                if turn_index >= max_turns - 1:
                    auto_final = self._build_auto_final_action(
                        history=history,
                        toolbox=toolbox,
                        reason="auto_final_after_last_tool_call",
                        session_context=session_context,
                    )
                    _append_model_log({
                        "model": "gemma",
                        "event": "run_auto_final",
                        "backend": self.backend,
                        "run_id": run_id,
                        "session_id": dict(session_context or {}).get("session_id"),
                        "phase": dict(session_context or {}).get("phase") or f"main_round_{dict(session_context or {}).get('round', 'unknown')}",
                        "final_action": auto_final,
                    })
                    return auto_final

                continue

            if kind == "ask_user":
                question = str(action.get("question", "")).strip() or "Need clarification for planning."
                if allow_user_input:
                    answer = input(f"\nPlanner clarification: {question}\nYour answer: ").strip()
                else:
                    answer = ""
                history.append({
                    "assistant_action": {
                        "action": "ask_user",
                        "question": question,
                    },
                    "user_answer": answer,
                })

                if not allow_user_input:
                    stop_reason = "ask_user_in_non_interactive_mode"
                    break

                if turn_index >= max_turns - 1:
                    stop_reason = "asked_user_on_last_turn"
                    break

                continue

            if kind == "final":
                action["_history"] = history
                if not isinstance(action.get("updated_constraints"), dict):
                    action["updated_constraints"] = {}
                if hasattr(toolbox, "consume_pending_constraint_updates"):
                    action["_tool_constraint_updates"] = toolbox.consume_pending_constraint_updates()
                else:
                    action["_tool_constraint_updates"] = {}

                has_mutating_tool = self._history_has_mutating_tool_call(history)
                if not str(action.get("final_summary", "")).strip():
                    if action.get("_tool_constraint_updates") or has_mutating_tool:
                        action["final_summary"] = "Applied tool-driven planning updates. Regenerate schedule to reflect changes."
                    else:
                        action["final_summary"] = "No additional model update. Keep current schedule."

                action["needs_regeneration"] = _normalize_bool(action.get("needs_regeneration"), default=False)
                if not action["needs_regeneration"] and bool(action.get("updated_constraints")):
                    action["needs_regeneration"] = True
                if not action["needs_regeneration"] and (action.get("_tool_constraint_updates") or has_mutating_tool):
                    action["needs_regeneration"] = True

                action["reason"] = str(action.get("reason") or "model_final").strip() or "model_final"

                has_gap, uncovered, partial = self._coverage_gap_status(session_context)
                has_any_updates = bool(action.get("updated_constraints")) or bool(action.get("_tool_constraint_updates"))
                if has_gap and not action["needs_regeneration"] and not has_any_updates:
                    gap_note = (
                        f"Coverage gap remains (uncovered={uncovered}, partial={partial}); "
                        "no constraint update was applied."
                    )
                    summary_text = str(action.get("final_summary", "")).strip()
                    if gap_note.lower() not in summary_text.lower():
                        action["final_summary"] = (summary_text + " " + gap_note).strip()
                    if "unresolved_coverage_gap_no_action" not in action["reason"]:
                        action["reason"] = action["reason"] + "; unresolved_coverage_gap_no_action"

                _append_model_log({
                    "model": "gemma",
                    "event": "run_final",
                    "backend": self.backend,
                    "run_id": run_id,
                    "session_id": dict(session_context or {}).get("session_id"),
                    "phase": dict(session_context or {}).get("phase") or f"main_round_{dict(session_context or {}).get('round', 'unknown')}",
                    "final_action": action,
                })
                return action

            history.append({
                "assistant_action": action,
                "tool_result": {"error": "unknown action"},
            })

            if turn_index >= max_turns - 1:
                stop_reason = "unknown_action_on_last_turn"

        auto_final = self._build_auto_final_action(
            history=history,
            toolbox=toolbox,
            reason=stop_reason,
            session_context=session_context,
        )
        _append_model_log({
            "model": "gemma",
            "event": "run_auto_final",
            "backend": self.backend,
            "run_id": run_id,
            "session_id": dict(session_context or {}).get("session_id"),
            "phase": dict(session_context or {}).get("phase") or f"main_round_{dict(session_context or {}).get('round', 'unknown')}",
            "final_action": auto_final,
        })
        return auto_final


def apply_constraint_updates(current, updates, locked_keys=None):
    if not isinstance(updates, dict):
        return _normalize_constraints(current)

    merged = dict(current)
    locked = {str(key) for key in set(locked_keys or set())}
    for key, value in updates.items():
        if key not in SCHEDULING_CONSTRAINT_KEYS:
            continue
        if key in locked:
            continue

        if key in {"daily_max_minutes", "min_slot_minutes", "max_slot_minutes", "buffer_days"}:
            if key == "buffer_days":
                merged[key] = _normalize_int(value, merged.get(key, DEFAULT_CONSTRAINTS[key]), minimum=0)
            else:
                merged[key] = _normalize_optional_int(value, minimum=0)
            continue

        if key == "include_weekends":
            merged[key] = _normalize_bool(value, default=merged.get(key, True))
            continue

        if key == "blocked_weekdays":
            merged[key] = _normalize_blocked_weekdays(value)
            continue

        if key == "blocked_dates":
            merged[key] = _normalize_blocked_dates(value)
            continue

        if key in {"study_window_start", "study_window_end"}:
            fallback = _parse_time_hhmm(merged.get(key), _parse_time_hhmm(DEFAULT_CONSTRAINTS[key], time(hour=18, minute=0)))
            merged[key] = _parse_time_hhmm(value, fallback).strftime("%H:%M")
            continue

        if key == "specific_day_windows":
            merged[key] = _normalize_specific_day_windows(value)
            continue

        if key == "slot_overrides":
            merged[key] = _normalize_slot_overrides(value)
            continue

        if key == "chunk_order_mode":
            mode = str(value or "").strip().lower()
            merged[key] = mode if mode in CHUNK_ORDER_MODES else merged.get("chunk_order_mode", "prerequisite")
            continue

        merged[key] = str(value or "").strip()

    return _normalize_constraints(merged)


def apply_chunk_prerequisite_updates(chunks, updates):
    if not isinstance(updates, list) or not chunks:
        return False

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    changed = False

    for row in updates:
        if not isinstance(row, dict):
            continue

        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            continue

        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue

        current = list(chunk.prerequisites)
        current_seen = {item.lower(): item for item in current}

        for prereq in _normalize_text_list(row.get("add")):
            key = prereq.lower()
            if key not in current_seen:
                current.append(prereq)
                current_seen[key] = prereq
                changed = True

        remove_keys = {item.lower() for item in _normalize_text_list(row.get("remove"))}
        if remove_keys:
            filtered = [item for item in current if item.lower() not in remove_keys]
            if len(filtered) != len(current):
                current = filtered
                changed = True

        chunk.prerequisites = current

    return changed


def collect_user_plan_feedback():
    answer = input("\nProvide feedback before scheduling? (y/N): ").strip().lower()
    if answer not in {"y", "yes"}:
        return ""

    feedback = input("Enter feedback for Gemma (topics, pace, prerequisites, or timing): ").strip()
    return feedback


def collect_additional_plan_feedback(round_index, max_rounds):
    answer = input(f"\nAnything else to adjust? ({round_index}/{max_rounds}) (y/N): ").strip().lower()
    if answer not in {"y", "yes"}:
        return ""

    feedback = input("Enter additional feedback: ").strip()
    return feedback


def _weekday_name(index):
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if 0 <= int(index) < len(names):
        return names[int(index)]
    return str(index)


def _feedback_corpus(user_feedback, feedback_history=None):
    rows = []

    text = str(user_feedback or "").strip()
    if text:
        rows.append(text)

    if isinstance(feedback_history, list):
        for item in feedback_history:
            item_text = str(item or "").strip()
            if item_text:
                rows.append(item_text)

    return " | ".join(rows)


def _normalize_reference_dates(reference_dates):
    rows = []
    if isinstance(reference_dates, (list, tuple, set)):
        rows = list(reference_dates)
    elif reference_dates is not None:
        rows = [reference_dates]

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
        normalized.append(parsed)

    return sorted(normalized)


def _resolve_day_number_to_date(day_number, reference_dates):
    candidates = [item for item in _normalize_reference_dates(reference_dates) if int(item.day) == int(day_number)]
    if not candidates:
        return None

    today_value = date.today()
    future = [item for item in candidates if item >= today_value]
    if future:
        return future[0].isoformat()
    return candidates[0].isoformat()


def _extract_blocked_dates_from_text(text, reference_dates=None):
    lowered = str(text or "").strip().lower()
    if not lowered:
        return []

    normalized = lowered.replace("don't", "do not").replace("dont", "do not")
    neg_markers = ["no", "avoid", "skip", "without", "exclude", "do not", "off", "blocked"]
    session_markers = ["session", "sessions", "study", "slot", "slots", "anything"]

    blocked = []
    seen = set()

    def _mark(iso_value):
        value = str(iso_value or "").strip()
        parsed = _parse_date_str(value)
        if parsed is None:
            return
        normalized_iso = parsed.isoformat()
        if normalized_iso in seen:
            return
        seen.add(normalized_iso)
        blocked.append(normalized_iso)

    # Direct ISO dates in negative scheduling context.
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", normalized):
        start = max(0, int(match.start()) - 80)
        end = min(len(normalized), int(match.end()) + 80)
        window = normalized[start:end]
        if any(marker in window for marker in neg_markers) and any(marker in window for marker in session_markers):
            _mark(match.group(0))

    # Ordinal days like "23rd" in negative scheduling context.
    for match in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)\b", normalized):
        start = max(0, int(match.start()) - 80)
        end = min(len(normalized), int(match.end()) + 80)
        window = normalized[start:end]
        if not (any(marker in window for marker in neg_markers) and any(marker in window for marker in session_markers)):
            continue
        day_number = _normalize_int(match.group(1), 0, minimum=1, maximum=31)
        resolved = _resolve_day_number_to_date(day_number, reference_dates=reference_dates)
        if resolved:
            _mark(resolved)

    # Plain day number with "on" prefix, e.g. "on 23".
    for match in re.finditer(r"\bon\s+(\d{1,2})\b", normalized):
        start = max(0, int(match.start()) - 80)
        end = min(len(normalized), int(match.end()) + 80)
        window = normalized[start:end]
        if not (any(marker in window for marker in neg_markers) and any(marker in window for marker in session_markers)):
            continue
        day_number = _normalize_int(match.group(1), 0, minimum=1, maximum=31)
        resolved = _resolve_day_number_to_date(day_number, reference_dates=reference_dates)
        if resolved:
            _mark(resolved)

    _append_gemma_raw_output_log({
                "result" : blocked,
                "ans": "it is the fallback"
            })
    
    return sorted(blocked)


def _extract_blocked_weekdays_from_text(text):
    lowered = str(text or "").strip().lower()
    if not lowered:
        return []

    normalized = lowered.replace("don't", "do not").replace("dont", "do not")
    blocked = set()

    alias_rows = sorted(WEEKDAY_NAME_TO_INDEX.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, day_index in alias_rows:
        day_pattern = re.escape(alias)
        if not re.search(rf"\b{day_pattern}\b", normalized):
            continue

        direct_negative = re.search(
            rf"\b(?:no|avoid|skip|without|exclude|do\s+not)\b[^.\n]{{0,40}}\b{day_pattern}\b",
            normalized,
        )
        no_session_phrase = re.search(
            rf"\b(?:no|avoid|skip|without|exclude|do\s+not)\b[^.\n]{{0,60}}"
            rf"\b(?:session|sessions|study|slot|slots)\b[^.\n]{{0,60}}\b{day_pattern}\b",
            normalized,
        )
        day_then_session = re.search(
            rf"\b{day_pattern}\b[^.\n]{{0,30}}\b(?:off|free|blocked|no\s+sessions?|no\s+study|no\s+slots?)\b",
            normalized,
        )
        no_day_session = re.search(
            rf"\bno\b[^.\n]{{0,20}}\b{day_pattern}\b[^.\n]{{0,20}}\b(?:sessions?|study|slots?)\b",
            normalized,
        )

        if direct_negative or no_session_phrase or day_then_session or no_day_session:
            blocked.add(int(day_index))

    return sorted(list(blocked))


def _extract_daily_max_minutes_from_text(text):
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None

    normalized = normalized.replace("don't", "do not").replace("dont", "do not")
    daily_pattern = r"(?:per\s+day|a\s+day|each\s+day|any\s+day|daily)"
    limit_pattern = (
        r"(?:no\s+more\s+than|at\s+most|max(?:imum)?|limit(?:ed)?\s+to|"
        r"only|do\s+not\s+want\s+more\s+than|do\s+not\s+want\s+over|up\s+to)"
    )

    amount_pattern = r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m)\b"

    def _duration_minutes_from_window(window_text):
        # Support compound durations like "1 hr 15 mins" in addition to single-unit values.
        compound = re.search(
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)\s*(?:and\s*)?(\d+(?:\.\d+)?)?\s*(minutes?|mins?|min|m)?\b",
            window_text,
        )
        if compound:
            try:
                hours = float(compound.group(1))
                mins = float(compound.group(3)) if compound.group(3) and compound.group(4) else 0.0
                total = int(round(hours * 60 + mins))
                return _normalize_optional_int(total, minimum=0)
            except Exception:
                pass

        single = re.search(amount_pattern, window_text)
        if not single:
            return None

        try:
            amount = float(single.group(1))
        except Exception:
            return None

        unit = str(single.group(2) or "").strip().lower()
        minutes = int(round(amount * 60)) if unit.startswith("h") else int(round(amount))
        return _normalize_optional_int(minutes, minimum=0)

    for match in re.finditer(amount_pattern, normalized):
        window_start = max(0, int(match.start()) - 80)
        window_end = min(len(normalized), int(match.end()) + 80)
        window = normalized[window_start:window_end]

        has_daily_context = bool(re.search(daily_pattern, window))
        has_limit_context = bool(re.search(limit_pattern, window))
        has_study_context = bool(re.search(r"\b(?:study|session|sessions)\b", window))

        if not has_daily_context:
            continue
        if not has_limit_context and not has_study_context:
            continue

        minutes = _duration_minutes_from_window(window)
        if minutes is None:
            continue

        return minutes

    return None


def _feedback_requests_from_text(text, reference_dates=None):
    normalized = str(text or "").strip().lower().replace("don't", "do not").replace("dont", "do not")
    blocked_weekdays = _extract_blocked_weekdays_from_text(normalized)
    blocked_dates = _extract_blocked_dates_from_text(normalized, reference_dates=reference_dates)
    daily_max_minutes = _extract_daily_max_minutes_from_text(normalized)

    avoid_weekends = bool(
        re.search(r"\b(?:weekdays\s+only|no\s+weekends?|avoid\s+weekends?|skip\s+weekends?|do\s+not\s+want\s+weekends?)\b", normalized)
    )
    include_weekends = bool(
        re.search(r"\b(?:include\s+weekends?|allow\s+weekends?|use\s+weekends?|study\s+on\s+weekends?)\b", normalized)
    )

    return {
        "blocked_weekdays": blocked_weekdays,
        "blocked_dates": blocked_dates,
        "daily_max_minutes": daily_max_minutes,
        "avoid_weekends": avoid_weekends,
        "include_weekends": include_weekends,
    }


def derive_constraint_updates_from_user_feedback(user_feedback, feedback_history=None, current_constraints=None, reference_dates=None):
    corpus = _feedback_corpus(user_feedback=user_feedback, feedback_history=feedback_history)
    if not corpus:
        return {}, []

    requests = _feedback_requests_from_text(corpus, reference_dates=reference_dates)
    current = dict(current_constraints or {})
    updates = {}
    notes = []

    current_blocked = _normalize_blocked_weekdays(current.get("blocked_weekdays", []))
    requested_blocked = _normalize_blocked_weekdays(requests.get("blocked_weekdays", []))
    if requested_blocked:
        merged_blocked = _normalize_blocked_weekdays(list(current_blocked) + list(requested_blocked))
        if merged_blocked != current_blocked:
            updates["blocked_weekdays"] = merged_blocked
            labels = ", ".join(_weekday_name(index) for index in merged_blocked)
            notes.append(f"Applied user day exclusions: {labels}.")

    current_blocked_dates = _normalize_blocked_dates(current.get("blocked_dates", []))
    requested_blocked_dates = _normalize_blocked_dates(requests.get("blocked_dates", []))
    if requested_blocked_dates:
        merged_blocked_dates = _normalize_blocked_dates(list(current_blocked_dates) + list(requested_blocked_dates))
        if merged_blocked_dates != current_blocked_dates:
            updates["blocked_dates"] = merged_blocked_dates
            notes.append("Applied user date exclusions: " + ", ".join(merged_blocked_dates) + ".")

    requested_daily_max = _normalize_optional_int(requests.get("daily_max_minutes"), minimum=0)
    if requested_daily_max is not None:
        current_daily_max = _normalize_optional_int(current.get("daily_max_minutes"), minimum=0)
        updates["daily_max_minutes"] = requested_daily_max
        if requested_daily_max != current_daily_max:
            notes.append(f"Applied user daily study cap: {requested_daily_max} minutes per day.")
        else:
            notes.append(f"Reaffirmed user daily study cap: {requested_daily_max} minutes per day.")

    if requests.get("avoid_weekends"):
        current_weekends = _normalize_bool(current.get("include_weekends"), default=True)
        if current_weekends:
            updates["include_weekends"] = False
            notes.append("Applied user request to avoid weekends.")

    if requests.get("include_weekends") and not requests.get("avoid_weekends"):
        current_weekends = _normalize_bool(current.get("include_weekends"), default=True)
        if not current_weekends:
            updates["include_weekends"] = True
            notes.append("Applied user request to include weekends.")

    return updates, notes


def _requested_last_day_start_time(user_feedback):
    text = str(user_feedback or "").strip().lower()
    if "last day" not in text:
        return None

    match = re.search(r"start(?:ing)?(?:\s+\w+){0,4}\s+at\s+(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return None

    hour = _normalize_int(match.group(1), -1, minimum=-1, maximum=23)
    minute = _normalize_int(match.group(2) if match.group(2) is not None else 0, 0, minimum=0, maximum=59)
    if hour < 0:
        return None
    return f"{hour:02d}:{minute:02d}"


def _extract_hhmm(value):
    text = str(value or "").strip()
    if not text:
        return None

    iso_match = re.search(r"T(\d{2}:\d{2})", text)
    if iso_match:
        return iso_match.group(1)

    plain_match = re.search(r"\b(\d{2}:\d{2})\b", text)
    if plain_match:
        return plain_match.group(1)

    return None


def _align_qwen_feedback_with_user_timing(payload, normalized):
    if not isinstance(normalized, dict):
        return normalized

    context = dict(payload or {})
    requested_start = _requested_last_day_start_time(context.get("user_feedback"))
    if not requested_start:
        return normalized

    plan = dict(context.get("plan") or {})
    slots = plan.get("slots")
    if not isinstance(slots, list) or not slots:
        return normalized

    starts_by_date = {}
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        start_time = str(slot.get("start_time") or "").strip()
        if not start_time:
            continue
        parsed_date = _parse_date_str(start_time[:10])
        if parsed_date is None:
            continue
        hhmm = _extract_hhmm(start_time)
        if not hhmm:
            continue
        starts_by_date.setdefault(parsed_date.isoformat(), []).append(hhmm)

    if not starts_by_date:
        return normalized

    last_date = sorted(starts_by_date.keys())[-1]
    starts_on_last_day = starts_by_date.get(last_date, [])
    if not starts_on_last_day:
        return normalized

    if any(start != requested_start for start in starts_on_last_day):
        return normalized

    filtered_adjustments = []
    dropped = False
    for suggestion in _normalize_text_list(normalized.get("suggested_adjustments")):
        lowered = suggestion.lower()
        times_in_text = re.findall(r"\b\d{2}:\d{2}\b", suggestion)
        mentions_last_day = "last day" in lowered or last_date in suggestion
        mentions_time_shift = "move" in lowered or "start" in lowered
        contradicts_request = bool(times_in_text) and any(slot_time != requested_start for slot_time in times_in_text)

        if mentions_last_day and mentions_time_shift and contradicts_request:
            dropped = True
            continue

        filtered_adjustments.append(suggestion)

    if not dropped:
        return normalized

    normalized["suggested_adjustments"] = filtered_adjustments
    if not filtered_adjustments:
        normalized["approval_ready"] = True
        severity = _normalize_int(normalized.get("severity"), 1, minimum=0, maximum=3)
        if severity > 1:
            normalized["severity"] = 1

    summary = str(normalized.get("summary", "")).strip()
    guard_note = f"Explicit user timing request is already satisfied on the last day ({requested_start})."
    if guard_note.lower() not in summary.lower():
        normalized["summary"] = (summary + " " + guard_note).strip()

    return normalized


def _feedback_compliance_issues(payload):
    context = dict(payload or {})
    corpus = _feedback_corpus(
        user_feedback=context.get("user_feedback"),
        feedback_history=context.get("feedback_history"),
    )
    if not corpus:
        return []

    requests = _feedback_requests_from_text(corpus)
    plan = dict(context.get("plan") or {})
    slots = plan.get("slots")
    if not isinstance(slots, list):
        slots = []

    slot_dates = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        start_time = str(slot.get("start_time") or "").strip()
        if not start_time:
            continue
        parsed = _parse_date_str(start_time[:10])
        if parsed is None:
            continue
        slot_dates.append(parsed.isoformat())

    issues = []

    requests = _feedback_requests_from_text(corpus, reference_dates=slot_dates)

    blocked_dates = _normalize_blocked_dates(requests.get("blocked_dates", []))
    if blocked_dates:
        blocked_set = set(blocked_dates)
        violating = sorted({value for value in slot_dates if value in blocked_set})
        if violating:
            issues.append(
                "User requested no sessions on specific dates, but slots exist on: " + ", ".join(violating) + "."
            )

    blocked_weekdays = _normalize_blocked_weekdays(requests.get("blocked_weekdays", []))
    if blocked_weekdays:
        blocked_dates = {}
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            start_time = str(slot.get("start_time") or "").strip()
            if not start_time:
                continue
            parsed_date = _parse_date_str(start_time[:10])
            if parsed_date is None:
                continue
            weekday = int(parsed_date.weekday())
            if weekday in blocked_weekdays:
                blocked_dates.setdefault(weekday, set()).add(parsed_date.isoformat())

        for weekday in blocked_weekdays:
            violated_dates = sorted(list(blocked_dates.get(weekday, set())))
            if violated_dates:
                issues.append(
                    f"User requested no sessions on {_weekday_name(weekday)}, but slots exist on: {', '.join(violated_dates)}."
                )

    if requests.get("avoid_weekends"):
        weekend_dates = set()
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            start_time = str(slot.get("start_time") or "").strip()
            if not start_time:
                continue
            parsed_date = _parse_date_str(start_time[:10])
            if parsed_date is None:
                continue
            if parsed_date.weekday() >= 5:
                weekend_dates.add(parsed_date.isoformat())

        if weekend_dates:
            issues.append(
                "User requested no weekend sessions, but weekend slots exist on: "
                + ", ".join(sorted(list(weekend_dates)))
                + "."
            )

    return issues


def _align_qwen_feedback_with_user_constraints(payload, normalized):
    if not isinstance(normalized, dict):
        return normalized

    issues = _feedback_compliance_issues(payload)
    if not issues:
        return normalized

    risks = _normalize_text_list(normalized.get("risks"))
    suggestions = _normalize_text_list(normalized.get("suggested_adjustments"))

    for issue in issues:
        if issue not in risks:
            risks.append(issue)

    for issue in issues:
        lowered = issue.lower()
        if "specific dates" in lowered:
            suggestion = "Set blocked_dates from user feedback and regenerate schedule to remove those dates."
        elif "no sessions on" in lowered:
            suggestion = "Rebuild slots with blocked_weekdays set from user feedback before approval."
        elif "no weekend sessions" in lowered:
            suggestion = "Set include_weekends=false and regenerate schedule to remove weekend slots."
        else:
            suggestion = "Regenerate schedule to satisfy explicit user feedback constraints."
        if suggestion not in suggestions:
            suggestions.append(suggestion)

    normalized["risks"] = risks
    normalized["suggested_adjustments"] = suggestions
    normalized["approval_ready"] = False
    normalized["severity"] = max(2, _normalize_int(normalized.get("severity"), 1, minimum=0, maximum=3))

    summary = str(normalized.get("summary", "")).strip()
    guard_note = "User feedback constraints are not fully incorporated in the current slot allocation."
    if guard_note.lower() not in summary.lower():
        normalized["summary"] = (summary + " " + guard_note).strip()

    return normalized


def _fallback_qwen_strengths(payload):
    context = dict(payload or {})
    plan = dict(context.get("plan") or {})
    coverage = dict(plan.get("coverage") or {})

    strengths = []

    try:
        coverage_pct = float(coverage.get("coverage_pct") or 0.0)
    except Exception:
        coverage_pct = 0.0

    uncovered = _normalize_int(coverage.get("uncovered_chunks"), 0, minimum=0)
    partial = _normalize_int(coverage.get("partial_chunks"), 0, minimum=0)

    if coverage_pct >= 100.0 and uncovered == 0 and partial == 0:
        strengths.append("Coverage is complete across all required chunks.")
    elif coverage_pct >= 90.0:
        strengths.append("Coverage is high across the scheduled date range.")

    slots = plan.get("slots")
    if isinstance(slots, list) and slots:
        strengths.append("The session sequence is structured with clear time bounds.")

    if not strengths:
        strengths.append("The plan has a coherent slot-based structure.")

    return strengths[:2]


def qwen_review_plan(payload):
    prompt = f"""
You are the plan reviewer model.

Return JSON only:
{{
  "summary": "one short paragraph",
  "strengths": [],
  "risks": [],
  "suggested_adjustments": [],
    "severity": 0,
  "approval_ready": true
}}

Plan payload:
{json.dumps(payload, ensure_ascii=True)}

Rules:
- Focus on coverage, prerequisite order, and feasibility.
- Keep response concise and actionable.
- Provide 1-3 concrete strengths; do not leave strengths empty.
- Set severity as: 0=excellent, 1=minor issues, 2=major issues, 3=critical and needs reset.
- If user_feedback is present, verify whether the current plan already satisfies that request before suggesting schedule changes.
- Never suggest a change that reverses explicit user timing feedback already reflected in the slots.
- If user_feedback or feedback_history contains explicit constraints and slots violate them, include this as a risk, set approval_ready=false, and use severity >= 2.
- If user_feedback requests a specific session duration limit but slots in the plan exceed that duration, set approval_ready=false, severity=2, and include it as a risk.
"""

    response = safe_ollama_call(prompt)
    context = dict(payload or {})
    phase = str(context.get("phase") or "qwen_review").strip() or "qwen_review"
    session_id = context.get("session_id")

    if not isinstance(response, dict) or not response:
        fallback = {
            "summary": "Qwen review unavailable. Use deterministic checks.",
            "strengths": ["Coverage check and prerequisites are attached per slot."],
            "risks": ["No model critique available."],
            "suggested_adjustments": ["Increase daily minutes or date range if uncovered chunks exist."],
            "severity": 2,
            "approval_ready": False,
        }
        _append_model_log({
            "model": "qwen",
            "event": "review_fallback",
            "phase": phase,
            "session_id": session_id,
            "payload": payload,
            "raw_response": response,
            "normalized_feedback": fallback,
        })
        return fallback

    normalized = {
        "summary": str(response.get("summary", "")).strip(),
        "strengths": _normalize_text_list(response.get("strengths")),
        "risks": _normalize_text_list(response.get("risks")),
        "suggested_adjustments": _normalize_text_list(response.get("suggested_adjustments")),
        "severity": _normalize_int(response.get("severity"), 1, minimum=0, maximum=3),
        "approval_ready": bool(response.get("approval_ready", False)),
    }
    normalized = _align_qwen_feedback_with_user_timing(payload, normalized)
    normalized = _align_qwen_feedback_with_user_constraints(payload, normalized)
    if not normalized.get("strengths"):
        normalized["strengths"] = _fallback_qwen_strengths(payload)
    _append_model_log({
        "model": "qwen",
        "event": "review",
        "phase": phase,
        "session_id": session_id,
        "payload": payload,
        "raw_response": response,
        "normalized_feedback": normalized,
    })
    return normalized


def _reset_constraints_for_severity(current_constraints, locked_keys):
    reset = _normalize_constraints(dict(DEFAULT_CONSTRAINTS))
    current = dict(current_constraints or {})
    for key in set(locked_keys or set()):
        if key in current:
            reset[key] = current.get(key)
    return _normalize_constraints(reset)


def _plan_snapshot(slots, coverage, constraints=None):
    preview_slots = []
    for slot in slots[:12]:
        preview_slots.append({
            "start_time": slot.get("start_time"),
            "end_time": slot.get("end_time"),
            "duration_minutes": slot.get("duration_minutes"),
            "difficulty": slot.get("difficulty"),
            "topics": [item.get("topic") for item in slot.get("items", [])],
            "items": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "topic": item.get("topic"),
                    "focus_topics": item.get("focus_topics", []),
                    "prerequisites": item.get("prerequisites", []),
                    "allocated_minutes": item.get("allocated_minutes"),
                    "chunk_progress_pct_start": item.get("chunk_progress_pct_start"),
                    "chunk_progress_pct_end": item.get("chunk_progress_pct_end"),
                }
                for item in slot.get("items", [])
            ],
            "prerequisites": slot.get("prerequisites", []),
        })

    return {
        "coverage": {
            "coverage_pct": coverage.get("coverage_pct"),
            "covered_chunks": coverage.get("covered_chunks"),
            "partial_chunks": coverage.get("partially_covered_chunks"),
            "uncovered_chunks": coverage.get("uncovered_chunks"),
            "required_minutes": coverage.get("total_required_minutes"),
            "planned_minutes": coverage.get("total_planned_minutes"),
            "uncovered_sample": coverage.get("uncovered", [])[:8],
        },
        "slots": preview_slots,
        "slot_overrides": _normalize_slot_overrides(dict(constraints or {}).get("slot_overrides", [])),
    }


def _apply_qwen_escalation_if_needed(
    qwen_feedback,
    constraints,
    locked_constraint_keys,
    chunks,
    start_date,
    end_date,
    schedule_end,
    tzinfo,
    calendar_service,
    calendar_id,
):
    severity = _normalize_int(dict(qwen_feedback or {}).get("severity"), 0, minimum=0, maximum=3)
    if severity < 3:
        return {
            "constraints": constraints,
            "slots": None,
            "coverage": None,
            "qwen_feedback": qwen_feedback,
            "schedule_end": schedule_end,
            "escalated": False,
        }

    _planner_log("[Planner] Qwen severity=3. Resetting constraints and rebuilding from defaults...")
    reset_constraints = _reset_constraints_for_severity(constraints, locked_constraint_keys)
    buffer_days = _normalize_int(reset_constraints.get("buffer_days"), 1, minimum=0, maximum=14)
    reset_schedule_end = end_date - timedelta(days=buffer_days)
    if reset_schedule_end < start_date:
        reset_schedule_end = start_date

    schedule_data = build_schedule_data(
        chunks=chunks,
        constraints=reset_constraints,
        start_date=start_date,
        end_date=end_date,
        schedule_end=reset_schedule_end,
        tzinfo=tzinfo,
        calendar_service=calendar_service,
        calendar_id=calendar_id,
    )

    escalated_constraints = dict(schedule_data.get("constraints", reset_constraints))
    escalated_slots = schedule_data.get("slots", [])
    escalated_coverage = schedule_data.get("coverage", {})

    review_payload = {
        "escalation": "severity_3_reset",
        "plan": _plan_snapshot(slots=escalated_slots, coverage=escalated_coverage, constraints=escalated_constraints),
    }
    escalated_qwen = qwen_review_plan(review_payload)

    return {
        "constraints": escalated_constraints,
        "slots": escalated_slots,
        "coverage": escalated_coverage,
        "qwen_feedback": escalated_qwen,
        "schedule_end": reset_schedule_end,
        "escalated": True,
    }


def _qwen_flags_severe_revision_needed(qwen_feedback):
    if not isinstance(qwen_feedback, dict):
        return False

    severity_markers = (
        "very poor",
        "poor",
        "unacceptable",
        "critical",
        "severe",
        "major issue",
        "major issues",
        "high risk",
        "fails",
        "failure",
    )

    fields = [
        qwen_feedback.get("summary", ""),
        " ".join(qwen_feedback.get("risks", [])),
        " ".join(qwen_feedback.get("suggested_adjustments", [])),
    ]
    combined = " ".join(str(field or "").lower() for field in fields)
    return any(marker in combined for marker in severity_markers)


def _post_review_refinement_reasons(qwen_feedback, coverage):
    reasons = []

    uncovered = _normalize_int((coverage or {}).get("uncovered_chunks"), 0, minimum=0)
    partial = _normalize_int((coverage or {}).get("partially_covered_chunks"), 0, minimum=0)
    if uncovered > 0 or partial > 0:
        reasons.append(f"coverage gaps remain (uncovered={uncovered}, partial={partial})")

    severity = _normalize_int((qwen_feedback or {}).get("severity"), 0, minimum=0, maximum=3)
    if severity >= 2:
        reasons.append(f"Qwen severity is high (severity={severity})")

    if _qwen_flags_severe_revision_needed(qwen_feedback):
        reasons.append("Qwen summary/risks suggest major or critical issues")

    top_risks = _normalize_text_list((qwen_feedback or {}).get("risks", []))[:3]
    if top_risks:
        reasons.append("Top Qwen risks: " + " | ".join(top_risks))

    return reasons


def _needs_post_review_refinement(qwen_feedback, coverage):
    return bool(_post_review_refinement_reasons(qwen_feedback, coverage))
